"""Muse Glimmer (meta-models/Muse-Glimmer-30B) text stack, ported natively.

Pinned to the transformers 5.15 reference (the checkpoint ships no remote code):

  - [sliding x3, full] pattern: 13 full / 39 sliding layers, window 2048
  - uniform geometry on every layer (32 q / 2 kv heads, head_dim 128), so both layer
    types share one KV cache shape and differ only in `window_left`
  - full-attention layers are NoPE; sliding layers share one rope (theta 500k)
  - q and k share one scale-free RMSNorm over head_dim, then q alone is scaled by
    `qk_scale_factor`; no v_norm. Order: norm, scale, rope
  - attention scaling head_dim ** -0.5
  - gated output: attn_out * sigmoid(gate_proj(x)) before o_proj, where x is the
    attention block input (after input_layernorm), not the attention output
  - centered sandwich norms (scale 1 + w) at two epsilons: pre_* rms_norm_eps (1e-5),
    post_* post_norm_eps (1e-8); the final norm is a plain `* w` RMSNorm
  - embeddings followed by a scale-free RMSNorm; untied lm_head
  - logits: lm_head(h) * output_multiplier, then tanh softcap 20

The vision tower is not loaded; only the text stack is served.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from tokasaurus.model.attention_utils import tokasaurus_attention
from tokasaurus.model.kv_cache import LayerKVCache
from tokasaurus.model.llama import (
    all_gather,
    calc_pipeline_layer_start_and_end,
    reduce_scatter,
)
from tokasaurus.model.muse_glimmer_config import (
    MuseGlimmerConfig,
    register_muse_glimmer_config,
)
from tokasaurus.model.sliding_window import SlidingWindowForCausalLM, SoftcapLMHead
from tokasaurus.model.types import AttentionInfo, BatchState, ExtraModelConfig

register_muse_glimmer_config()


class MuseGlimmerRMSNorm(nn.Module):
    """HF MuseGlimmerRMSNorm: fp32 norm (pow(-0.5), as in the reference), plain `* w`."""

    def __init__(self, dim: int | None, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states.float()
        out = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(hidden_states.dtype)


class MuseGlimmerCenteredRMSNorm(nn.Module):
    """HF MuseGlimmerTextCenteredRMSNorm: scale 1 + w (w initialized to 0), via rsqrt."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).to(hidden_states.dtype)


class MuseGlimmerAttention(nn.Module):
    """x + post_attention_layernorm(gated_attn(input_layernorm(x)))."""

    def __init__(
        self, config: MuseGlimmerConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.extra_config = extra_config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.is_nope = config.is_nope(layer_idx)

        tp_size = extra_config.tp_size or 1
        assert config.num_attention_heads % tp_size == 0
        assert config.num_key_value_heads % tp_size == 0, (
            f"tp_size={tp_size} must divide num_key_value_heads={config.num_key_value_heads}"
        )
        self.num_attention_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.qk_scale_factor = config.qk_scale_factor

        hs = config.hidden_size
        q_out = self.num_attention_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hs, q_out, bias=False)
        self.k_proj = nn.Linear(hs, kv_out, bias=False)
        self.v_proj = nn.Linear(hs, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hs, bias=False)
        self.gate_proj = nn.Linear(hs, q_out, bias=False)

        self.qk_norm = MuseGlimmerRMSNorm(None, config.rms_norm_eps, with_scale=False)
        self.input_layernorm = MuseGlimmerCenteredRMSNorm(hs, config.rms_norm_eps)
        self.post_attention_layernorm = MuseGlimmerCenteredRMSNorm(
            hs, config.post_norm_eps
        )

        self.layer_cache: LayerKVCache | None = None
        self.wrapper_collection = None
        self.attention_info: AttentionInfo | None = None
        self.attn_fn = self.make_attn_fn()

    def make_attn_fn(self):
        @torch.library.custom_op(
            f"tokasaurus::muse_glimmer_attention_layer_{self.layer_idx}",
            mutates_args=("k_cache", "v_cache"),
        )
        def attn_fn(
            ragged_q: Tensor,
            ragged_k: Tensor,
            ragged_v: Tensor,
            k_cache: Tensor,
            v_cache: Tensor,
        ) -> Tensor:
            attn_info = self.attention_info
            assert attn_info is not None and self.wrapper_collection is not None

            orig_q = ragged_q
            if (num_padding := attn_info.num_padding) > 0:
                ragged_q = ragged_q[:-num_padding]
                ragged_k = ragged_k[:-num_padding]
                ragged_v = ragged_v[:-num_padding]

            out = tokasaurus_attention(
                ragged_q=ragged_q,
                ragged_k=ragged_k,
                ragged_v=ragged_v,
                k_cache=k_cache,
                v_cache=v_cache,
                attn_info=attn_info,
                wrappers=self.wrapper_collection,
            )

            if num_padding > 0:
                out = F.pad(out, (0, 0, 0, 0, 0, num_padding))
            assert out.shape == orig_q.shape
            return out

        @attn_fn.register_fake
        def _(ragged_q, ragged_k, ragged_v, k_cache, v_cache) -> Tensor:
            return torch.empty_like(ragged_q)

        return attn_fn

    def forward(self, batch_state: BatchState) -> BatchState:
        assert batch_state.hidden_states is not None
        assert self.layer_cache is not None and self.layer_cache.v_cache is not None

        residual = batch_state.hidden_states
        hidden_states = all_gather(self.input_layernorm(residual), self.extra_config)
        num_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states).view(
            num_tokens, self.num_attention_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        dtype = q.dtype
        q = self.qk_norm(q) * self.qk_scale_factor
        k = self.qk_norm(k)
        if not self.is_nope:
            assert batch_state.position_embeddings is not None
            cos, sin = batch_state.position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        attn_output = self.attn_fn(
            ragged_q=q.to(dtype),
            ragged_k=k.to(dtype),
            ragged_v=v.to(dtype),
            k_cache=self.layer_cache.k_cache,
            v_cache=self.layer_cache.v_cache,
        ).clone()  # same torch.compile workaround as llama.py

        # the gate reads the block input, not the attention output
        attn_output = attn_output.view(num_tokens, -1) * torch.sigmoid(
            self.gate_proj(hidden_states)
        )
        o_proj = reduce_scatter(self.o_proj(attn_output), self.extra_config)
        batch_state.hidden_states = residual + self.post_attention_layernorm(o_proj)
        return batch_state


class MuseGlimmerMLP(nn.Module):
    """x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x))), SiLU gated."""

    def __init__(self, config: MuseGlimmerConfig, extra_config: ExtraModelConfig):
        super().__init__()
        assert config.hidden_activation == "silu", config.hidden_activation
        self.extra_config = extra_config
        tp_size = extra_config.tp_size or 1
        assert config.intermediate_size % tp_size == 0
        inter = config.intermediate_size // tp_size

        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size, config.post_norm_eps
        )

    def forward(self, batch_state: BatchState) -> BatchState:
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = all_gather(
            self.pre_feedforward_layernorm(inp), self.extra_config
        )
        down = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        down = reduce_scatter(down, self.extra_config)
        batch_state.hidden_states = inp + self.post_feedforward_layernorm(down)
        return batch_state


class MuseGlimmerBlock(nn.Module):
    def __init__(
        self, config: MuseGlimmerConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.extra_config = extra_config
        self.self_attn = MuseGlimmerAttention(config, extra_config, layer_idx)
        self.mlp = MuseGlimmerMLP(config, extra_config)
        self.is_last = layer_idx == config.num_hidden_layers - 1

    def forward(self, batch_state: BatchState) -> BatchState:
        out = self.mlp(self.self_attn(batch_state))
        if self.is_last and (self.extra_config.tp_size or 1) == 1:
            assert out.lm_head_indices is not None
            out.hidden_states = out.hidden_states[out.lm_head_indices]
        return out


class MuseGlimmerEmbeddings(nn.Module):
    """Embedding lookup followed by a scale-free RMSNorm."""

    def __init__(self, config: MuseGlimmerConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = MuseGlimmerRMSNorm(
            None, config.rms_norm_eps, with_scale=False
        )

    def forward(self, batch_state: BatchState) -> BatchState:
        batch_state.hidden_states = self.embed_norm(
            self.embed_tokens(batch_state.input_ids)
        )
        return batch_state


class MuseGlimmerModel(nn.Module):
    def __init__(self, config: MuseGlimmerConfig, extra_config: ExtraModelConfig):
        super().__init__()
        config.validate_supported()

        layer_start, layer_end = calc_pipeline_layer_start_and_end(config, extra_config)
        self.embed_tokens = (
            MuseGlimmerEmbeddings(config) if layer_start == 0 else nn.Identity()
        )
        if layer_end is None:
            layer_end = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                MuseGlimmerBlock(config, extra_config, i)
                if layer_start <= i < layer_end
                else nn.Identity()
                for i in range(config.num_hidden_layers)
            ]
        )

        head_dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(self, batch_state: BatchState) -> BatchState:
        out: BatchState = self.embed_tokens(batch_state)
        # one rope for every rotary layer; NoPE layers ignore it
        freqs = batch_state.position_ids.float()[:, None] * self.rope_inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        out.position_embeddings = (emb.cos(), emb.sin())
        for layer in self.layers:
            out = layer(out)
        return out


class MuseGlimmerLMHead(SoftcapLMHead):
    def __init__(self, config: MuseGlimmerConfig, extra_config: ExtraModelConfig):
        super().__init__(config, extra_config)
        self.logit_multiplier = config.output_multiplier
        # tokasaurus' RMSNorm computes `weight * x.to(bf16)`; HF multiplies in fp32 and
        # casts after, so use the HF formula for the final norm
        self.input_norm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps)


class MuseGlimmerForCausalLM(SlidingWindowForCausalLM):
    model_cls = MuseGlimmerModel
    config_cls = MuseGlimmerConfig
    lm_head_cls = MuseGlimmerLMHead
    attention_cls = MuseGlimmerAttention

    def make_name_to_hf_name(self):
        assert not self.config.tie_word_embeddings
        return super().make_name_to_hf_name()
