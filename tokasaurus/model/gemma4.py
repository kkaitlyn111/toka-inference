"""Gemma 4 (e.g. google/gemma-4-31B-it), ported natively; no transformers-5 dependency.

Architecture, as pinned against the transformers 5.x reference:

  - 5:1 sliding (window 1024) : full attention layers, per `layer_types`
  - per-layer-type geometry: sliding 16 kv heads x 256, full 4 kv heads x 512
  - attention scaling 1.0 (no 1/sqrt(d))
  - full layers reuse the k projection for V (`attention_k_eq_v`):
    K = rope(k_norm(k_proj(x))), V = v_norm(k_proj(x))
  - scale-free v_norm on every layer
  - per-type rope: sliding default (theta 10k); full "proportional" (theta 1M,
    partial_rotary_factor 0.25, i.e. zero-padded inv_freq)
  - sandwich norms, and a trained per-layer `layer_scalar` on the whole layer output
  - embeddings scaled by bf16(sqrt(hidden_size)); tied lm_head; final softcap 30

flashinfer's prefill kernels reject head_dim 512, so full-layer prefill uses a dense
fallback; full-layer decode uses flashinfer's CUDA-core decode kernel, which accepts it.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from tokasaurus.model.attention_utils import (
    append_to_kv_cache,
    create_workspace_buffer,
    tokasaurus_attention,
)
from tokasaurus.model.attn_backend import BatchDecodeWithPagedKVCacheWrapper
from tokasaurus.model.gemma4_config import Gemma4Config, register_gemma4_config
from tokasaurus.model.kv_cache import LayerKVCache
from tokasaurus.model.layer_types import FULL_ATTENTION, LAYER_TYPES, SLIDING_ATTENTION
from tokasaurus.model.llama import (
    all_gather,
    calc_pipeline_layer_start_and_end,
    reduce_scatter,
)
from tokasaurus.model.sliding_window import SlidingWindowForCausalLM, SoftcapLMHead
from tokasaurus.model.types import AttentionInfo, BatchState, ExtraModelConfig

register_gemma4_config()

# largest head_dim flashinfer's paged prefill kernels accept
FLASHINFER_MAX_PREFILL_HEAD_DIM = 256
# queries per chunk in the dense prefill fallback; a full 5k x 5k fp32 score tensor
# per layer does not fit on an 80GB card
DENSE_PREFILL_Q_CHUNK = 512


def dense_paged_prefill(
    prefill_q: Tensor, k_cache: Tensor, v_cache: Tensor, attn_info: AttentionInfo
) -> Tensor:
    """Causal fp32 prefill attention over the paged cache (scaling 1.0, no window)."""
    page_size = attn_info.page_size
    _, _, num_kv_heads, head_dim = k_cache.shape
    out = torch.empty_like(prefill_q)
    rep = prefill_q.shape[1] // num_kv_heads

    pre = attn_info.prefill_info
    qo_indptr = pre.qo_indptr.tolist()
    kv_indptr = pre.kv_indptr.tolist()
    last_lens = pre.kv_last_page_len.tolist()
    for s in range(pre.num_seqs):
        q0, q1 = qo_indptr[s], qo_indptr[s + 1]
        pages = pre.kv_indices[kv_indptr[s] : kv_indptr[s + 1]]
        kv_len = (len(pages) - 1) * page_size + last_lens[s]
        k = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        v = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        kpos = torch.arange(kv_len, device=k.device)[None, :]
        for c0 in range(0, q1 - q0, DENSE_PREFILL_Q_CHUNK):
            c1 = min(c0 + DENSE_PREFILL_Q_CHUNK, q1 - q0)
            q = prefill_q[q0 + c0 : q0 + c1].float()
            scores = torch.einsum("qhd,khd->hqk", q, k)
            qpos = torch.arange(c0, c1, device=q.device)[:, None] + (kv_len - (q1 - q0))
            scores = scores.masked_fill(kpos[None] > qpos[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q0 + c0 : q0 + c1] = torch.einsum("hqk,khd->qhd", probs, v).to(
                prefill_q.dtype
            )
    return out


class Gemma4RMSNorm(nn.Module):
    """HF Gemma4RMSNorm: fp32 norm and scale multiply, then cast back; optionally scale-free."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states.float()
        out = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(hidden_states.dtype)


def rope_inv_freq(config: Gemma4Config, layer_type: str) -> Tensor:
    params = config.rope_parameters[layer_type]
    head_dim = config.head_dim_for_type(layer_type)
    theta = params["rope_theta"]
    rope_type = params.get("rope_type", "default")

    if rope_type == "default":
        return 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
    if rope_type == "proportional":
        # frequencies of a full-width rope, truncated to the rotated angles; the rest
        # of each rotate_half half is identity
        rope_angles = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim)
        )
        return torch.cat([inv_freq, torch.zeros(head_dim // 2 - rope_angles)])
    raise NotImplementedError(f"rope_type {rope_type}")


class Gemma4Attention(nn.Module):
    """x + post_attention_layernorm(attn(input_layernorm(x)))."""

    def __init__(
        self, config: Gemma4Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.extra_config = extra_config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        is_sliding = self.layer_type == SLIDING_ATTENTION

        tp_size = extra_config.tp_size or 1
        assert config.num_attention_heads % tp_size == 0
        assert config.kv_heads_for_layer(layer_idx) % tp_size == 0
        self.num_attention_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.kv_heads_for_layer(layer_idx) // tp_size
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.dense_prefill = (
            not is_sliding and self.head_dim > FLASHINFER_MAX_PREFILL_HEAD_DIM
        )
        k_eq_v = config.attention_k_eq_v and not is_sliding

        hs = config.hidden_size
        self.q_proj = nn.Linear(
            hs, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(hs, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = (
            None
            if k_eq_v
            else nn.Linear(hs, self.num_kv_heads * self.head_dim, bias=False)
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, hs, bias=False
        )

        eps = config.rms_norm_eps
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps, with_scale=False)
        self.input_layernorm = Gemma4RMSNorm(hs, eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hs, eps)

        self.layer_cache: LayerKVCache | None = None
        self.wrapper_collection = None
        self.attention_info: AttentionInfo | None = None
        self.attn_fn = self.make_attn_fn()

    def run_dense_prefill(self, q, k, v, k_cache, v_cache, attn_info: AttentionInfo):
        append_to_kv_cache(
            token_indices=attn_info.append_kv_token_indices,
            key=k,
            value=v,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        num_prefill = attn_info.prefill_info.num_tokens
        parts = []
        if num_prefill > 0:
            parts.append(
                dense_paged_prefill(q[:num_prefill], k_cache, v_cache, attn_info)
            )
        if attn_info.decode_info.num_tokens > 0:
            parts.append(
                self.wrapper_collection.decode_wrapper.run(
                    q=q[num_prefill:], paged_kv_cache=(k_cache, v_cache)
                )
            )
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def make_attn_fn(self):
        @torch.library.custom_op(
            f"tokasaurus::gemma4_attention_layer_{self.layer_idx}",
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

            if self.dense_prefill:
                out = self.run_dense_prefill(
                    ragged_q, ragged_k, ragged_v, k_cache, v_cache, attn_info
                )
            else:
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
        assert batch_state.position_embeddings is not None
        assert self.layer_cache is not None and self.layer_cache.v_cache is not None

        residual = batch_state.hidden_states
        hidden_states = all_gather(self.input_layernorm(residual), self.extra_config)
        num_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states).view(
            num_tokens, self.num_attention_heads, self.head_dim
        )
        k_raw = self.k_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_raw = (
            k_raw
            if self.v_proj is None
            else self.v_proj(hidden_states).view(
                num_tokens, self.num_kv_heads, self.head_dim
            )
        )

        # per-head norms before rope (HF order); v gets no rope
        q = self.q_norm(q)
        k = self.k_norm(k_raw)
        v = self.v_norm(v_raw)

        cos, sin = batch_state.position_embeddings[self.layer_type]
        dtype = q.dtype
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        attn_output = self.attn_fn(
            ragged_q=q.to(dtype),
            ragged_k=k.to(dtype),
            ragged_v=v.to(dtype),
            k_cache=self.layer_cache.k_cache,
            v_cache=self.layer_cache.v_cache,
        ).clone()  # same torch.compile workaround as llama.py

        o_proj = reduce_scatter(
            self.o_proj(attn_output.view(num_tokens, -1)), self.extra_config
        )
        batch_state.hidden_states = residual + self.post_attention_layernorm(o_proj)
        return batch_state


class Gemma4MLP(nn.Module):
    """x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x))), tanh-GELU gated."""

    def __init__(self, config: Gemma4Config, extra_config: ExtraModelConfig):
        super().__init__()
        self.extra_config = extra_config
        tp_size = extra_config.tp_size or 1
        assert config.intermediate_size % tp_size == 0
        inter = config.intermediate_size // tp_size

        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, batch_state: BatchState) -> BatchState:
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = all_gather(
            self.pre_feedforward_layernorm(inp), self.extra_config
        )
        gate = self.gate_proj(hidden_states)
        down = self.down_proj(
            F.gelu(gate, approximate="tanh") * self.up_proj(hidden_states)
        )
        down = reduce_scatter(down, self.extra_config)
        batch_state.hidden_states = inp + self.post_feedforward_layernorm(down)
        return batch_state


class Gemma4Block(nn.Module):
    def __init__(
        self, config: Gemma4Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.extra_config = extra_config
        self.self_attn = Gemma4Attention(config, extra_config, layer_idx)
        self.mlp = Gemma4MLP(config, extra_config)
        # trained values (~0.04 to 0.99 in the 31B checkpoint), loaded from the checkpoint
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.is_last = layer_idx == config.num_hidden_layers - 1

    def forward(self, batch_state: BatchState) -> BatchState:
        out = self.mlp(self.self_attn(batch_state))
        assert out.hidden_states is not None
        out.hidden_states = out.hidden_states * self.layer_scalar

        # keep only the positions the lm head needs (llama.py does this before o_proj;
        # after the block is equivalent and keeps the layer_scalar math simple)
        if self.is_last and (self.extra_config.tp_size or 1) == 1:
            assert out.lm_head_indices is not None
            out.hidden_states = out.hidden_states[out.lm_head_indices]
        return out


class Gemma4Embeddings(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState) -> BatchState:
        emb = self.embed_tokens(batch_state.input_ids)
        # HF casts the scale to the weight dtype first (bf16: sqrt(5376)=73.32 -> 73.5)
        scale = torch.tensor(self.config.hidden_size**0.5, dtype=torch.float32).to(
            emb.dtype
        )
        batch_state.hidden_states = emb * scale
        return batch_state


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config, extra_config: ExtraModelConfig):
        super().__init__()
        config.validate_supported()

        layer_start, layer_end = calc_pipeline_layer_start_and_end(config, extra_config)
        self.embed_tokens = (
            Gemma4Embeddings(config) if layer_start == 0 else nn.Identity()
        )
        if layer_end is None:
            layer_end = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Gemma4Block(config, extra_config, i)
                if layer_start <= i < layer_end
                else nn.Identity()
                for i in range(config.num_hidden_layers)
            ]
        )

        # cos/sin are computed per batch: a table over 262K positions would cost GBs
        for layer_type in LAYER_TYPES:
            self.register_buffer(
                f"rope_inv_freq_{layer_type}",
                rope_inv_freq(config, layer_type),
                persistent=False,
            )

    def forward(self, batch_state: BatchState) -> BatchState:
        out: BatchState = self.embed_tokens(batch_state)
        pos = batch_state.position_ids.float()
        position_embeddings = {}
        for layer_type in LAYER_TYPES:
            freqs = pos[:, None] * getattr(self, f"rope_inv_freq_{layer_type}")[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            position_embeddings[layer_type] = (emb.cos(), emb.sin())
        out.position_embeddings = position_embeddings

        for layer in self.layers:
            out = layer(out)
        return out


class Gemma4ForCausalLM(SlidingWindowForCausalLM):
    model_cls = Gemma4Model
    config_cls = Gemma4Config
    lm_head_cls = SoftcapLMHead
    attention_cls = Gemma4Attention

    def kv_heads_for_type(self, layer_type: str) -> int:
        return self.config.kv_heads_for_type(layer_type)

    def head_dim_for_type(self, layer_type: str) -> int:
        return self.config.head_dim_for_type(layer_type)

    def sm_scale(self) -> float:
        return 1.0

    def uses_flashinfer_prefill(self, layer_type: str) -> bool:
        return self.head_dim_for_type(layer_type) <= FLASHINFER_MAX_PREFILL_HEAD_DIM

    def make_layer_wrappers(self) -> dict:
        wrappers = super().make_layer_wrappers()
        if not self.uses_flashinfer_prefill(FULL_ATTENTION):
            # tensor-core decode reuses the prefill kernel, which rejects head_dim 512;
            # the CUDA-core decode kernel accepts it
            wrappers[
                FULL_ATTENTION
            ].decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                create_workspace_buffer(self.device), use_tensor_cores=False
            )
        return wrappers

    def make_name_to_hf_name(self):
        assert self.config.tie_word_embeddings
        return super().make_name_to_hf_name()
