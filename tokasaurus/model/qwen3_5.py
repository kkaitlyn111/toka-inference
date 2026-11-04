"""Qwen 3.5 (e.g. Qwen/Qwen3.5-9B): a hybrid of gated softmax attention and gated
DeltaNet linear attention.

Every fourth layer is softmax attention over the paged KV cache; the rest are gated
delta-rule layers whose state is a fixed-size recurrent tensor per sequence. That state
lives in a flat slot table indexed by the sequence's batch index (see the state-slot
fields on AttentionInfo), so it costs O(1) memory per sequence regardless of length.

Numerics follow the transformers reference:
  - norms are x / rms(x) * (1 + w) in fp32, except the DeltaNet output norm (plain w);
  - q_proj emits 2 * head_dim per head, split head-major into query and a sigmoid gate
    applied to the attention output before o_proj;
  - partial rope (first 64 of 256 dims), with cos/sin cast to the activation dtype.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tokasaurus.model.attention_utils import create_wrappers, tokasaurus_attention
from tokasaurus.model.gated_deltanet import (
    RMSNormGated,
    causal_conv1d,
    causal_conv1d_step,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from tokasaurus.model.glm4_moe import apply_partial_rotary_pos_emb
from tokasaurus.model.kv_cache import LayerKVCache
from tokasaurus.model.llama import (
    LlamaForCausalLM,
    LlamaLMHead,
    calc_pipeline_layer_start_and_end,
)
from tokasaurus.model.qwen3_5_config import Qwen35Config, register_qwen3_5_config
from tokasaurus.model.types import AttentionInfo, BatchState, ExtraModelConfig

register_qwen3_5_config()


class Qwen35RMSNorm(nn.Module):
    """x / rms(x) * (1 + w), computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class Qwen35Attention(nn.Module):
    def __init__(
        self, config: Qwen35Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        hs, hd = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(hs, self.num_attention_heads * hd * 2, bias=False)
        self.k_proj = nn.Linear(hs, self.num_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(hs, self.num_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * hd, hs, bias=False)
        self.q_norm = Qwen35RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(hd, eps=config.rms_norm_eps)

        self.layer_cache: LayerKVCache | None = None
        self.wrapper_collection = None
        self.attention_info: AttentionInfo | None = None
        self.attn_fn = self.make_attn_fn()

    def make_attn_fn(self):
        @torch.library.custom_op(
            f"tokasaurus::qwen3_5_attention_layer_{self.layer_idx}",
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

    def forward(self, hidden_states: Tensor, batch_state: BatchState) -> Tensor:
        assert self.layer_cache is not None and self.layer_cache.v_cache is not None
        assert batch_state.position_embeddings is not None
        num_tokens = hidden_states.shape[0]
        hd = self.head_dim

        q, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                num_tokens, self.num_attention_heads, hd * 2
            ),
            2,
            dim=-1,
        )
        q = self.q_norm(q)
        k = self.k_norm(
            self.k_proj(hidden_states).view(num_tokens, self.num_kv_heads, hd)
        )
        v = self.v_proj(hidden_states).view(num_tokens, self.num_kv_heads, hd)

        dtype = q.dtype
        cos, sin = batch_state.position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin)

        attn_output = self.attn_fn(
            ragged_q=q.to(dtype),
            ragged_k=k.to(dtype),
            ragged_v=v,
            k_cache=self.layer_cache.k_cache,
            v_cache=self.layer_cache.v_cache,
        ).clone()  # same torch.compile workaround as llama.py

        attn_output = attn_output.view(num_tokens, -1) * torch.sigmoid(
            gate.reshape(num_tokens, -1)
        )
        return self.o_proj(attn_output)


class Qwen35DeltaNet(nn.Module):
    """Gated delta-rule layer whose recurrent state lives in per-sequence slots.

    conv_cache is [slots, conv_dim, kernel] in the model dtype; ssm_cache is
    [slots, v_heads, k_head_dim, v_head_dim] in fp32. Both are allocated by setup_caches.
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        self.conv_dim = config.linear_conv_dim
        self.kv_group = self.num_v_heads // self.num_k_heads

        hs = config.hidden_size
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=config.linear_conv_kernel_dim,
            groups=self.conv_dim,
            bias=False,
            padding=config.linear_conv_kernel_dim - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hs, bias=False)
        self.in_proj_qkv = nn.Linear(hs, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hs, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hs, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hs, self.num_v_heads, bias=False)

        self.attention_info: AttentionInfo | None = None
        self.conv_cache: Tensor | None = None
        self.ssm_cache: Tensor | None = None
        self.delta_fn = self.make_delta_fn()

    def split_heads(self, mixed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """[B, L, conv_dim] -> q, k [B, L, v_heads, Dk] (k heads repeated), v [B, L, v_heads, Dv]."""
        batch, length, _ = mixed.shape
        q, k, v = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.reshape(batch, length, -1, self.head_k_dim)
        k = k.reshape(batch, length, -1, self.head_k_dim)
        v = v.reshape(batch, length, -1, self.head_v_dim)
        if self.kv_group > 1:
            q = q.repeat_interleave(self.kv_group, dim=2)
            k = k.repeat_interleave(self.kv_group, dim=2)
        return q, k, v

    def make_delta_fn(self):
        @torch.library.custom_op(
            f"tokasaurus::qwen3_5_deltanet_layer_{self.layer_idx}",
            mutates_args=("conv_cache", "ssm_cache"),
        )
        def delta_fn(
            mixed_qkv: Tensor,
            b: Tensor,
            a: Tensor,
            conv_weight: Tensor,
            dt_bias: Tensor,
            A_log: Tensor,
            conv_cache: Tensor,
            ssm_cache: Tensor,
        ) -> Tensor:
            info = self.attention_info
            assert info is not None
            assert info.hydragen_info is None or info.hydragen_info.num_tokens == 0

            total_tokens = mixed_qkv.shape[0]
            if (num_padding := info.num_padding) > 0:
                mixed_qkv, b, a = (
                    mixed_qkv[:-num_padding],
                    b[:-num_padding],
                    a[:-num_padding],
                )

            out = mixed_qkv.new_empty((mixed_qkv.shape[0], self.value_dim))
            # fp32: -exp(A_log) underflows in half precision
            neg_a = -A_log.float().exp()
            num_prefill = info.prefill_info.num_tokens

            if num_prefill > 0:
                slots = info.prefill_state_slots
                has_init = info.prefill_has_init_state
                assert slots is not None and has_init is not None
                init_ssm = ssm_cache[slots].clone()
                init_ssm[~has_init] = 0
                init_conv = conv_cache[slots].clone()
                init_conv[~has_init] = 0

                # prefill sequences are ragged and the scan has no varlen form, so run
                # them one at a time; a chunked prefill continues from the stored state
                bounds = info.prefill_info.qo_indptr.tolist()
                for i in range(info.prefill_info.num_seqs):
                    start, end = bounds[i], bounds[i + 1]
                    seq_len = end - start
                    window = torch.cat(
                        [init_conv[i : i + 1], mixed_qkv[start:end].t().unsqueeze(0)],
                        dim=-1,
                    )
                    init_conv[i] = window[0, :, -conv_cache.shape[-1] :]
                    conv_out = causal_conv1d(window, conv_weight)[:, :, -seq_len:]

                    q, k, v = self.split_heads(conv_out.transpose(1, 2))
                    beta = b[start:end].unsqueeze(0).sigmoid()
                    g = neg_a * F.softplus(a[start:end].unsqueeze(0).float() + dt_bias)
                    core, state = chunk_gated_delta_rule(
                        q, k, v, g=g, beta=beta, initial_state=init_ssm[i : i + 1]
                    )
                    init_ssm[i] = state[0]
                    out[start:end] = core.reshape(seq_len, self.value_dim)

                ssm_cache[slots] = init_ssm
                conv_cache[slots] = init_conv

            if info.decode_info.num_tokens > 0:
                # decode tokens follow the prefill tokens, one per sequence
                slots = info.decode_state_slots
                assert slots is not None
                num_decode = info.decode_info.num_tokens

                conv_state = conv_cache[slots].clone()
                conv_out = causal_conv1d_step(
                    mixed_qkv[num_prefill:].unsqueeze(-1), conv_state, conv_weight
                )
                conv_cache[slots] = conv_state

                q, k, v = self.split_heads(conv_out.transpose(1, 2))
                beta = b[num_prefill:].unsqueeze(1).sigmoid()
                g = neg_a * F.softplus(a[num_prefill:].unsqueeze(1).float() + dt_bias)
                core, state = recurrent_gated_delta_rule(
                    q, k, v, g=g, beta=beta, initial_state=ssm_cache[slots]
                )
                ssm_cache[slots] = state
                out[num_prefill:] = core.reshape(num_decode, self.value_dim)

            if num_padding > 0:
                out = F.pad(out, (0, 0, 0, num_padding))
            assert out.shape[0] == total_tokens
            return out

        @delta_fn.register_fake
        def _(
            mixed_qkv, b, a, conv_weight, dt_bias, A_log, conv_cache, ssm_cache
        ) -> Tensor:
            return mixed_qkv.new_empty((mixed_qkv.shape[0], self.value_dim))

        return delta_fn

    def forward(self, hidden_states: Tensor, batch_state: BatchState) -> Tensor:
        assert self.conv_cache is not None and self.ssm_cache is not None
        num_tokens = hidden_states.shape[0]
        core = self.delta_fn(
            self.in_proj_qkv(hidden_states),
            self.in_proj_b(hidden_states),
            self.in_proj_a(hidden_states),
            self.conv1d.weight.squeeze(1),
            self.dt_bias,
            self.A_log,
            self.conv_cache,
            self.ssm_cache,
        ).clone()  # same torch.compile workaround as llama.py

        z = self.in_proj_z(hidden_states)
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(core.reshape(num_tokens, -1))


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        hs, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hs, inter, bias=False)
        self.up_proj = nn.Linear(hs, inter, bias=False)
        self.down_proj = nn.Linear(inter, hs, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35Block(nn.Module):
    def __init__(
        self, config: Qwen35Config, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.is_last = layer_idx == config.num_hidden_layers - 1
        # the mixer is named self_attn for both layer types; the checkpoint calls the
        # DeltaNet one linear_attn (see make_name_to_hf_name)
        if config.is_full_attention(layer_idx):
            self.self_attn: nn.Module = Qwen35Attention(config, extra_config, layer_idx)
        else:
            self.self_attn = Qwen35DeltaNet(config, layer_idx)
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, batch_state: BatchState) -> BatchState:
        hidden = batch_state.hidden_states
        assert hidden is not None
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), batch_state)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        if self.is_last:
            assert batch_state.lm_head_indices is not None
            hidden = hidden[batch_state.lm_head_indices]
        batch_state.hidden_states = hidden
        return batch_state


class Qwen35Embeddings(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState) -> BatchState:
        batch_state.hidden_states = self.embed_tokens(batch_state.input_ids)
        return batch_state


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config, extra_config: ExtraModelConfig):
        super().__init__()
        config.validate_supported()

        layer_start, layer_end = calc_pipeline_layer_start_and_end(config, extra_config)
        self.embed_tokens = (
            Qwen35Embeddings(config) if layer_start == 0 else nn.Identity()
        )
        if layer_end is None:
            layer_end = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Qwen35Block(config, extra_config, i)
                if layer_start <= i < layer_end
                else nn.Identity()
                for i in range(config.num_hidden_layers)
            ]
        )

        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(self, batch_state: BatchState) -> BatchState:
        out: BatchState = self.embed_tokens(batch_state)
        assert out.hidden_states is not None
        freqs = batch_state.position_ids.float()[:, None] * self.rope_inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        # HF casts cos/sin to the activation dtype before applying them
        dtype = out.hidden_states.dtype
        out.position_embeddings = (emb.cos().to(dtype), emb.sin().to(dtype))
        for layer in self.layers:
            out = layer(out)
        return out


class Qwen35LMHead(LlamaLMHead):
    def __init__(self, config: Qwen35Config, extra_config: ExtraModelConfig):
        super().__init__(config, extra_config)
        self.input_norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen35ForCausalLM(LlamaForCausalLM):
    model_cls = Qwen35Model
    config_cls = Qwen35Config

    def __init__(self, config: Qwen35Config, extra_config: ExtraModelConfig):
        if (extra_config.tp_size or 1) > 1:
            raise NotImplementedError("qwen3_5 does not support tensor parallelism yet")
        nn.Module.__init__(self)
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self.model = self.model_cls(config, extra_config)

        _, layer_end = calc_pipeline_layer_start_and_end(config, extra_config)
        if layer_end in [None, config.num_hidden_layers]:
            self.lm_head: nn.Module = Qwen35LMHead(config, extra_config)
        else:
            self.lm_head = nn.Identity()

    def num_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    def num_qo_heads(self) -> int:
        return self.config.num_attention_heads

    def head_dim(self) -> int:
        return self.config.head_dim

    def mixers(self, cls):
        for module in self.model.modules():
            if isinstance(module, cls):
                yield module

    def set_wrappers(self, wrappers):
        self.wrapper_collection = wrappers
        for layer in self.mixers(Qwen35Attention):
            layer.wrapper_collection = wrappers

    def set_attention_info(self, attn_info: AttentionInfo):
        for layer in self.mixers((Qwen35Attention, Qwen35DeltaNet)):
            layer.attention_info = attn_info

    def setup_caches(self, num_pages: int, page_size: int):
        self.set_wrappers(
            create_wrappers(
                device=self.device,
                num_attention_heads=self.num_qo_heads(),
                num_key_value_heads=self.num_kv_heads(),
            )
        )
        for layer in self.mixers(Qwen35Attention):
            layer.layer_cache = LayerKVCache(
                head_dim=self.head_dim(),
                num_kv_heads=layer.num_kv_heads,
                num_pages=num_pages,
                page_size=page_size,
                device=self.device,
                dtype=self.dtype,
            )

        # one recurrent-state slot per batch index the manager can hand out
        num_slots = self.extra_config.max_seqs_per_forward
        config = self.config
        for layer in self.mixers(Qwen35DeltaNet):
            layer.conv_cache = torch.zeros(
                num_slots,
                config.linear_conv_dim,
                config.linear_conv_kernel_dim,
                device=self.device,
                dtype=self.dtype,
            )
            layer.ssm_cache = torch.zeros(
                num_slots,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                device=self.device,
                dtype=torch.float32,
            )

    def make_name_to_hf_name(self):
        # LM weights sit under model.language_model.; lm_head.weight is top-level (untied)
        prefix = "model.language_model."
        name_to_hf_name = {}
        for key in self.state_dict():
            if key.startswith("model.layers."):
                rest = key[len("model.") :]
                if not self.config.is_full_attention(int(rest.split(".")[1])):
                    rest = rest.replace(".self_attn.", ".linear_attn.", 1)
                name_to_hf_name[key] = prefix + rest
            else:
                name_to_hf_name[key] = key

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            prefix + "embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = prefix + "norm.weight"
        name_to_hf_name["lm_head.lm_head.weight"] = (
            prefix + "embed_tokens.weight"
            if self.config.tie_word_embeddings
            else "lm_head.weight"
        )
        return name_to_hf_name
