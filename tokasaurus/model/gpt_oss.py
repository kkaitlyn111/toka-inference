"""gpt-oss (openai/gpt-oss-20b, -120b): MoE with attention sinks and alternating
sliding-window (128) / full attention layers.

  - attention: q/k/v/o biases; a learned per-head sink logit joins each softmax as a
    virtual key with zero value, i.e. out *= Z / (Z + exp(sink)), Z the softmax
    denominator, applied here from the attention LSE;
  - rope: YaRN without truncation of the correction range;
  - MoE: router top-k, then softmax over the selected logits; experts are clamped
    SwiGLU with interleaved gate/up columns and biases;
  - experts ship as MXFP4 (`*_blocks` + `*_scales`) and are dequantized at load.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from tokasaurus.model.attention_utils import append_to_kv_cache
from tokasaurus.model.gpt_oss_config import GptOssConfig, register_gpt_oss_config
from tokasaurus.model.kv_cache import LayerKVCache
from tokasaurus.model.llama import calc_pipeline_layer_start_and_end
from tokasaurus.model.safetensors_utils import load_safetensors_repo
from tokasaurus.model.sliding_window import SlidingWindowForCausalLM, SoftcapLMHead
from tokasaurus.model.types import AttentionInfo, BatchState, ExtraModelConfig

register_gpt_oss_config()

LOG2_E = 1.4426950408889634
# clamp and gate slope of gpt-oss's SwiGLU
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
# MXFP4: e2m1 values indexed by the 4-bit code; scales are e8m0 with this bias
FP4_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
MXFP4_SCALE_BIAS = 127


def dequantize_mxfp4(blocks: Tensor, scales: Tensor, dtype: torch.dtype) -> Tensor:
    """[E, rows, G, 16] uint8 blocks + [E, rows, G] scales -> [E, G * 32, rows].

    Each byte packs two 4-bit codes (low nibble first) and every 32 values share one
    power-of-two scale. The result is transposed into the [E, in, out] expert layout.
    """
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    *prefix, groups, block_bytes = blocks.shape
    blocks = blocks.to(torch.uint8).reshape(-1, block_bytes)
    exponents = (scales.to(torch.int32) - MXFP4_SCALE_BIAS).reshape(-1, 1)

    out = torch.empty(
        blocks.shape[0], block_bytes * 2, dtype=dtype, device=blocks.device
    )
    out[:, 0::2] = lut[(blocks & 0x0F).int()]
    out[:, 1::2] = lut[(blocks >> 4).int()]
    out = torch.ldexp(out, exponents)
    return out.reshape(*prefix, groups * block_bytes * 2).transpose(1, 2).contiguous()


def yarn_inv_freq(config: GptOssConfig) -> tuple[Tensor, float]:
    """YaRN inverse frequencies and cos/sin scale (transformers' _compute_yarn_parameters)."""
    rope = config.rope
    base, factor, dim = rope["rope_theta"], rope["factor"], config.head_dim
    original_max = rope["original_max_position_embeddings"]
    attention_factor = rope.get("attention_factor") or (
        0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    )

    def correction_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = correction_dim(rope.get("beta_fast") or 32)
    high = correction_dim(rope.get("beta_slow") or 1)
    if rope.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    ramp = ((torch.arange(dim // 2, dtype=torch.float) - low) / (high - low)).clamp(
        0, 1
    )
    extrapolation = 1 - ramp
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - extrapolation) + (
        1.0 / pos_freqs
    ) * extrapolation
    return inv_freq, attention_factor


class GptOssRMSNorm(nn.Module):
    """Multiplies by the weight in fp32 before casting back (unlike llama's RMSNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * out).to(x.dtype)


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        hs, hd = config.hidden_size, config.head_dim
        self.input_layernorm = GptOssRMSNorm(hs, config.rms_norm_eps)
        self.q_proj = nn.Linear(hs, self.num_attention_heads * hd)
        self.k_proj = nn.Linear(hs, self.num_kv_heads * hd)
        self.v_proj = nn.Linear(hs, self.num_kv_heads * hd)
        self.o_proj = nn.Linear(self.num_attention_heads * hd, hs)
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads))

        self.layer_cache: LayerKVCache | None = None
        self.wrapper_collection = None
        self.attention_info: AttentionInfo | None = None
        self.attn_fn = self.make_attn_fn()

    def make_attn_fn(self):
        @torch.library.custom_op(
            f"tokasaurus::gpt_oss_attention_layer_{self.layer_idx}",
            mutates_args=("k_cache", "v_cache"),
        )
        def attn_fn(
            ragged_q: Tensor,
            ragged_k: Tensor,
            ragged_v: Tensor,
            k_cache: Tensor,
            v_cache: Tensor,
            sinks: Tensor,
        ) -> Tensor:
            info = self.attention_info
            assert info is not None and self.wrapper_collection is not None
            orig_q = ragged_q
            if (num_padding := info.num_padding) > 0:
                ragged_q = ragged_q[:-num_padding]
                ragged_k = ragged_k[:-num_padding]
                ragged_v = ragged_v[:-num_padding]

            append_to_kv_cache(
                token_indices=info.append_kv_token_indices,
                key=ragged_k,
                value=ragged_v,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            num_prefill = info.prefill_info.num_tokens
            outs, lses = [], []
            if num_prefill > 0:
                out, lse = self.wrapper_collection.prefill_wrapper.run_return_lse(
                    q=ragged_q[:num_prefill], paged_kv_cache=(k_cache, v_cache)
                )
                outs.append(out)
                lses.append(lse)
            if info.decode_info.num_tokens > 0:
                out, lse = self.wrapper_collection.decode_wrapper.run_return_lse(
                    q=ragged_q[num_prefill:], paged_kv_cache=(k_cache, v_cache)
                )
                outs.append(out)
                lses.append(lse)

            out, lse = torch.cat(outs), torch.cat(lses)
            # lse is log2 of the softmax denominator; the sink is a natural-log logit
            factor = 1.0 / (1.0 + torch.exp2(sinks.float()[None, :] * LOG2_E - lse))
            out = out * factor.unsqueeze(-1).to(out.dtype)

            if num_padding > 0:
                out = F.pad(out, (0, 0, 0, 0, 0, num_padding))
            assert out.shape == orig_q.shape
            return out

        @attn_fn.register_fake
        def _(ragged_q, ragged_k, ragged_v, k_cache, v_cache, sinks) -> Tensor:
            return torch.empty_like(ragged_q)

        return attn_fn

    def forward(self, batch_state: BatchState) -> BatchState:
        assert self.layer_cache is not None and self.layer_cache.v_cache is not None
        assert batch_state.position_embeddings is not None
        residual = batch_state.hidden_states
        hidden = self.input_layernorm(residual)
        num_tokens = hidden.shape[0]

        q = self.q_proj(hidden).view(
            num_tokens, self.num_attention_heads, self.head_dim
        )
        k = self.k_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        cos, sin = batch_state.position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        attn_output = self.attn_fn(
            ragged_q=q,
            ragged_k=k,
            ragged_v=v,
            k_cache=self.layer_cache.k_cache,
            v_cache=self.layer_cache.v_cache,
            sinks=self.sinks,
        ).clone()  # same torch.compile workaround as llama.py
        batch_state.hidden_states = residual + self.o_proj(
            attn_output.view(num_tokens, -1)
        )
        return batch_state


class GptOssExperts(nn.Module):
    """Stacked experts in the checkpoint layout: gate_up [E, H, 2I] (gate and up
    interleaved column-wise), down [E, I, H], each with a bias."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        e, h, i = config.num_local_experts, config.hidden_size, config.intermediate_size
        self.num_experts = e
        self.gate_up_proj = nn.Parameter(torch.empty(e, h, 2 * i))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(e, 2 * i))
        self.down_proj = nn.Parameter(torch.empty(e, i, h))
        self.down_proj_bias = nn.Parameter(torch.empty(e, h))

    def forward(
        self, hidden: Tensor, topk_indices: Tensor, topk_weights: Tensor
    ) -> Tensor:
        top_k = topk_indices.shape[1]
        flat_expert = topk_indices.reshape(-1)
        order = torch.argsort(flat_expert)
        token_indices = order // top_k
        weights = topk_weights.reshape(-1)[order]

        out = torch.zeros_like(hidden)
        start = 0
        for expert, n in enumerate(
            torch.bincount(flat_expert, minlength=self.num_experts).tolist()
        ):
            if n == 0:
                continue
            rows = token_indices[start : start + n]
            gate_up = (
                hidden[rows] @ self.gate_up_proj[expert]
                + self.gate_up_proj_bias[expert]
            )
            gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
            up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            act = (up + 1) * gate * torch.sigmoid(gate * SWIGLU_ALPHA)
            y = act @ self.down_proj[expert] + self.down_proj_bias[expert]
            out.index_add_(
                0, rows, (y * weights[start : start + n, None]).to(hidden.dtype)
            )
            start += n
        return out


class GptOssMoE(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.post_attention_layernorm = GptOssRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.router = nn.Linear(config.hidden_size, config.num_local_experts)
        self.experts = GptOssExperts(config)

    def forward(self, batch_state: BatchState) -> BatchState:
        residual = batch_state.hidden_states
        hidden = self.post_attention_layernorm(residual)
        top_logits, top_indices = torch.topk(self.router(hidden), self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1, dtype=top_logits.dtype)
        batch_state.hidden_states = residual + self.experts(
            hidden, top_indices, top_weights
        )
        return batch_state


class GptOssBlock(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx)
        self.mlp = GptOssMoE(config)
        self.is_last = layer_idx == config.num_hidden_layers - 1

    def forward(self, batch_state: BatchState) -> BatchState:
        out = self.mlp(self.self_attn(batch_state))
        if self.is_last:
            assert out.lm_head_indices is not None
            out.hidden_states = out.hidden_states[out.lm_head_indices]
        return out


class GptOssEmbeddings(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState) -> BatchState:
        batch_state.hidden_states = self.embed_tokens(batch_state.input_ids)
        return batch_state


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, extra_config: ExtraModelConfig):
        super().__init__()
        layer_start, layer_end = calc_pipeline_layer_start_and_end(config, extra_config)
        self.embed_tokens = (
            GptOssEmbeddings(config) if layer_start == 0 else nn.Identity()
        )
        if layer_end is None:
            layer_end = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                GptOssBlock(config, i)
                if layer_start <= i < layer_end
                else nn.Identity()
                for i in range(config.num_hidden_layers)
            ]
        )
        inv_freq, self.rope_scale = yarn_inv_freq(config)
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(self, batch_state: BatchState) -> BatchState:
        out: BatchState = self.embed_tokens(batch_state)
        assert out.hidden_states is not None
        freqs = batch_state.position_ids.float()[:, None] * self.rope_inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = out.hidden_states.dtype
        out.position_embeddings = (
            (emb.cos() * self.rope_scale).to(dtype),
            (emb.sin() * self.rope_scale).to(dtype),
        )
        for layer in self.layers:
            out = layer(out)
        return out


class GptOssLMHead(SoftcapLMHead):
    def __init__(self, config: GptOssConfig, extra_config: ExtraModelConfig):
        super().__init__(config, extra_config)
        self.input_norm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)


class GptOssForCausalLM(SlidingWindowForCausalLM):
    model_cls = GptOssModel
    config_cls = GptOssConfig
    lm_head_cls = GptOssLMHead
    attention_cls = GptOssAttention

    def __init__(self, config: GptOssConfig, extra_config: ExtraModelConfig):
        if (extra_config.tp_size or 1) > 1:
            raise NotImplementedError("gpt_oss does not support tensor parallelism yet")
        super().__init__(config, extra_config)

    def make_name_to_hf_name(self):
        name_to_hf_name = {}
        for key in self.state_dict():
            hf = key
            hf = hf.replace(".self_attn.input_layernorm.", ".input_layernorm.")
            hf = hf.replace(
                ".mlp.post_attention_layernorm.", ".post_attention_layernorm."
            )
            name_to_hf_name[key] = hf
        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            "model.embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = "model.norm.weight"
        name_to_hf_name["lm_head.lm_head.weight"] = "lm_head.weight"
        return name_to_hf_name

    def load_from_safetensors(self, model_path):
        name_to_hf_name = self.make_name_to_hf_name()
        quantized = [
            name
            for name in name_to_hf_name
            if name.endswith(("experts.gate_up_proj", "experts.down_proj"))
        ]
        wanted = set(name_to_hf_name.values())
        for name in quantized:
            wanted |= {f"{name}_blocks", f"{name}_scales"}

        hf_state_dict = load_safetensors_repo(
            model_path, include_parameters=wanted, device=self.device
        )
        state_dict = {}
        for name, hf_name in name_to_hf_name.items():
            if hf_name in hf_state_dict:
                state_dict[name] = hf_state_dict[hf_name].to(self.dtype)
            elif name in quantized:
                state_dict[name] = dequantize_mxfp4(
                    hf_state_dict[f"{hf_name}_blocks"],
                    hf_state_dict[f"{hf_name}_scales"],
                    self.dtype,
                )
            else:
                raise KeyError(f"{hf_name} not found in the checkpoint")
        self.load_state_dict(state_dict, assign=True, strict=False)
