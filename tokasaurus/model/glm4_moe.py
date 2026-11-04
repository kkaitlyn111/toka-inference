"""GLM-4.5 / GLM-4.6 (`glm4_moe`) serving.

Deltas from Llama: explicit head_dim (96 x 128 != hidden_size), q/k/v biases, optional
per-head qk norm, partial rope (only the first half of each head is rotated), and a
sigmoid-routed MoE FFN with a shared expert on layers >= first_k_dense_replace.

Tensor parallelism shards the expert intermediate dimension: every rank holds a slice
of every expert, and the router runs replicated on the all-gathered hidden states, so
routing is rank-invariant. There is no expert parallelism.

MoE blocks cannot be captured in CUDA/HIP graphs (the reference loop syncs on
per-expert counts and torch._grouped_mm is not capturable), so serve with
use_cudagraphs=False.

Routed outputs are combined with index_add_, which uses atomics and is nondeterministic
on GPU. On GLM-4.6 the routed outputs largely cancel, so the small per-op error compounds
over 89 MoE layers into a 2.72-nat logprob floor. GLM4_MOE_DETERMINISTIC=1 removes it at
small cost; set it on every arm of an A/B comparison.
"""

import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tokasaurus.model.glm4_moe_config import Glm4MoeConfig, register_glm4_moe_config
from tokasaurus.model.llama import (
    LlamaAttention,
    LlamaBlock,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
    all_gather,
    reduce_scatter,
)
from tokasaurus.model.qwen3 import Qwen3RMSNorm
from tokasaurus.model.safetensors_utils import load_safetensors_repo
from tokasaurus.model.types import BatchState, ExtraModelConfig

register_glm4_moe_config()

if os.environ.get("GLM4_MOE_DETERMINISTIC", "0") == "1":
    torch.use_deterministic_algorithms(True, warn_only=True)

FP8_DTYPES = {
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz")
    if hasattr(torch, name)
}
# torch._grouped_mm requires 16-byte aligned strides
GROUPED_MM_ALIGNMENT_BYTES = 16


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Rotate the first cos.shape[-1] dims of each head; pass the rest through."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_embed = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
    k_embed = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q_embed, k_embed


class Glm4MoeAttention(LlamaAttention):
    qkv_bias = True

    def __init__(
        self, config: Glm4MoeConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__(config, extra_config, layer_idx)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim(), config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim(), config.rms_norm_eps)

    def head_dim(self):
        return self.config.head_dim

    def forward(self, batch_state: BatchState):
        assert batch_state.hidden_states is not None
        assert batch_state.position_embeddings is not None
        assert self.layer_cache is not None and self.layer_cache.v_cache is not None

        residual = batch_state.hidden_states
        hidden_states = all_gather(self.input_layernorm(residual), self.extra_config)
        bsz = hidden_states.shape[0]
        head_dim = self.head_dim()

        q = self.q_proj(hidden_states).view(bsz, self.num_attention_heads, head_dim)
        k = self.k_proj(hidden_states).view(bsz, self.num_kv_heads, head_dim)
        v = self.v_proj(hidden_states).view(bsz, self.num_kv_heads, head_dim)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

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
        attn_output = attn_output.view(bsz, self.num_attention_heads * head_dim)

        if (
            self.layer_idx == self.config.num_hidden_layers - 1
            and self.extra_config.tp_size == 1
        ):
            attn_output = attn_output[batch_state.lm_head_indices]
            residual = residual[batch_state.lm_head_indices]

        o_proj = reduce_scatter(self.o_proj(attn_output), self.extra_config)
        batch_state.hidden_states = residual + o_proj
        return batch_state


class Glm4MoeTopkRouter(nn.Module):
    """Sigmoid-scored, group-limited top-k router (DeepSeek-V3 style), as in HF,
    including the fp32 router matmul and the score-correction bias."""

    def __init__(self, config: Glm4MoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size))
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    @torch.no_grad()
    def get_topk_indices(self, scores: Tensor) -> Tensor:
        experts_per_group = self.n_routed_experts // self.n_group
        scores_for_choice = (
            scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias
        )
        group_scores = (
            scores_for_choice.view(-1, self.n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        return torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        scores = F.linear(hidden_states.float(), self.weight.float()).sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_indices, topk_weights * self.routed_scaling_factor


class Glm4MoeExperts(nn.Module):
    """Routed experts as stacked [E, out, in] parameters.

    HF stores one nn.Linear triple per expert; at 160 experts x 92 layers that is ~44k
    modules, which makes module traversal slow, so the loader stacks them instead.
    """

    def __init__(self, config: Glm4MoeConfig, intermediate_size: int, tp_size: int = 1):
        super().__init__()
        assert intermediate_size % tp_size == 0, (
            f"moe_intermediate_size {intermediate_size} must be divisible by tp_size {tp_size}"
        )
        e, h, i = (
            config.n_routed_experts,
            config.hidden_size,
            intermediate_size // tp_size,
        )
        self.num_experts = e
        self.intermediate_size = i
        self.gate_proj = nn.Parameter(torch.empty(e, i, h))
        self.up_proj = nn.Parameter(torch.empty(e, i, h))
        self.down_proj = nn.Parameter(torch.empty(e, h, i))

    def use_grouped_mm(self, x: Tensor) -> bool:
        """GLM4_MOE_DISPATCH=loop|grouped|auto (default auto: grouped on GPU in bf16/fp16,
        since the grouped GEMM has no fp32 path)."""
        mode = os.environ.get("GLM4_MOE_DISPATCH", "auto").lower()
        if mode != "auto":
            return mode == "grouped"
        return (
            hasattr(torch, "_grouped_mm")
            and x.is_cuda
            and x.dtype
            in (
                torch.bfloat16,
                torch.float16,
            )
        )

    def forward(
        self, hidden_states: Tensor, topk_indices: Tensor, topk_weights: Tensor
    ):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MoE blocks cannot be captured in a CUDA/HIP graph; serve glm4_moe with "
                "use_cudagraphs=False"
            )

        top_k = topk_indices.shape[1]
        flat_expert = topk_indices.reshape(-1)
        flat_weight = topk_weights.reshape(-1)

        # sort (token, slot) pairs by expert so each expert's rows are contiguous
        order = torch.argsort(flat_expert)
        token_indices = order // top_k
        counts = torch.bincount(flat_expert, minlength=self.num_experts)

        if self.use_grouped_mm(hidden_states):
            y = self.forward_grouped(hidden_states, token_indices, counts)
        else:
            y = self.forward_loop(hidden_states, token_indices, counts)

        # accumulate in the router's fp32, as HF does
        out = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        out.index_add_(0, token_indices, y * flat_weight[order].unsqueeze(-1))
        return out.type(hidden_states.dtype)

    def forward_grouped(
        self, hidden_states: Tensor, token_indices: Tensor, counts: Tensor
    ):
        """Three torch._grouped_mm calls; per-expert offsets stay on device (no host sync)."""
        if (
            self.intermediate_size
            * self.gate_proj.element_size()
            % GROUPED_MM_ALIGNMENT_BYTES
        ):
            raise RuntimeError(
                f"sharded expert width {self.intermediate_size} is not 16-byte aligned, which "
                "torch._grouped_mm requires; lower tp_size or set GLM4_MOE_DISPATCH=loop"
            )
        offs = counts.cumsum(0).to(torch.int32)
        x = hidden_states[token_indices]
        # weights are stored [E, out, in]; _grouped_mm wants [E, in, out]
        gate = torch._grouped_mm(x, self.gate_proj.transpose(1, 2), offs=offs)
        up = torch._grouped_mm(x, self.up_proj.transpose(1, 2), offs=offs)
        return torch._grouped_mm(
            F.silu(gate) * up, self.down_proj.transpose(1, 2), offs=offs
        )

    def forward_loop(
        self, hidden_states: Tensor, token_indices: Tensor, counts: Tensor
    ):
        """Reference path (and the only fp32 / CPU one): one matmul triple per active expert."""
        out = torch.empty(
            token_indices.shape[0],
            self.down_proj.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        start = 0
        for expert_idx, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            x = hidden_states[token_indices[start : start + n]]
            gate = F.linear(x, self.gate_proj[expert_idx])
            up = F.linear(x, self.up_proj[expert_idx])
            out[start : start + n] = F.linear(
                F.silu(gate) * up, self.down_proj[expert_idx]
            )
            start += n
        return out


class Glm4MoeSharedExpert(nn.Module):
    def __init__(self, config: Glm4MoeConfig, intermediate_size: int, tp_size: int = 1):
        super().__init__()
        assert intermediate_size % tp_size == 0
        h, i = config.hidden_size, intermediate_size // tp_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Glm4MoeMoE(nn.Module):
    """MoE FFN with LlamaMLP's contract: owns the post-attention norm (named
    input_layernorm so the inherited name map applies), the collectives and the residual."""

    def __init__(self, config: Glm4MoeConfig, extra_config: ExtraModelConfig):
        super().__init__()
        self.extra_config = extra_config
        tp_size = extra_config.tp_size or 1
        self.gate = Glm4MoeTopkRouter(config)
        self.experts = Glm4MoeExperts(
            config, config.moe_intermediate_size, tp_size=tp_size
        )
        self.shared_experts = Glm4MoeSharedExpert(
            config,
            config.moe_intermediate_size * config.n_shared_experts,
            tp_size=tp_size,
        )
        self.input_layernorm = RMSNorm(config)

    def forward(self, batch_state: BatchState):
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = all_gather(self.input_layernorm(inp), self.extra_config)
        topk_indices, topk_weights = self.gate(hidden_states)
        out = self.experts(hidden_states, topk_indices, topk_weights)
        out = out + self.shared_experts(hidden_states)
        batch_state.hidden_states = inp + reduce_scatter(out, self.extra_config)
        return batch_state


class Glm4MoeBlock(LlamaBlock):
    attn_cls = Glm4MoeAttention

    def __init__(
        self, config: Glm4MoeConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        nn.Module.__init__(self)
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx
        self.self_attn = self.attn_cls(config, extra_config, layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4MoeMoE(config, extra_config)
        else:
            self.mlp = LlamaMLP(config, extra_config, layer_idx)


class Glm4MoeModel(LlamaModel):
    block_cls = Glm4MoeBlock


class Glm4MoeForCausalLM(LlamaForCausalLM):
    """Checkpoints may also hold a multi-token-prediction layer at index
    num_hidden_layers; it is never built or loaded."""

    model_cls = Glm4MoeModel
    config_cls = Glm4MoeConfig

    def head_dim(self):
        return self.config.head_dim

    # FP8: zai-org/GLM-4.6-FP8 is compressed-tensors with one fp32 scale per output
    # channel (e4m3). Weights are dequantized to self.dtype at load.
    def is_fp8_checkpoint(self) -> bool:
        qc = getattr(self.config, "quantization_config", None)
        if qc is None:
            return False
        method = (
            qc.get("quant_method")
            if isinstance(qc, dict)
            else getattr(qc, "quant_method", None)
        )
        return method == "compressed-tensors"

    @staticmethod
    def scale_name(hf_name: str) -> str:
        return f"{hf_name}_scale"

    def dequantize(self, weight: Tensor, scale: Tensor | None) -> Tensor:
        # scale is [out, 1] ([E, out, 1] once stacked); multiply in fp32 to avoid
        # rounding twice through bf16
        if scale is None:
            return weight
        return (weight.float() * scale.float()).to(self.dtype)

    def moe_layer_indices(self) -> list[int]:
        return list(
            range(self.config.first_k_dense_replace, self.config.num_hidden_layers)
        )

    def make_stacked_name_to_hf_names(self) -> dict[str, list[str]]:
        stacked = {}
        for layer_idx in self.moe_layer_indices():
            for proj in ("gate_proj", "up_proj", "down_proj"):
                stacked[f"model.layers.{layer_idx}.mlp.experts.{proj}"] = [
                    f"model.layers.{layer_idx}.mlp.experts.{e}.{proj}.weight"
                    for e in range(self.config.n_routed_experts)
                ]
        return stacked

    def make_name_to_hf_name(self):
        """1:1 names; the stacked expert parameters are mapped separately."""
        name_to_hf_name = super().make_name_to_hf_name()
        for stacked_name in self.make_stacked_name_to_hf_names():
            name_to_hf_name.pop(stacked_name, None)
        return name_to_hf_name

    def make_tp_map(self):
        """Keyed by both our names (tp_modify_state_dict) and HF names
        (load_safetensors_repo), which differ for the stacked experts."""
        tp_map = super().make_tp_map()

        # q/k/v biases shard with their weights; o_proj has none, and the router's
        # e_score_correction_bias is replicated, so match exact suffixes
        for param_name, _ in self.named_parameters():
            if param_name.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias")):
                tp_map[param_name] = 0

        # stacked params have a leading expert dim, shifting the dense rule by one
        for stacked_name, hf_names in self.make_stacked_name_to_hf_names().items():
            is_down = stacked_name.endswith("down_proj")
            tp_map[stacked_name] = 2 if is_down else 1
            for hf_name in hf_names:
                tp_map[hf_name] = 1 if is_down else 0

        # per-output-channel scales shard only with weights split on the output dim
        if self.is_fp8_checkpoint():
            for hf_name, split_dim in list(tp_map.items()):
                if hf_name.endswith(".weight") and split_dim == 0:
                    tp_map[self.scale_name(hf_name)] = 0

        return tp_map

    def stack_expert_tensors(self, hf_state_dict: dict) -> dict:
        """Stack (and dequantize) per-expert tensors, popping sources as they are used
        so fp8 originals are freed incrementally rather than held until the end."""
        stacked_state = {}
        fp8 = self.is_fp8_checkpoint()
        for stacked_name, hf_names in self.make_stacked_name_to_hf_names().items():
            present = [n for n in hf_names if n in hf_state_dict]
            if not present:
                continue
            if len(present) != len(hf_names):
                missing = next(n for n in hf_names if n not in hf_state_dict)
                raise RuntimeError(
                    f"{stacked_name}: partial expert stack, first missing {missing}"
                )
            weight = torch.stack([hf_state_dict.pop(n) for n in hf_names])

            scale = None
            scale_names = [self.scale_name(n) for n in hf_names]
            if fp8 and any(n in hf_state_dict for n in scale_names):
                if not all(n in hf_state_dict for n in scale_names):
                    raise RuntimeError(f"{stacked_name}: partial expert weight_scales")
                scale = torch.stack([hf_state_dict.pop(n) for n in scale_names])

            stacked_state[stacked_name] = self.dequantize(weight, scale)
        return stacked_state

    def load_from_safetensors(self, model_path):
        name_to_hf_name = self.make_name_to_hf_name()
        wanted = set(name_to_hf_name.values())
        for hf_names in self.make_stacked_name_to_hf_names().values():
            wanted.update(hf_names)
        if self.is_fp8_checkpoint():
            wanted |= {self.scale_name(n) for n in wanted if n.endswith(".weight")}

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=wanted,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=self.make_tp_map(),
        )

        # dequantize while popping, so peak memory is max(fp8, bf16) rather than the sum
        # (GLM-4.6 at tp=4 OOMs otherwise)
        state_dict = {}
        for name, hf_name in name_to_hf_name.items():
            if hf_name in hf_state_dict:
                weight = hf_state_dict.pop(hf_name)
                state_dict[name] = self.dequantize(
                    weight, hf_state_dict.pop(self.scale_name(hf_name), None)
                )
        state_dict.update(self.stack_expert_tensors(hf_state_dict))
        del hf_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.load_state_dict(state_dict, assign=True, strict=False)
        self.check_fully_loaded()

    def load_from_hf_pretrained(self, model_name_or_path, device=None, dtype=None):
        import transformers

        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, device_map=device
        )
        hf_state_dict = dict(hf_model.state_dict())
        state_dict = {
            name: self.dequantize(
                hf_state_dict[hf_name], hf_state_dict.get(self.scale_name(hf_name))
            )
            for name, hf_name in self.make_name_to_hf_name().items()
            if hf_name in hf_state_dict
        }
        state_dict.update(self.stack_expert_tensors(hf_state_dict))
        if self.extra_config.tp_size > 1:
            state_dict = self.tp_modify_state_dict(
                state_dict, self.extra_config.tp_rank, self.extra_config.tp_size
            )
        self.load_state_dict(state_dict, assign=True, strict=False)
        del hf_model
        self.check_fully_loaded()

    def check_fully_loaded(self):
        """Fail on silent partial loads (what an expert-name mismatch produces) and on
        fp8 weights whose scale was missing (they would run but produce garbage)."""
        meta = [n for n, p in self.named_parameters() if p.is_meta]
        meta += [n for n, b in self.named_buffers() if b.is_meta and "rope" not in n]
        if meta:
            raise RuntimeError(
                f"{len(meta)} parameters not loaded from the checkpoint: {meta[:8]}"
            )
        fp8_left = [n for n, p in self.named_parameters() if p.dtype in FP8_DTYPES]
        if fp8_left:
            raise RuntimeError(
                f"{len(fp8_left)} parameters still float8 after loading (missing weight_scale; "
                f"only compressed-tensors per-channel fp8 is supported): {fp8_left[:8]}"
            )
