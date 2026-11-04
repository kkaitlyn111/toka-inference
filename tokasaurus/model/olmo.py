"""Native OLMo support for checkpoints in AI2's original `hf_olmo` format.

This covers the `model_type: "hf_olmo"` checkpoints published by AI2 (for example the
DataDecide suite, `allenai/DataDecide-*`), which transformers cannot load without
`ai2-olmo`'s remote code. Instead of going through transformers, the config is read
straight from `config.json` and the weights straight from the safetensors files, and
both are mapped onto tokasaurus' Llama implementation:

  * OLMo's fused `att_proj` (QKV) and `ff_proj` (up/gate) are split at load time,
  * the config keys (`d_model`, `n_heads`, ...) are renamed to the Llama names that the
    rest of tokasaurus reads (`hidden_size`, `num_attention_heads`, ...).

Architectural variants that the Llama blocks cannot express (QK-norm, non-RMS or
non-affine layer norms, biases, QKV clipping, ALiBi, tied embeddings) are rejected
up front instead of silently producing wrong logits.
"""

from pathlib import Path

import huggingface_hub
import torch
from accelerate import init_empty_weights

from tokasaurus.model.llama import (
    LlamaAttention,
    LlamaBlock,
    LlamaForCausalLM,
    LlamaModel,
)
from tokasaurus.model.olmo_config import OlmoConfig
from tokasaurus.model.safetensors_utils import load_safetensors_repo
from tokasaurus.model.types import ExtraModelConfig


class OlmoAttention(LlamaAttention):
    pass


class OlmoBlock(LlamaBlock):
    attn_cls = OlmoAttention


class OlmoModel(LlamaModel):
    block_cls = OlmoBlock


class OlmoForCausalLM(LlamaForCausalLM):
    model_cls = OlmoModel
    config_cls = OlmoConfig

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        extra_config: ExtraModelConfig | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        config = cls.config_cls.from_pretrained(path)

        if extra_config is None:
            extra_config = ExtraModelConfig()

        assert extra_config.tp_size == 1, (
            "The OLMo loader does not shard weights yet; use tp_size=1 (dp/pp work)."
        )

        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        with init_empty_weights(include_buffers=False):
            model = cls(config, extra_config)

        model.dtype = dtype
        model.device = device

        if (as_path := Path(path)).exists():
            model_path = as_path
        else:
            model_path = Path(
                huggingface_hub.snapshot_download(
                    path, allow_patterns=["*.safetensors", "*.json"]
                )
            )

        model.load_from_safetensors(model_path, device, dtype)

        model.to(device=device)
        return model

    def load_from_safetensors(
        self, path: Path, device: torch.device | None, dtype: torch.dtype | None
    ):
        name_to_hf_name = self.make_name_to_hf_name()

        hf_state_dict = load_safetensors_repo(
            path,
            include_parameters=set(name_to_hf_name.values()),
            device=device,
        )

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        q_size = self.config.num_attention_heads * head_dim
        kv_size = self.config.num_key_value_heads * head_dim

        state_dict = {}
        for toka_name, hf_name in name_to_hf_name.items():
            # k/v and up are produced when their fused partner (q / gate) is split.
            if (
                "self_attn.k_proj.weight" in toka_name
                or "self_attn.v_proj.weight" in toka_name
                or "mlp.up_proj.weight" in toka_name
            ):
                continue

            tensor = hf_state_dict[hf_name]

            if "self_attn.q_proj.weight" in toka_name:
                # att_proj is [(n_q + 2 * n_kv) * head_dim, d_model], stacked q, k, v.
                q, k, v = torch.split(tensor, [q_size, kv_size, kv_size], dim=0)
                state_dict[toka_name] = q
                state_dict[toka_name.replace("q_proj", "k_proj")] = k
                state_dict[toka_name.replace("q_proj", "v_proj")] = v

            elif "mlp.gate_proj.weight" in toka_name:
                # OLMo's SwiGLU does `x, gate = ff_proj(x).chunk(2); silu(gate) * x`,
                # so ff_proj is stacked [W_up; W_gate] -- up first, gate second.
                up, gate = torch.split(tensor, self.config.intermediate_size, dim=0)
                state_dict[toka_name] = gate
                state_dict[toka_name.replace("gate_proj", "up_proj")] = up

            else:
                state_dict[toka_name] = tensor

        if dtype is not None:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        self.load_state_dict(state_dict, assign=True, strict=False)

    def make_name_to_hf_name(self):
        """Map tokasaurus (Llama-style) parameter names to hf_olmo safetensors names.

        hf_olmo:                                    tokasaurus:
          model.transformer.wte                       model.embed_tokens.embed_tokens
          model.transformer.blocks.{i}.att_proj       model.layers.{i}.self_attn.{q,k,v}_proj
          model.transformer.blocks.{i}.attn_norm      model.layers.{i}.self_attn.input_layernorm
          model.transformer.blocks.{i}.attn_out       model.layers.{i}.self_attn.o_proj
          model.transformer.blocks.{i}.ff_norm        model.layers.{i}.mlp.input_layernorm
          model.transformer.blocks.{i}.ff_proj        model.layers.{i}.mlp.{gate,up}_proj
          model.transformer.blocks.{i}.ff_out         model.layers.{i}.mlp.down_proj
          model.transformer.ln_f                      lm_head.input_norm
          model.transformer.ff_out                    lm_head.lm_head
        """
        name_to_hf_name = {
            "model.embed_tokens.embed_tokens.weight": "model.transformer.wte.weight",
            "lm_head.input_norm.weight": "model.transformer.ln_f.weight",
            "lm_head.lm_head.weight": "model.transformer.ff_out.weight",
        }

        for i in range(self.config.num_hidden_layers):
            block = f"model.transformer.blocks.{i}"
            layer = f"model.layers.{i}"
            for proj in ("q_proj", "k_proj", "v_proj"):
                name_to_hf_name[f"{layer}.self_attn.{proj}.weight"] = (
                    f"{block}.att_proj.weight"
                )
            name_to_hf_name[f"{layer}.self_attn.o_proj.weight"] = (
                f"{block}.attn_out.weight"
            )
            name_to_hf_name[f"{layer}.self_attn.input_layernorm.weight"] = (
                f"{block}.attn_norm.weight"
            )
            name_to_hf_name[f"{layer}.mlp.input_layernorm.weight"] = (
                f"{block}.ff_norm.weight"
            )
            for proj in ("gate_proj", "up_proj"):
                name_to_hf_name[f"{layer}.mlp.{proj}.weight"] = (
                    f"{block}.ff_proj.weight"
                )
            name_to_hf_name[f"{layer}.mlp.down_proj.weight"] = f"{block}.ff_out.weight"

        return name_to_hf_name
