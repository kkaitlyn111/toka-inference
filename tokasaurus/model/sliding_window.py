"""Shared engine glue for models that mix sliding-window and full-attention layers.

flashinfer bakes `window_left` into plan(), so these models keep one WrapperCollection
per layer type and plan each with its own window. Cudagraphs and hydragen assume a
single collection, so ServerConfig disables both for these models.
"""

import torch
from torch import nn

from tokasaurus.model.attention_utils import create_wrappers
from tokasaurus.model.kv_cache import LayerKVCache
from tokasaurus.model.layer_types import LAYER_TYPES, SLIDING_ATTENTION
from tokasaurus.model.llama import (
    LlamaForCausalLM,
    LlamaLMHead,
    all_gather,
    calc_pipeline_layer_start_and_end,
    calc_tokens_and_logprobs,
)
from tokasaurus.model.types import (
    AttentionInfo,
    BatchState,
    ExtraModelConfig,
    ModelOutputTensors,
)

# checkpoints are multimodal wrappers; only the text stack is served
TEXT_PREFIX = "model.language_model."

# sandwich norms live inside self_attn/mlp here, at the layer level in HF checkpoints
NORM_RELOCATIONS = {
    ".self_attn.input_layernorm.": ".input_layernorm.",
    ".self_attn.post_attention_layernorm.": ".post_attention_layernorm.",
    ".mlp.pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
    ".mlp.post_feedforward_layernorm.": ".post_feedforward_layernorm.",
}


class SoftcapLMHead(LlamaLMHead):
    """LM head with an optional logit multiplier and tanh softcap, both applied to the
    activation-dtype logits before float() (HF's order)."""

    logit_multiplier: float = 1.0

    def forward(self, batch_state: BatchState):
        assert batch_state.hidden_states is not None
        assert batch_state.lm_head_indices is not None
        hidden_states = batch_state.hidden_states

        if batch_state.lm_head_indices.numel() == 0:
            batch_state.outputs = ModelOutputTensors(
                output_ids=torch.empty(
                    0, device=hidden_states.device, dtype=torch.long
                ),
                chosen_logprobs=torch.empty(
                    0, device=hidden_states.device, dtype=torch.float32
                ),
            )
            return batch_state

        if self.extra_config.tp_size > 1:
            hidden_states = all_gather(hidden_states, self.extra_config)
            hidden_states = hidden_states[batch_state.lm_head_indices]

        logits = self.lm_head(self.input_norm(hidden_states))
        if self.logit_multiplier != 1.0:
            logits = logits * self.logit_multiplier
        if (softcap := self.config.final_logit_softcapping) is not None:
            logits = torch.tanh(logits / softcap) * softcap
        logits = logits.float()

        assert batch_state.sampling_params.top_p is None
        assert batch_state.sampling_params.temperature is not None
        assert batch_state.sampling_params.greedy_mask is not None
        outputs = calc_tokens_and_logprobs(
            logits,
            temperature=batch_state.sampling_params.temperature,
            greedy_mask=batch_state.sampling_params.greedy_mask,
            config=self.extra_config,
        )

        for name in ("output_ids", "chosen_logprobs", "topk_indices", "topk_logprobs"):
            if (value := getattr(outputs, name)) is not None:
                setattr(outputs, name, all_gather(value, self.extra_config))
        batch_state.outputs = outputs
        return batch_state


class SlidingWindowForCausalLM(LlamaForCausalLM):
    """Subclasses set model_cls, config_cls, lm_head_cls and attention_cls, and their
    config provides layer_type(i), sliding_window and final_logit_softcapping."""

    lm_head_cls: type[SoftcapLMHead]
    attention_cls: type[nn.Module]

    def __init__(self, config, extra_config: ExtraModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self.model = self.model_cls(config, extra_config)

        _, layer_end = calc_pipeline_layer_start_and_end(config, extra_config)
        if layer_end in [None, config.num_hidden_layers]:
            self.lm_head = self.lm_head_cls(config, extra_config)
        else:
            self.lm_head = nn.Identity()

    # per-layer-type geometry; uniform unless a model overrides it
    def kv_heads_for_type(self, layer_type: str) -> int:
        return self.config.num_key_value_heads

    def head_dim_for_type(self, layer_type: str) -> int:
        return self.config.head_dim

    def sm_scale(self) -> float:
        return self.config.head_dim**-0.5

    def uses_flashinfer_prefill(self, layer_type: str) -> bool:
        return True

    def tp_size(self) -> int:
        return self.extra_config.tp_size or 1

    def num_kv_heads(self):
        return self.kv_heads_for_type(SLIDING_ATTENTION) // self.tp_size()

    def num_qo_heads(self):
        return self.config.num_attention_heads // self.tp_size()

    def head_dim(self):
        return self.head_dim_for_type(SLIDING_ATTENTION)

    def attention_layers(self):
        for module in self.model.modules():
            if isinstance(module, self.attention_cls):
                yield module

    def make_layer_wrappers(self) -> dict:
        return {
            layer_type: create_wrappers(
                device=self.device,
                num_attention_heads=self.num_qo_heads(),
                num_key_value_heads=self.kv_heads_for_type(layer_type)
                // self.tp_size(),
            )
            for layer_type in LAYER_TYPES
        }

    def set_wrappers(self, wrappers: dict):
        self.wrapper_collection = wrappers
        for layer in self.attention_layers():
            layer.wrapper_collection = wrappers[layer.layer_type]

    def setup_caches(self, num_pages: int, page_size: int):
        self.set_wrappers(self.make_layer_wrappers())
        for layer in self.attention_layers():
            layer.layer_cache = LayerKVCache(
                head_dim=layer.head_dim,
                num_kv_heads=layer.num_kv_heads,
                num_pages=num_pages,
                page_size=page_size,
                device=self.device,
                dtype=self.dtype,
            )

    def set_attention_info(self, attn_info: AttentionInfo):
        for layer in self.attention_layers():
            layer.attention_info = attn_info

    def plan(self, attn_info: AttentionInfo, non_blocking: bool = False):
        assert self.wrapper_collection is not None
        assert (
            attn_info.hydragen_info is None or attn_info.hydragen_info.num_tokens == 0
        ), "hydragen is not supported with per-layer-type attention windows"
        self.set_attention_info(attn_info)

        for layer_type, wrappers in self.wrapper_collection.items():
            plan_kwargs = dict(
                num_kv_heads=self.kv_heads_for_type(layer_type) // self.tp_size(),
                num_qo_heads=self.num_qo_heads(),
                head_dim=self.head_dim_for_type(layer_type),
                page_size=attn_info.page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                sm_scale=self.sm_scale(),
                window_left=self.config.sliding_window - 1
                if layer_type == SLIDING_ATTENTION
                else -1,
                non_blocking=non_blocking,
            )
            pre = attn_info.prefill_info
            if pre.num_tokens > 0 and self.uses_flashinfer_prefill(layer_type):
                wrappers.prefill_wrapper.plan(
                    qo_indptr=pre.qo_indptr,
                    paged_kv_indptr=pre.kv_indptr,
                    paged_kv_indices=pre.kv_indices,
                    paged_kv_last_page_len=pre.kv_last_page_len,
                    causal=True,
                    **plan_kwargs,
                )
            if (dec := attn_info.decode_info).num_tokens > 0:
                wrappers.decode_wrapper.plan(
                    indptr=dec.kv_indptr,
                    indices=dec.kv_indices,
                    last_page_len=dec.kv_last_page_len,
                    **plan_kwargs,
                )

    def make_name_to_hf_name(self):
        name_to_hf_name = {}
        for k in self.state_dict():
            if k.startswith("model.layers."):
                rest = k[len("model.") :]
                for ours, hf in NORM_RELOCATIONS.items():
                    rest = rest.replace(ours, hf)
                name_to_hf_name[k] = TEXT_PREFIX + rest
            else:
                name_to_hf_name[k] = k

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            TEXT_PREFIX + "embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = TEXT_PREFIX + "norm.weight"
        name_to_hf_name["lm_head.lm_head.weight"] = (
            TEXT_PREFIX + "embed_tokens.weight"
            if self.config.tie_word_embeddings
            else "lm_head.weight"
        )
        return name_to_hf_name

    def make_tp_map(self):
        # load_safetensors_repo looks the split dim up by checkpoint name, which differs
        # from ours (prefix + norm relocations), so key the map both ways
        tp_map = super().make_tp_map()
        name_to_hf = self.make_name_to_hf_name()
        return {**tp_map, **{name_to_hf.get(k, k): dim for k, dim in tp_map.items()}}
