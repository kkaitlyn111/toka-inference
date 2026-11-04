"""Gemma 4 config for transformers 4.x.

The gemma4 model type only exists in transformers >= 5.5. This lets
AutoConfig.from_pretrained resolve `model_type: gemma4` checkpoints, flattening the
nested `text_config` (the checkpoints are Gemma4ForConditionalGeneration wrappers) onto
the top-level config that the tokasaurus model reads.
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from tokasaurus.model.layer_types import FULL_ATTENTION, SLIDING_ATTENTION


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"

    def __init__(self, text_config: dict | None = None, **kwargs):
        # Multimodal wrapper fields (vision_config etc.) ride along in kwargs;
        # we only flatten what the text stack needs.
        text = dict(text_config or {})

        self.hidden_size = text.get("hidden_size", 5376)
        self.intermediate_size = text.get("intermediate_size", 21504)
        self.num_hidden_layers = text.get("num_hidden_layers", 60)
        self.num_attention_heads = text.get("num_attention_heads", 32)
        self.num_key_value_heads = text.get("num_key_value_heads", 16)
        self.num_global_key_value_heads = text.get("num_global_key_value_heads", 4)
        self.head_dim = text.get("head_dim", 256)
        self.global_head_dim = text.get("global_head_dim", 512)
        self.vocab_size = text.get("vocab_size", 262144)
        self.rms_norm_eps = text.get("rms_norm_eps", 1e-6)
        self.hidden_activation = text.get("hidden_activation", "gelu_pytorch_tanh")
        self.max_position_embeddings = text.get("max_position_embeddings", 262144)
        self.sliding_window = text.get("sliding_window", 1024)
        self.attention_bias = text.get("attention_bias", False)
        self.attention_k_eq_v = text.get("attention_k_eq_v", True)
        self.final_logit_softcapping = text.get("final_logit_softcapping", 30.0)
        self.layer_types = text.get(
            "layer_types",
            [
                FULL_ATTENTION if (i + 1) % 6 == 0 else SLIDING_ATTENTION
                for i in range(self.num_hidden_layers)
            ],
        )
        self.rope_parameters = text.get(
            "rope_parameters",
            {
                SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 10_000.0},
                FULL_ATTENTION: {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        )
        # Paths we do not support in the port — fail loud if a future
        # checkpoint enables them.
        self.num_kv_shared_layers = text.get("num_kv_shared_layers", 0)
        self.enable_moe_block = text.get("enable_moe_block", False)
        self.hidden_size_per_layer_input = text.get("hidden_size_per_layer_input", 0)
        self.use_bidirectional_attention = text.get(
            "use_bidirectional_attention", "vision"
        )

        # passed through super(): PretrainedConfig.__init__ otherwise resets it to True
        tie = text.get("tie_word_embeddings", kwargs.pop("tie_word_embeddings", True))
        # eos/bos/pad from the text config (wrapper-level ids are for multimodal)
        kwargs.setdefault("eos_token_id", text.get("eos_token_id", 1))
        kwargs.setdefault("bos_token_id", text.get("bos_token_id", 2))
        kwargs.setdefault("pad_token_id", text.get("pad_token_id", 0))
        super().__init__(tie_word_embeddings=tie, **kwargs)

    # --- per-layer geometry helpers -------------------------------------
    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == FULL_ATTENTION

    def head_dim_for_layer(self, layer_idx: int) -> int:
        return (
            self.global_head_dim if self.is_full_attention(layer_idx) else self.head_dim
        )

    def kv_heads_for_layer(self, layer_idx: int) -> int:
        return (
            self.num_global_key_value_heads
            if self.is_full_attention(layer_idx)
            else self.num_key_value_heads
        )

    def head_dim_for_type(self, layer_type: str) -> int:
        return self.global_head_dim if layer_type == FULL_ATTENTION else self.head_dim

    def kv_heads_for_type(self, layer_type: str) -> int:
        return (
            self.num_global_key_value_heads
            if layer_type == FULL_ATTENTION
            else self.num_key_value_heads
        )

    def validate_supported(self):
        assert self.num_kv_shared_layers == 0, "kv-shared layers not supported"
        assert not self.enable_moe_block, "MoE blocks not supported"
        assert self.hidden_size_per_layer_input == 0, (
            "per-layer embeddings not supported"
        )
        assert self.use_bidirectional_attention in ("vision", None), (
            "non-causal text attention not supported"
        )
        assert not self.attention_bias


def register_gemma4_config():
    try:
        AutoConfig.register("gemma4", Gemma4Config)
    except ValueError:
        pass  # already registered
