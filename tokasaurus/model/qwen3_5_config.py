"""Qwen 3.5 (`qwen3_5`) config for transformers versions that predate it.

The released checkpoints are vision-language wrappers; this flattens the nested
`text_config` onto the top-level config the tokasaurus model reads.
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from tokasaurus.model.layer_types import FULL_ATTENTION, LINEAR_ATTENTION


class Qwen35Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, text_config: dict | None = None, **kwargs):
        text = dict(text_config or {})

        self.hidden_size = text.get("hidden_size", 4096)
        self.intermediate_size = text.get("intermediate_size", 12288)
        self.num_hidden_layers = text.get("num_hidden_layers", 32)
        self.vocab_size = text.get("vocab_size", 248320)
        self.rms_norm_eps = text.get("rms_norm_eps", 1e-6)
        self.hidden_act = text.get("hidden_act", "silu")
        self.max_position_embeddings = text.get("max_position_embeddings", 262144)

        # softmax-attention layers; q_proj emits 2 * head_dim per head (query + output gate)
        self.num_attention_heads = text.get("num_attention_heads", 16)
        self.num_key_value_heads = text.get("num_key_value_heads", 4)
        self.head_dim = text.get("head_dim", 256)
        self.attention_bias = text.get("attention_bias", False)
        self.attn_output_gate = text.get("attn_output_gate", True)

        # gated-DeltaNet layers
        self.linear_num_key_heads = text.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = text.get("linear_num_value_heads", 32)
        self.linear_key_head_dim = text.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = text.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = text.get("linear_conv_kernel_dim", 4)
        self.mamba_ssm_dtype = text.get("mamba_ssm_dtype", "float32")

        self.full_attention_interval = text.get("full_attention_interval", 4)
        self.layer_types = text.get(
            "layer_types",
            [
                FULL_ATTENTION
                if (i + 1) % self.full_attention_interval == 0
                else LINEAR_ATTENTION
                for i in range(self.num_hidden_layers)
            ],
        )

        # the checkpoint declares interleaved mrope, which reduces to plain 1D rope for
        # text-only input (all three position rows are equal)
        self.rope_parameters = text.get(
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
            },
        )
        self.rope_theta = self.rope_parameters.get("rope_theta", 10_000_000.0)
        self.partial_rotary_factor = self.rope_parameters.get(
            "partial_rotary_factor", 1.0
        )

        self.mlp_only_layers = text.get("mlp_only_layers", [])
        # passed through super(): PretrainedConfig.__init__ otherwise resets it to True
        tie = kwargs.pop("tie_word_embeddings", text.get("tie_word_embeddings", False))
        kwargs.setdefault("eos_token_id", text.get("eos_token_id", 248044))
        super().__init__(tie_word_embeddings=tie, **kwargs)

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == FULL_ATTENTION

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    def validate_supported(self):
        assert not self.attention_bias, "q/k/v bias not supported"
        assert self.attn_output_gate, "only the gated-attention variant is supported"
        assert not self.mlp_only_layers, "mlp_only_layers not supported"
        assert set(self.layer_types) <= {FULL_ATTENTION, LINEAR_ATTENTION}, (
            self.layer_types
        )
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.rope_parameters.get("rope_type", "default") == "default"
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.linear_num_value_heads % self.linear_num_key_heads == 0
        assert self.mamba_ssm_dtype in (None, "float32"), (
            "only an fp32 recurrent state is supported"
        )


def register_qwen3_5_config():
    try:
        AutoConfig.register("qwen3_5", Qwen35Config)
    except ValueError:
        pass  # already registered
