"""gpt-oss (`gpt_oss`) config for transformers versions that predate it (< 4.55)."""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from tokasaurus.model.layer_types import FULL_ATTENTION, SLIDING_ATTENTION

DEFAULT_ROPE = {
    "rope_type": "yarn",
    "rope_theta": 150_000.0,
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
    "original_max_position_embeddings": 4096,
}


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(
        self,
        num_hidden_layers: int = 24,
        num_local_experts: int = 32,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        num_experts_per_tok: int = 4,
        layer_types: list[str] | None = None,
        rope_theta: float | None = None,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_local_experts = num_local_experts
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_experts_per_tok = num_experts_per_tok
        self.layer_types = layer_types or [
            SLIDING_ATTENTION if (i + 1) % 2 else FULL_ATTENTION
            for i in range(num_hidden_layers)
        ]
        # released checkpoints use rope_theta + rope_scaling; transformers >= 5 nests both
        # under rope_parameters
        self.rope = {**DEFAULT_ROPE, **(rope_scaling or {}), **(rope_parameters or {})}
        if rope_theta is not None:
            self.rope["rope_theta"] = rope_theta
        self.final_logit_softcapping = None
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]


def register_gpt_oss_config():
    try:
        AutoConfig.register("gpt_oss", GptOssConfig)
    except ValueError:
        pass  # already registered
