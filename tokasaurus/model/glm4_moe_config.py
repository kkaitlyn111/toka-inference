"""GLM-4.5/4.6 (`glm4_moe`) config for transformers versions that predate it (< 4.54)."""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Glm4MoeConfig(PretrainedConfig):
    model_type = "glm4_moe"

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 5120,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 92,
        num_attention_heads: int = 96,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float | None = None,
        rope_scaling: dict | None = None,
        partial_rotary_factor: float | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = True,
        use_qk_norm: bool = False,
        moe_intermediate_size: int = 1536,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 160,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        first_k_dense_replace: int = 3,
        norm_topk_prob: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # transformers >= 5 nests rope settings under rope_parameters
        rope_parameters = rope_parameters or {}
        self.rope_theta = rope_theta or rope_parameters.get("rope_theta", 1_000_000.0)
        self.partial_rotary_factor = partial_rotary_factor or rope_parameters.get(
            "partial_rotary_factor", 0.5
        )
        self.rope_scaling = rope_scaling

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def register_glm4_moe_config():
    try:
        AutoConfig.register("glm4_moe", Glm4MoeConfig)
    except ValueError:
        pass  # transformers already knows glm4_moe
