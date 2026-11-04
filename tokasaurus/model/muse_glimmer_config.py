"""Muse Glimmer config for transformers 4.x.

The `muse_glimmer` model type only exists in transformers >= 5.15; this shim lets
AutoConfig.from_pretrained resolve `model_type: muse_glimmer` checkpoints without
upgrading transformers. It flattens the nested `text_config` (the checkpoints are
MuseGlimmerForConditionalGeneration wrappers with a vision tower we do not serve)
onto the top-level config, which is the shape the tokasaurus model code expects.

Architecture facts pinned from the transformers 5.15.0 source
(`transformers/models/muse_glimmer/modeling_muse_glimmer.py`):

  * 52 layers, `[sliding, sliding, sliding, full]` repeating -> 13 full / 39
    sliding, window 2048. Full-attention layers are NoPE: the model passes
    `position_embeddings=None` wherever `layer_rope_theta[i] == 0`.
  * Uniform geometry on every layer: 32 query heads, 2 KV heads, head_dim 128.
    (Unlike Gemma 4, local and global layers share a shape, so the KV cache does
    not need per-layer-type geometry.)
  * q and k share ONE scale-free RMSNorm over head_dim, and q is then multiplied
    by `qk_scale_factor`. There is no v_norm.
  * Attention output is gated: `attn_out * sigmoid(gate_proj(x))` before o_proj,
    where x is the *input* to the attention block (post input_layernorm).
  * The four sandwich norms are "centered" (scale is `1 + w`, w init 0); the
    final model norm is a plain `* w` RMSNorm. The post_* norms use
    `post_norm_eps` (1e-8), the pre_* norms use `rms_norm_eps` (1e-5).
  * Embeddings are followed by a scale-free RMSNorm (not a sqrt(H) multiplier).
  * Logits: `lm_head(h) * output_multiplier`, then Gemma-style tanh softcap at
    `final_logit_softcapping`.
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from tokasaurus.model.layer_types import FULL_ATTENTION, SLIDING_ATTENTION


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"

    def __init__(self, text_config: dict | None = None, **kwargs):
        # Multimodal wrapper fields (vision_config, vision_adapter, ...) ride
        # along in kwargs; we only flatten what the text stack needs.
        text = dict(text_config or {})

        self.hidden_size = text.get("hidden_size", 6656)
        self.intermediate_size = text.get("intermediate_size", 19968)
        self.num_hidden_layers = text.get("num_hidden_layers", 52)
        self.num_attention_heads = text.get("num_attention_heads", 32)
        self.num_key_value_heads = text.get("num_key_value_heads", 2)
        self.head_dim = text.get("head_dim", 128)
        self.vocab_size = text.get("vocab_size", 202048)
        self.rms_norm_eps = text.get("rms_norm_eps", 1e-5)
        self.post_norm_eps = text.get("post_norm_eps", 1e-8)
        self.hidden_activation = text.get("hidden_activation", "silu")
        self.max_position_embeddings = text.get("max_position_embeddings", 131072)
        self.sliding_window = text.get("sliding_window", 2048)
        self.attention_bias = text.get("attention_bias", False)
        self.qk_scale_factor = text.get("qk_scale_factor", 3.87)
        self.output_multiplier = text.get("output_multiplier", 0.19611613513818404)
        self.final_logit_softcapping = text.get("final_logit_softcapping", 20.0)

        # Default pattern from MuseGlimmerTextConfig.__post_init__: full attention
        # on every 4th layer counted BACKWARD from the last.
        self.layer_types = text.get(
            "layer_types",
            [
                FULL_ATTENTION
                if (self.num_hidden_layers - 1 - i) % 4 == 0
                else SLIDING_ATTENTION
                for i in range(self.num_hidden_layers)
            ],
        )
        self.rope_parameters = text.get(
            "rope_parameters", {"rope_type": "default", "rope_theta": 500_000.0}
        )
        # 0 marks a NoPE layer. Default: NoPE exactly on the full-attention layers.
        self.layer_rope_theta = text.get(
            "layer_rope_theta",
            [
                0 if lt == FULL_ATTENTION else self.rope_parameters["rope_theta"]
                for lt in self.layer_types
            ],
        )

        tie = text.get("tie_word_embeddings", kwargs.pop("tie_word_embeddings", False))
        kwargs.setdefault("bos_token_id", text.get("bos_token_id", 200000))
        kwargs.setdefault("eos_token_id", text.get("eos_token_id", 200001))
        kwargs.setdefault("pad_token_id", text.get("pad_token_id", None))
        super().__init__(**kwargs)
        # AFTER super(): PretrainedConfig.__init__ unconditionally assigns
        # tie_word_embeddings (defaulting to True) and would clobber the text
        # config's False. Muse Glimmer ships a separate lm_head.weight, so
        # tying here would silently serve the embedding matrix as the LM head.
        self.tie_word_embeddings = tie

    # --- per-layer helpers ------------------------------------------------
    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == FULL_ATTENTION

    def is_nope(self, layer_idx: int) -> bool:
        """NoPE layers skip rotary entirely (HF passes position_embeddings=None)."""
        return not self.layer_rope_theta[layer_idx]

    def validate_supported(self):
        assert not self.attention_bias, "attention bias not supported"
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert len(self.layer_types) == self.num_hidden_layers
        assert len(self.layer_rope_theta) == self.num_hidden_layers
        assert set(self.layer_types) <= {FULL_ATTENTION, SLIDING_ATTENTION}, (
            f"unexpected layer types: {set(self.layer_types)}"
        )
        # Every rotary layer shares one theta in the released checkpoint; the
        # tokasaurus port builds a single cos/sin table on that assumption.
        thetas = {t for t in self.layer_rope_theta if t}
        assert len(thetas) <= 1, f"multiple rope thetas not supported: {thetas}"
        # NoPE is expected on full-attention layers only. A checkpoint that
        # breaks this would silently change which layers get rotary.
        for i, lt in enumerate(self.layer_types):
            if lt == SLIDING_ATTENTION:
                assert self.layer_rope_theta[i], (
                    f"sliding layer {i} is unexpectedly NoPE"
                )

    @property
    def rope_theta(self) -> float:
        thetas = {t for t in self.layer_rope_theta if t}
        return float(next(iter(thetas))) if thetas else 0.0


def register_muse_glimmer_config():
    try:
        AutoConfig.register("muse_glimmer", MuseGlimmerConfig)
    except ValueError:
        pass  # already registered
