"""Config, generation config and tokenizer loading for `hf_olmo` checkpoints.

Kept free of model/kernel imports so the server and manager processes can load the
tokenizer and generation config without pulling in flashinfer. The model itself is in
`tokasaurus/model/olmo.py`.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub

OLMO_MODEL_TYPE = "hf_olmo"


def read_config_dict(path: str) -> dict:
    """Read config.json from a local checkpoint directory or a HF hub repo."""
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        config_path = Path(
            huggingface_hub.hf_hub_download(path, filename="config.json")
        )
    with open(config_path) as f:
        return json.load(f)


def is_olmo_checkpoint(path: str) -> bool:
    """True if `path` holds an `hf_olmo` checkpoint (decided by its config.json)."""
    try:
        return read_config_dict(path).get("model_type") == OLMO_MODEL_TYPE
    except (OSError, ValueError):  # no config.json locally or on the hub
        return False


def check_supported(config_dict: dict):
    """Fail loudly on OLMo variants that the Llama-based blocks would compute wrongly."""
    expected = {
        "activation_type": "swiglu",
        "alibi": False,
        "attention_layer_norm": False,
        "block_type": "sequential",
        "clip_qkv": None,
        "include_bias": False,
        "layer_norm_type": "rms",
        "layer_norm_with_affine": True,
        "norm_after": False,
        "rope": True,
        "scale_logits": False,
        "weight_tying": False,
    }
    unsupported = {
        key: config_dict[key]
        for key, value in expected.items()
        if key in config_dict and config_dict[key] != value
    }
    if unsupported:
        raise NotImplementedError(
            f"Unsupported OLMo config values for tokasaurus' OLMo loader: {unsupported}"
        )


@dataclass
class OlmoConfig:
    """OLMo config, stored under the Llama attribute names tokasaurus reads."""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    num_key_value_heads: int | None = None
    rope_theta: float = 10000
    tie_word_embeddings: bool = False

    model_type: str = "olmo"

    # Kept for reference; tokasaurus does not use them at inference time.
    activation_type: str = "swiglu"
    rope_full_precision: bool = True

    @classmethod
    def from_pretrained(cls, path: str):
        config_dict = read_config_dict(path)
        check_supported(config_dict)

        return cls(
            hidden_size=config_dict["d_model"],
            num_attention_heads=config_dict["n_heads"],
            num_hidden_layers=config_dict["n_layers"],
            # OLMo's mlp_hidden_size/mlp_ratio count both halves of the fused SwiGLU
            # projection, so the Llama intermediate size is half of it.
            intermediate_size=(
                config_dict.get("mlp_hidden_size")
                or (config_dict["d_model"] * config_dict["mlp_ratio"])
            )
            // 2,
            # embedding_size is the padded vocab the weights are actually shaped by.
            vocab_size=config_dict.get("embedding_size") or config_dict["vocab_size"],
            max_position_embeddings=config_dict["max_sequence_length"],
            rms_norm_eps=config_dict["layer_norm_eps"],
            # DataDecide sets n_kv_heads to null, which means MHA.
            num_key_value_heads=config_dict.get("n_kv_heads") or config_dict["n_heads"],
            rope_theta=config_dict["rope_theta"],
            tie_word_embeddings=config_dict.get("weight_tying", False),
            activation_type=config_dict.get("activation_type", "swiglu"),
            rope_full_precision=config_dict.get("rope_full_precision", True),
        )


@dataclass
class OlmoGenerationConfig:
    """hf_olmo checkpoints ship no generation_config.json; the token ids live in config.json."""

    pad_token_id: int
    eos_token_id: int

    @classmethod
    def from_pretrained(cls, path: str):
        config_dict = read_config_dict(path)
        return cls(
            pad_token_id=config_dict["pad_token_id"],
            eos_token_id=config_dict["eos_token_id"],
        )


def load_olmo_tokenizer(path: str):
    """Load the ai2-olmo tokenizer wrapper for a hub id, checkpoint dir, or tokenizer.json.

    Intermediate training checkpoints are often saved without a tokenizer, so for a local
    directory this also searches parent directories, and finally falls back to the
    `allenai/<name>` hub repo embedded in the path (minus any `-seedN` suffix).
    """
    from olmo.tokenizer import Tokenizer  # ai2-olmo, installed via the [olmo] extra

    path_obj = Path(path)
    if not path_obj.exists():
        return Tokenizer.from_pretrained(path)

    if path_obj.is_file():
        return Tokenizer.from_file(str(path_obj))

    for directory in [path_obj, *path_obj.parents]:
        if (tokenizer_file := directory / "tokenizer.json").exists():
            return Tokenizer.from_file(str(tokenizer_file))

    if match := re.search(r"(allenai/[^/]+)", str(path)):
        hub_id = re.sub(r"-seed\d+$", "", match.group(1))
        return Tokenizer.from_pretrained(hub_id)

    raise FileNotFoundError(
        f"No tokenizer.json in {path} or its parents, and no allenai/<model> id in the path."
    )
