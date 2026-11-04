"""Config / generation-config / tokenizer loading that works for both HF and hf_olmo models.

transformers' Auto* classes cannot read AI2's original `hf_olmo` checkpoints without
ai2-olmo's remote code, so those are routed to `olmo_config`. Gemma 4, Muse Glimmer,
GLM-4.6, Qwen 3.5 and gpt-oss configs are registered with AutoConfig, since the pinned transformers predates them.
"""

import json
from pathlib import Path

import huggingface_hub
from tokenizers import Tokenizer

from tokasaurus.model.gemma4_config import register_gemma4_config
from tokasaurus.model.glm4_moe_config import register_glm4_moe_config
from tokasaurus.model.gpt_oss_config import register_gpt_oss_config
from tokasaurus.model.muse_glimmer_config import register_muse_glimmer_config
from tokasaurus.model.olmo_config import (
    OlmoConfig,
    OlmoGenerationConfig,
    is_olmo_checkpoint,
    load_olmo_tokenizer,
)
from tokasaurus.model.qwen3_5_config import register_qwen3_5_config


def load_config(model_path: str, trust_remote_code: bool = False):
    if is_olmo_checkpoint(model_path):
        return OlmoConfig.from_pretrained(model_path)

    from transformers import AutoConfig

    # model types the pinned transformers predates
    register_gemma4_config()
    register_glm4_moe_config()
    register_gpt_oss_config()
    register_muse_glimmer_config()
    register_qwen3_5_config()
    return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def load_gen_config(model_path: str, trust_remote_code: bool = False):
    if is_olmo_checkpoint(model_path):
        return OlmoGenerationConfig.from_pretrained(model_path)

    from transformers import GenerationConfig

    return GenerationConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )


# tokenizer_class written by transformers >= 5, unknown to the 4.x pinned here
TRANSFORMERS5_TOKENIZER_CLASSES = {"TokenizersBackend"}
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token", "unk_token")


def read_tokenizer_file(path: str, filename: str) -> Path | None:
    local = Path(path) / filename
    if local.exists():
        return local
    if Path(path).exists():
        return None
    try:
        return Path(huggingface_hub.hf_hub_download(path, filename=filename))
    except huggingface_hub.errors.EntryNotFoundError:
        return None


def load_transformers5_tokenizer(path: str, tokenizer_config: dict):
    """Build a fast tokenizer from tokenizer.json for checkpoints saved by transformers 5
    (e.g. Gemma 4, Muse Glimmer), whose tokenizer_config.json 4.x cannot parse."""
    from transformers import PreTrainedTokenizerFast

    special_tokens = {
        k: v["content"] if isinstance(v, dict) else v
        for k in SPECIAL_TOKEN_KEYS
        if (v := tokenizer_config.get(k)) is not None
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(read_tokenizer_file(path, "tokenizer.json")),
        **special_tokens,
    )
    chat_template = tokenizer_config.get("chat_template")
    if chat_template is None and (
        jinja := read_tokenizer_file(path, "chat_template.jinja")
    ):
        chat_template = jinja.read_text()
    tokenizer.chat_template = chat_template
    return tokenizer


def load_tokenizer(tokenizer_path: str):
    if is_olmo_checkpoint(tokenizer_path):
        return load_olmo_tokenizer(tokenizer_path)

    if config_file := read_tokenizer_file(tokenizer_path, "tokenizer_config.json"):
        tokenizer_config = json.loads(config_file.read_text())
        if tokenizer_config.get("tokenizer_class") in TRANSFORMERS5_TOKENIZER_CLASSES:
            return load_transformers5_tokenizer(tokenizer_path, tokenizer_config)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def get_inner_tokenizer(tokenizer) -> Tokenizer:
    """The underlying `tokenizers.Tokenizer` (used for fast batch encode/decode)."""
    if hasattr(tokenizer, "base_tokenizer"):  # ai2-olmo wrapper
        return tokenizer.base_tokenizer
    return tokenizer._tokenizer  # transformers fast tokenizer


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Encode a completions prompt the way each tokenizer family expects.

    The ai2-olmo wrapper appends EOS by default; prompts must not end in EOS.
    """
    if hasattr(tokenizer, "base_tokenizer"):
        return tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer(prompt)["input_ids"]
