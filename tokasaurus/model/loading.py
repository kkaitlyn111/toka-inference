"""Config / generation-config / tokenizer loading that works for both HF and hf_olmo models.

transformers' Auto* classes cannot read AI2's original `hf_olmo` checkpoints without
ai2-olmo's remote code, so those are routed to `olmo_config`; everything else goes
through transformers as before.
"""

from tokenizers import Tokenizer

from tokasaurus.model.olmo_config import (
    OlmoConfig,
    OlmoGenerationConfig,
    is_olmo_checkpoint,
    load_olmo_tokenizer,
)


def load_config(model_path: str, trust_remote_code: bool = False):
    if is_olmo_checkpoint(model_path):
        return OlmoConfig.from_pretrained(model_path)

    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def load_gen_config(model_path: str, trust_remote_code: bool = False):
    if is_olmo_checkpoint(model_path):
        return OlmoGenerationConfig.from_pretrained(model_path)

    from transformers import GenerationConfig

    return GenerationConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )


def load_tokenizer(tokenizer_path: str):
    if is_olmo_checkpoint(tokenizer_path):
        return load_olmo_tokenizer(tokenizer_path)

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
