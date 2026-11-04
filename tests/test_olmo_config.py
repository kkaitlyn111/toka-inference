"""CPU-only unit tests for the hf_olmo config mapping and weight loading.

These build tiny synthetic checkpoints in a temp dir, so they need no GPU and no network:

    pytest tests/test_olmo_config.py
"""

import json

import pytest
import torch

from tokasaurus.model.olmo_config import (
    OlmoConfig,
    OlmoGenerationConfig,
    check_supported,
    is_olmo_checkpoint,
)

# Trimmed copy of allenai/DataDecide-dolma1_6plus-1B's config.json.
DATADECIDE_1B = {
    "model_type": "hf_olmo",
    "activation_type": "swiglu",
    "alibi": False,
    "attention_layer_norm": False,
    "block_type": "sequential",
    "clip_qkv": None,
    "d_model": 2048,
    "embedding_size": 50304,
    "eos_token_id": 50279,
    "include_bias": False,
    "layer_norm_eps": 1e-06,
    "layer_norm_type": "rms",
    "layer_norm_with_affine": True,
    "max_sequence_length": 2048,
    "mlp_hidden_size": None,
    "mlp_ratio": 8,
    "n_heads": 16,
    "n_kv_heads": None,
    "n_layers": 16,
    "norm_after": False,
    "pad_token_id": 1,
    "rope": True,
    "rope_full_precision": True,
    "rope_theta": 10000,
    "scale_logits": False,
    "vocab_size": 50280,
    "weight_tying": False,
}


def write_config(tmp_path, **overrides):
    config = {**DATADECIDE_1B, **overrides}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_config_maps_to_llama_names(tmp_path):
    config = OlmoConfig.from_pretrained(write_config(tmp_path))
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 16  # n_kv_heads null => MHA
    assert config.num_hidden_layers == 16
    assert config.intermediate_size == 2048 * 8 // 2  # one half of the fused SwiGLU
    assert config.vocab_size == 50304  # padded embedding size, not the tokenizer size
    assert config.max_position_embeddings == 2048
    assert config.model_type == "olmo"


def test_mlp_hidden_size_counts_both_swiglu_halves(tmp_path):
    config = OlmoConfig.from_pretrained(write_config(tmp_path, mlp_hidden_size=11008))
    assert config.intermediate_size == 11008 // 2


def test_gqa(tmp_path):
    config = OlmoConfig.from_pretrained(write_config(tmp_path, n_kv_heads=4))
    assert config.num_key_value_heads == 4


@pytest.mark.parametrize(
    "override",
    [
        {"weight_tying": True},
        {"attention_layer_norm": True},
        {"clip_qkv": 8.0},
        {"layer_norm_type": "default"},
        {"include_bias": True},
    ],
)
def test_rejects_unsupported_variants(override):
    with pytest.raises(NotImplementedError):
        check_supported({**DATADECIDE_1B, **override})


def test_generation_config(tmp_path):
    gen = OlmoGenerationConfig.from_pretrained(write_config(tmp_path))
    assert gen.eos_token_id == 50279
    assert gen.pad_token_id == 1


def test_detection(tmp_path):
    assert is_olmo_checkpoint(write_config(tmp_path))
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert not is_olmo_checkpoint(str(hf_dir))
    assert not is_olmo_checkpoint(str(tmp_path / "does-not-exist"))


TINY = dict(
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    mlp_ratio=4,
    embedding_size=128,
    vocab_size=120,
    max_sequence_length=64,
)


def write_tiny_checkpoint(tmp_path):
    """A random hf_olmo checkpoint with OLMo's fused att_proj / ff_proj layout."""
    from safetensors.torch import save_file

    d, h, kv, ff = 64, 4, 2, 64 * 4  # ff = both SwiGLU halves
    head_dim = d // h
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    tensors = {
        "model.transformer.wte.weight": rand(128, d),
        "model.transformer.ln_f.weight": rand(d),
        "model.transformer.ff_out.weight": rand(128, d),
    }
    for i in range(2):
        b = f"model.transformer.blocks.{i}"
        tensors[f"{b}.att_proj.weight"] = rand((h + 2 * kv) * head_dim, d)
        tensors[f"{b}.attn_out.weight"] = rand(d, d)
        tensors[f"{b}.attn_norm.weight"] = rand(d)
        tensors[f"{b}.ff_norm.weight"] = rand(d)
        tensors[f"{b}.ff_proj.weight"] = rand(ff, d)
        tensors[f"{b}.ff_out.weight"] = rand(d, ff // 2)

    save_file(tensors, str(tmp_path / "model.safetensors"))
    write_config(tmp_path, **TINY)
    return tensors


def test_weight_loading_splits_fused_projections(tmp_path):
    pytest.importorskip("flashinfer")  # the Llama blocks import it at module level
    from tokasaurus.model.olmo import OlmoForCausalLM

    hf = write_tiny_checkpoint(tmp_path)
    model = OlmoForCausalLM.from_pretrained(
        str(tmp_path), device="cpu", dtype=torch.float32
    )

    params = dict(model.named_parameters())
    # every parameter is covered by the name map (nothing left on the meta device)
    assert set(params) == set(model.make_name_to_hf_name())
    assert all(p.device.type == "cpu" for p in params.values())

    head_dim = 64 // 4
    q_size, kv_size = 4 * head_dim, 2 * head_dim
    for i in range(2):
        att = hf[f"model.transformer.blocks.{i}.att_proj.weight"]
        attn = f"model.layers.{i}.self_attn"
        torch.testing.assert_close(params[f"{attn}.q_proj.weight"], att[:q_size])
        torch.testing.assert_close(
            params[f"{attn}.k_proj.weight"], att[q_size : q_size + kv_size]
        )
        torch.testing.assert_close(
            params[f"{attn}.v_proj.weight"], att[q_size + kv_size :]
        )

        ff = hf[f"model.transformer.blocks.{i}.ff_proj.weight"]
        mlp = f"model.layers.{i}.mlp"
        # OLMo computes `x, gate = ff_proj(x).chunk(2)`: up first, gate second
        torch.testing.assert_close(params[f"{mlp}.up_proj.weight"], ff[:128])
        torch.testing.assert_close(params[f"{mlp}.gate_proj.weight"], ff[128:])

    torch.testing.assert_close(
        params["model.embed_tokens.embed_tokens.weight"],
        hf["model.transformer.wte.weight"],
    )
    torch.testing.assert_close(
        params["lm_head.lm_head.weight"], hf["model.transformer.ff_out.weight"]
    )
