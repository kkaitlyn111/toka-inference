"""Build tiny random checkpoints plus HF reference logits for the natively ported models
(Gemma 4, Muse Glimmer, GLM-4.6, Qwen 3.5).

Run under transformers >= 5.15, which implements all of them:

    python tools/model_parity/make_tiny_checkpoints.py --out tests/fixtures/tiny_models

Each model is written in its released checkpoint layout (for Gemma 4 and Muse Glimmer,
a multimodal-wrapper config.json with a nested text_config and weights under
model.language_model.) so the test exercises tokasaurus' config shims and weight-name
mapping, not just the math.
Every parameter is randomized: default init leaves norm scales at exactly 1 (or 0
for centered norms), which would hide a norm-formula mistake. The window is far
shorter than the sequence so the sliding/full layer split matters.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

SEQ_LEN = 96
VOCAB = 256
PARAM_STD = 0.05

GEMMA4_TEXT = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=6,  # 5 sliding + 1 full, the released 5:1 pattern
    num_attention_heads=4,
    num_key_value_heads=2,
    num_global_key_value_heads=1,
    head_dim=32,
    global_head_dim=512,  # > 256 exercises the dense prefill / CUDA-core decode path
    vocab_size=VOCAB,
    max_position_embeddings=512,
    sliding_window=16,
    attention_k_eq_v=True,
    final_logit_softcapping=30.0,
    tie_word_embeddings=True,
    # off in gemma-4-31B, on by default in Gemma4TextConfig
    hidden_size_per_layer_input=0,
    num_kv_shared_layers=0,
    enable_moe_block=False,
    use_double_wide_mlp=False,
    bos_token_id=2,
    eos_token_id=1,
    pad_token_id=0,
)

MUSE_GLIMMER_TEXT = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=8,  # [sliding x3, full] x 2
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=VOCAB,
    max_position_embeddings=512,
    sliding_window=16,
    qk_scale_factor=3.87,
    output_multiplier=0.19611613513818404,
    final_logit_softcapping=20.0,
    tie_word_embeddings=False,
    bos_token_id=2,
    eos_token_id=1,
)

GLM4_MOE = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    first_k_dense_replace=1,  # one dense layer, three MoE layers
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    partial_rotary_factor=0.5,
    attention_bias=True,
    use_qk_norm=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    moe_intermediate_size=64,
    vocab_size=VOCAB,
    max_position_embeddings=512,
    tie_word_embeddings=False,
)

QWEN3_5_TEXT = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,  # 3 gated-DeltaNet layers, then 1 attention layer
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    vocab_size=VOCAB,
    max_position_embeddings=512,
    rope_parameters=dict(
        rope_type="default",
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        mrope_interleaved=True,
        mrope_section=[2, 1, 1],
    ),
    tie_word_embeddings=False,
)

def randomize(module, g):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * PARAM_STD)
        for name, buf in module.named_buffers():
            if name.endswith("layer_scalar"):
                buf.copy_(torch.rand(buf.shape, generator=g) + 0.5)


def gemma4_reference(text_config, g, input_ids):
    from transformers import Gemma4ForCausalLM

    model = Gemma4ForCausalLM(text_config).float().eval()
    randomize(model, g)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    # tied lm_head; text-model weights sit under model.language_model. in the release
    weights = {
        "model.language_model." + k[len("model.") :]: v
        for k, v in model.state_dict().items()
        if k.startswith("model.")
    }
    return weights, logits


def muse_glimmer_reference(text_config, g, input_ids):
    # transformers ships Muse Glimmer only as a text model inside the multimodal
    # wrapper; the head below is MuseGlimmerForConditionalGeneration.forward's:
    # T * tanh(lm_head(h) * output_multiplier / T)
    from transformers import MuseGlimmerTextModel

    model = MuseGlimmerTextModel(text_config).float().eval()
    randomize(model, g)
    lm_head = (
        torch.randn(text_config.vocab_size, text_config.hidden_size, generator=g)
        * PARAM_STD
    )
    with torch.no_grad():
        hidden = model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        cap = text_config.final_logit_softcapping
        logits = (
            torch.tanh(hidden @ lm_head.T * text_config.output_multiplier / cap) * cap
        )
    weights = {"model.language_model." + k: v for k, v in model.state_dict().items()}
    weights["lm_head.weight"] = lm_head
    return weights, logits


def glm4_moe_reference(config, g, input_ids):
    from transformers import Glm4MoeForCausalLM

    model = Glm4MoeForCausalLM(config).float().eval()
    randomize(model, g)
    with torch.no_grad():
        # the router's correction bias is a buffer; give it non-trivial values too
        for name, buf in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buf.copy_(torch.randn(buf.shape, generator=g) * PARAM_STD)
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    return unfuse_experts(model.state_dict()), logits


def unfuse_experts(state_dict: dict) -> dict:
    """transformers 5 holds experts fused ([E, 2I, H] gate|up, [E, H, I] down); released
    checkpoints store one gate/up/down weight per expert."""
    out = {}
    for name, tensor in state_dict.items():
        if name.endswith("mlp.experts.gate_up_proj"):
            prefix = name[: -len("gate_up_proj")]
            for e, expert in enumerate(tensor):
                gate, up = expert.chunk(2, dim=0)
                out[f"{prefix}{e}.gate_proj.weight"] = gate
                out[f"{prefix}{e}.up_proj.weight"] = up
        elif name.endswith("mlp.experts.down_proj"):
            prefix = name[: -len("down_proj")]
            for e, down in enumerate(tensor):
                out[f"{prefix}{e}.down_proj.weight"] = down
        else:
            out[name] = tensor
    return out


def build_glm4_moe(out_dir: Path, seed: int):
    from transformers import Glm4MoeConfig

    config = Glm4MoeConfig(**GLM4_MOE)
    config._attn_implementation = "eager"
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN), generator=g)
    weights, logits = glm4_moe_reference(config, g, input_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in weights.items()},
        str(out_dir / "model.safetensors"),
    )
    config.save_pretrained(out_dir)
    # released GLM configs keep the rope settings top-level
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps({**json.loads(config_path.read_text()), **GLM4_MOE}, indent=2)
    )
    torch.save(
        {"input_ids": input_ids[0], "logits": logits.float()}, out_dir / "reference.pt"
    )
    print(f"glm4_moe: {sum(v.numel() for v in weights.values())} params -> {out_dir}")


def qwen3_5_reference(text_config, g, input_ids):
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM(text_config).float().eval()
    randomize(model, g)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    # released checkpoints are vision-language wrappers: LM weights under
    # model.language_model., lm_head at the top level
    weights = {
        (
            "model.language_model." + k[len("model.") :]
            if k.startswith("model.")
            else k
        ): v
        for k, v in model.state_dict().items()
    }
    return weights, logits


def build(model_type: str, text_kwargs: dict, out_dir: Path, seed: int):
    import transformers

    prefix = {"gemma4": "Gemma4", "muse_glimmer": "MuseGlimmer", "qwen3_5": "Qwen3_5"}[
        model_type
    ]
    text_config = getattr(transformers, f"{prefix}TextConfig")(**text_kwargs)
    text_config._attn_implementation = "eager"

    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN), generator=g)
    make_reference = {
        "gemma4": gemma4_reference,
        "muse_glimmer": muse_glimmer_reference,
        "qwen3_5": qwen3_5_reference,
    }[model_type]
    weights, logits = make_reference(text_config, g, input_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in weights.items()},
        str(out_dir / "model.safetensors"),
    )
    # transformers 5.15 folds some fields into per_layer_config on save; write them the
    # way the released checkpoints do
    text_dict = {
        k: v for k, v in text_config.to_dict().items() if k != "per_layer_config"
    }
    text_dict.update(text_kwargs)
    config = {
        "model_type": model_type,
        "text_config": text_dict,
        "tie_word_embeddings": text_kwargs["tie_word_embeddings"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(
        {"input_ids": input_ids[0], "logits": logits.float()}, out_dir / "reference.pt"
    )
    print(
        f"{model_type}: {sum(v.numel() for v in weights.values())} params -> {out_dir}"
    )


def main():
    builders = {
        "gemma4": lambda out: build("gemma4", GEMMA4_TEXT, out, seed=0),
        "muse_glimmer": lambda out: build(
            "muse_glimmer", MUSE_GLIMMER_TEXT, out, seed=1
        ),
        "glm4_moe": lambda out: build_glm4_moe(out, seed=2),
        "qwen3_5": lambda out: build("qwen3_5", QWEN3_5_TEXT, out, seed=3),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=list(builders), default=list(builders)
    )
    args = parser.parse_args()
    for name in args.models:
        builders[name](args.out / name)


if __name__ == "__main__":
    main()
