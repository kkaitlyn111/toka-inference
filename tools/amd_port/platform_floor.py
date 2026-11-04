"""The CUDA-vs-ROCm logit floor from stock HF transformers (SDPA), no tokasaurus.

cuBLAS and hipBLASLt reduce in different orders, so any port inherits some deviation.
Running the same model and prompts through HF on both platforms measures that floor;
only deviation beyond it is attributable to the tokasaurus ROCm backend.

    python -m tools.amd_port.platform_floor --out <file.pt>
"""

from __future__ import annotations

import argparse

import torch

PROMPTS = [
    "The capital of France is",
    "Q: What is 17 * 23? Think step by step.\nA:",
    "Once upon a time in a distant galaxy, there lived",
]

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )

    out = {}
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float().cpu()
        out[f"p{i}"] = logits
        top = logits.topk(5)
        print(
            f"p{i}: top5={[tok.decode([t]) for t in top.indices.tolist()]} "
            f"vals={[round(v, 4) for v in top.values.tolist()]}",
            flush=True,
        )

    # A long prompt as well: attention differences grow with KV length, so a
    # short-prompt-only floor would understate it.
    long_p = (
        "The following is a list of facts.\n"
        + "".join(
            f"Fact {i}: the value of item {i} is {(i * 7919) % 1000}.\n"
            for i in range(400)
        )
        + "Question: what is the value of item 137?\nAnswer:"
    )
    ids = tok(long_p, return_tensors="pt").input_ids.cuda()
    print(f"long prompt tokens: {ids.shape[1]}", flush=True)
    with torch.no_grad():
        out["long"] = model(ids).logits[0, -1].float().cpu()

    torch.save(out, args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
