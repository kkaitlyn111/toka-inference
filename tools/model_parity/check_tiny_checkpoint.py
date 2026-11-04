"""Compare tokasaurus against the HF reference logits for a tiny checkpoint.

Loads the checkpoint through `from_pretrained` (config shim + weight-name mapping),
prefills all but the last token in two chunks, then decodes the last token, all through
tokasaurus' paged attention (and recurrent-state slots for hybrid models), and compares
the full next-token distributions to the reference written by make_tiny_checkpoints.py.

    python -m tools.model_parity.check_tiny_checkpoint tests/fixtures/tiny_models/gemma4
    python -m tools.model_parity.check_tiny_checkpoint <dir> --backend portable   # CPU

Prints one JSON line with the metrics; exits non-zero if they exceed the tolerances.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

PAGE_SIZE = 16
# use a nonzero slot so a slot-indexing mistake can't pass by accident
STATE_SLOTS = 3
# fp32 on the portable backend should agree to rounding; bf16 flashinfer kernels
# reorder reductions, so allow a looser bound there
TOLERANCES = {
    "portable": {"max_abs_logprob_diff": 1e-3, "min_argmax_agreement": 1.0},
    "flashinfer": {"max_mean_kl": 5e-3, "min_argmax_agreement": 0.95},
}


def page_info(PageInformation, torch, *, start: int, end: int, device, decode: bool):
    """Page table for one sequence whose cache holds positions [0, end), querying [start, end)."""
    num_pages = math.ceil(end / PAGE_SIZE)
    return PageInformation(
        qo_indptr=None
        if decode
        else torch.tensor([0, end - start], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        kv_indices=torch.arange(num_pages, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor(
            [end - (num_pages - 1) * PAGE_SIZE], dtype=torch.int32, device=device
        ),
        num_seqs=1,
        num_tokens=end - start,
    )


def empty_page_info(PageInformation, torch, device, decode: bool):
    i32 = dict(dtype=torch.int32, device=device)
    return PageInformation(
        qo_indptr=None if decode else torch.zeros(1, **i32),
        kv_indptr=torch.zeros(1, **i32),
        kv_indices=torch.zeros(0, **i32),
        kv_last_page_len=torch.zeros(0, **i32),
        num_seqs=0,
        num_tokens=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--backend", choices=["flashinfer", "portable"], default="flashinfer"
    )
    args = parser.parse_args()
    if args.backend == "portable":
        os.environ["TOKA_PORTABLE_ATTN"] = "1"

    import torch

    from tokasaurus.model.gemma4 import Gemma4ForCausalLM
    from tokasaurus.model.glm4_moe import Glm4MoeForCausalLM
    from tokasaurus.model.gpt_oss import GptOssForCausalLM
    from tokasaurus.model.muse_glimmer import MuseGlimmerForCausalLM
    from tokasaurus.model.qwen3_5 import Qwen35ForCausalLM
    from tokasaurus.model.types import (
        AttentionInfo,
        BatchSamplingParams,
        BatchState,
        ExtraModelConfig,
        PageInformation,
    )

    model_classes = {
        "gemma4": Gemma4ForCausalLM,
        "muse_glimmer": MuseGlimmerForCausalLM,
        "glm4_moe": Glm4MoeForCausalLM,
        "qwen3_5": Qwen35ForCausalLM,
        "gpt_oss": GptOssForCausalLM,
    }
    config = json.loads((args.model_dir / "config.json").read_text())
    vocab = config.get("text_config", config)["vocab_size"]
    ref = torch.load(args.model_dir / "reference.pt")
    input_ids, ref_logits = ref["input_ids"], ref["logits"]
    seq_len = len(input_ids)

    device = "cuda" if args.backend == "flashinfer" else "cpu"
    dtype = torch.bfloat16 if args.backend == "flashinfer" else torch.float32
    torch.set_default_dtype(dtype)
    model = model_classes[config["model_type"]].from_pretrained(
        str(args.model_dir),
        extra_config=ExtraModelConfig(
            topk_logprobs=vocab, max_seqs_per_forward=STATE_SLOTS
        ),
        device=device,
        dtype=dtype,
    )
    model.setup_caches(
        num_pages=math.ceil(seq_len / PAGE_SIZE) + 1, page_size=PAGE_SIZE
    )

    def run(start: int, end: int, decode: bool, slot: int = STATE_SLOTS - 1):
        num = end - start
        query = page_info(
            PageInformation, torch, start=start, end=end, device=device, decode=decode
        )
        empty = empty_page_info(PageInformation, torch, device, decode=not decode)
        attn_info = AttentionInfo(
            page_size=PAGE_SIZE,
            append_kv_token_indices=torch.arange(start, end, device=device),
            prefill_info=empty if decode else query,
            decode_info=query if decode else empty,
            prefill_state_slots=torch.tensor([] if decode else [slot]),
            prefill_has_init_state=torch.tensor(
                [] if decode else [start > 0], dtype=torch.bool
            ),
            decode_state_slots=torch.tensor([slot] if decode else [], dtype=torch.long),
        )
        batch_state = BatchState(
            input_ids=input_ids[start:end].to(device),
            position_ids=torch.arange(start, end, device=device),
            attention_info=attn_info,
            lm_head_indices=torch.arange(num, device=device),
            sampling_params=BatchSamplingParams(
                temperature=torch.ones(num, device=device),
                greedy_mask=torch.ones(num, dtype=torch.bool, device=device),
            ),
        )
        model.plan(attn_info)
        with torch.inference_mode():
            out = model(batch_state).outputs
        logprobs = torch.full((num, vocab), float("-inf"), device=device)
        logprobs.scatter_(1, out.topk_indices.long(), out.topk_logprobs.float())
        return logprobs.cpu()

    # chunked prefill [0, S/2) + [S/2, S-1), then decode position S-1
    half = seq_len // 2
    toka = torch.cat(
        [
            run(0, half, decode=False),
            run(half, seq_len - 1, decode=False),
            run(seq_len - 1, seq_len, decode=True),
        ]
    )
    ref_logprobs = torch.log_softmax(ref_logits.float(), dim=-1)

    finite = torch.isfinite(toka)
    metrics = {
        "model": config["model_type"],
        "backend": args.backend,
        "positions": seq_len,
        "max_abs_logprob_diff": float((toka - ref_logprobs)[finite].abs().max()),
        "mean_kl": float(
            (ref_logprobs.exp() * (ref_logprobs - toka.clamp(min=-1e4))).sum(-1).mean()
        ),
        "argmax_agreement": float(
            (toka.argmax(-1) == ref_logprobs.argmax(-1)).float().mean()
        ),
        "decode_step_argmax_match": bool(
            toka[-1].argmax() == ref_logprobs[-1].argmax()
        ),
    }
    print(json.dumps(metrics))

    tol = TOLERANCES[args.backend]
    ok = metrics["argmax_agreement"] >= tol["min_argmax_agreement"]
    if "max_abs_logprob_diff" in tol:
        ok &= metrics["max_abs_logprob_diff"] <= tol["max_abs_logprob_diff"]
    if "max_mean_kl" in tol:
        ok &= metrics["mean_kl"] <= tol["max_mean_kl"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
