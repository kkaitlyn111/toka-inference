"""Diff two serve_e2e.sh outputs (e.g. CUDA vs ROCm).

Greedy decoding is deterministic given identical math, so an exact match is the
ideal. Where kernels differ by rounding, a single near-tie logit flips one token
and everything after it diverges -- so the useful metric is the *prefix agreement
length*, not a boolean.

Usage:
    python -m tools.amd_port.compare_e2e <a.json> <b.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())

    ra, rb = a["results"], b["results"]
    assert len(ra) == len(rb), f"prompt count differs: {len(ra)} vs {len(rb)}"

    exact = 0
    total_chars = matched_chars = 0
    print(f"{'prompt':44s} {'exact':>6s} {'prefix':>8s} {'len':>6s}")
    print("-" * 68)
    for x, y in zip(ra, rb, strict=True):
        assert x["prompt"] == y["prompt"], "prompt order differs"
        ta, tb = x["text"], y["text"]
        p = common_prefix_len(ta, tb)
        is_exact = ta == tb
        exact += is_exact
        total_chars += max(len(ta), len(tb))
        matched_chars += p
        print(
            f"{x['prompt'][:44]!r:44s} {'YES' if is_exact else 'no':>6s} {p:>8d} {max(len(ta), len(tb)):>6d}"
        )
        if not is_exact:
            print(f"    A: ...{ta[max(0, p - 30) : p + 40]!r}")
            print(f"    B: ...{tb[max(0, p - 30) : p + 40]!r}")

    print("-" * 68)
    pct = 100.0 * matched_chars / max(total_chars, 1)
    print(f"exact: {exact}/{len(ra)} prompts | char prefix agreement: {pct:.1f}%")

    la, lb = a.get("logprobs"), b.get("logprobs")
    if la and lb:
        print("\nsingle-step top-20 logprobs (no autoregressive amplification)")
        print(f"{'probe':10s} {'top1==':>7s} {'overlap':>8s} {'max|dlogp|':>11s}")
        print("-" * 68)
        n_top1 = n_probe = 0
        worst = 0.0
        for x, y in zip(la, lb, strict=True):
            if "error" in x or "error" in y:
                print(f"{x['name']:10s} {'ERR':>7s}")
                continue
            n_probe += 1
            same_top1 = x["top1"] == y["top1"]
            n_top1 += same_top1
            ka, kb = x["top_logprobs"], y["top_logprobs"]
            shared = set(ka) & set(kb)
            overlap = len(shared) / max(len(ka), 1)
            dmax = max((abs(ka[k] - kb[k]) for k in shared), default=0.0)
            worst = max(worst, dmax)
            print(
                f"{x['name']:10s} {'YES' if same_top1 else 'NO':>7s} "
                f"{overlap * 100:7.0f}% {dmax:11.3e}"
            )
        print("-" * 68)
        print(
            f"top-1 agreement: {n_top1}/{n_probe} | "
            f"worst logprob deviation across all shared tokens: {worst:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
