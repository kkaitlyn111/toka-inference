"""Call AITER paged attention directly (no tokasaurus wrappers) against the dense fp32
oracle, and measure its LSE convention rather than assuming it.

    python -m tools.amd_port.probe_aiter [--case NAME] [--page-size 16]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.amd_port.golden_spec import (
    CASES,
    build_inputs,
    reference_attention,
)

LOG2_E = 1.4426950408889634


def run_aiter(case, inp):
    from tokasaurus.model.attn_backend.aiter_attention import (
        AiterPagedPlan,
        aiter_paged_attention,
    )

    plan = AiterPagedPlan(
        kv_indptr=inp["kv_indptr"],
        kv_indices=inp["kv_indices"],
        last_page_len=inp["last_page_len"],
        page_size=case.page_size,
        num_qo_heads=case.num_qo_heads,
        qo_indptr=None if case.is_decode else inp["qo_indptr"],
        causal=case.causal,
        sm_scale=1.0 / math.sqrt(case.head_dim),
    )
    return aiter_paged_attention(inp["q"], inp["k_cache"], inp["v_cache"], plan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="substring filter on case name")
    ap.add_argument("--out-tol", type=float, default=2e-2)
    ap.add_argument("--lse-tol", type=float, default=5e-2)
    args = ap.parse_args()

    cases = [c for c in CASES if args.case is None or args.case in c.name]
    print(f"running {len(cases)} cases\n")

    # Measured once over every case: is AITER's LSE base-2 or natural log?
    # Ratio of AITER's LSE to the reference's BASE-2 LSE is 1.0 if base 2, and
    # 1/log2(e) = 0.693 if natural log.
    ratios = []
    n_fail = 0
    rows = []

    for case in cases:
        inp = build_inputs(case, device="cuda")
        try:
            out, lse = run_aiter(case, inp)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {case.name}: {type(e).__name__}: {e}")
            n_fail += 1
            continue

        ref_out, ref_lse = reference_attention(case, inp)

        od = (out.float() - ref_out.float()).abs().max().item()
        finite = torch.isfinite(ref_lse) & torch.isfinite(lse)
        ld = (lse[finite] - ref_lse[finite]).abs().max().item() if finite.any() else 0.0

        if finite.any():
            r = (lse[finite] / ref_lse[finite].clamp(min=1e-6)).median().item()
            if math.isfinite(r):
                ratios.append(r)

        ok = od <= args.out_tol and ld <= args.lse_tol
        n_fail += not ok
        rows.append((case.name, od, ld, ok))
        print(
            f"{'ok ' if ok else 'FAIL'} {case.name:<42} out {od:9.2e}   lse {ld:9.2e}"
        )

    print()
    if ratios:
        med = sorted(ratios)[len(ratios) // 2]
        conv = (
            "BASE 2 (already matches flashinfer)"
            if abs(med - 1.0) < 0.02
            else "NATURAL LOG (needs x1.4426950408889634)"
            if abs(med - 1 / LOG2_E) < 0.02
            else f"UNKNOWN (ratio {med:.4f})"
        )
        print(f"LSE convention after conversion: ratio-to-base2 = {med:.5f} -> {conv}")
    n_pass = sum(ok for _, _, _, ok in rows)
    n_err = len(cases) - len(rows)
    print(
        f"\n{n_pass}/{len(cases)} cases pass, {len(rows) - n_pass} mismatched, "
        f"{n_err} raised (out tol {args.out_tol}, lse tol {args.lse_tol})"
    )
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
