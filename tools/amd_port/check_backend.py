"""Compare the active attention backend against the recorded NVIDIA ground truth.

Runs the same cases `dump_golden.py` recorded, through whatever backend
`tokasaurus.model.attn_backend` selects for this machine, and reports the
deviation from flashinfer's output as well as from the dense fp32 reference.

On a CUDA box this should be ~0 (the backend re-exports flashinfer). On ROCm it
measures the actual port.

Usage:
    python -m tools.amd_port.check_backend --golden <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.amd_port.golden_spec import (
    CASES_BY_NAME,
    AttnCase,
    build_inputs,
    reference_attention,
)

# bf16 carries ~3 decimal digits. Two different kernels summing the same values in
# a different order land within ~1e-2 on activations of this magnitude; anything
# larger is a semantic difference, not rounding.
OUT_TOL = 2e-2
LSE_TOL = 5e-2


def run_backend(case: AttnCase, inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    from tokasaurus.model.attn_backend import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

    workspace = torch.empty(
        512 * 1024 * 1024, dtype=torch.uint8, device=inp["q"].device
    )
    q, k_cache, v_cache = inp["q"], inp["k_cache"], inp["v_cache"]

    if case.is_decode:
        gqa_ratio = case.num_qo_heads // case.num_kv_heads
        w = BatchDecodeWithPagedKVCacheWrapper(
            workspace, use_tensor_cores=gqa_ratio >= 4
        )
        w.plan(
            indptr=inp["kv_indptr"],
            indices=inp["kv_indices"],
            last_page_len=inp["last_page_len"],
            num_kv_heads=case.num_kv_heads,
            num_qo_heads=case.num_qo_heads,
            head_dim=case.head_dim,
            page_size=case.page_size,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            non_blocking=False,
        )
    else:
        w = BatchPrefillWithPagedKVCacheWrapper(workspace)
        w.plan(
            qo_indptr=inp["qo_indptr"],
            paged_kv_indptr=inp["kv_indptr"],
            paged_kv_indices=inp["kv_indices"],
            paged_kv_last_page_len=inp["last_page_len"],
            num_kv_heads=case.num_kv_heads,
            num_qo_heads=case.num_qo_heads,
            head_dim_qk=case.head_dim,
            page_size=case.page_size,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            causal=case.causal,
            non_blocking=False,
        )

    return w.run_return_lse(q=q, paged_kv_cache=(k_cache, v_cache))


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float().cpu() - b.float().cpu()).abs().max().item()


def _lse_maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    ac, bc = a.float().cpu(), b.float().cpu()
    finite = torch.isfinite(ac) & torch.isfinite(bc)
    if not bool(finite.any()):
        return 0.0
    return (ac[finite] - bc[finite]).abs().max().item()


def check_merge_state(golden_dir: Path) -> dict:
    from tokasaurus.model.attn_backend import cascade

    path = golden_dir / "merge_state.pt"
    if not path.exists():
        return {"status": "no_golden"}
    gold = torch.load(path, map_location="cpu")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(1234)
    t, h, d = 64, 32, 128
    oa = (torch.randn(t, h, d, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    ob = (torch.randn(t, h, d, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    la = (torch.randn(t, h, generator=g) * 2.0).float().to(dev)
    lb = (torch.randn(t, h, generator=g) * 2.0).float().to(dev)

    out, lse = cascade.merge_state(oa, la, ob, lb)
    return {
        "status": "ok",
        "out_maxabs": _maxdiff(out, gold["out"]),
        "lse_maxabs": _lse_maxdiff(lse, gold["lse"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    golden_dir = Path(args.golden)
    from tokasaurus.model.attn_backend import BACKEND

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"backend={BACKEND} torch={torch.__version__} hip={torch.version.hip} device={dev}"
    )

    files = sorted(p for p in golden_dir.glob("*.pt") if p.stem != "merge_state")
    if args.only:
        files = [p for p in files if args.only in p.stem]

    n_pass = n_fail = 0
    rows = []
    for path in files:
        name = path.stem
        case = CASES_BY_NAME.get(name)
        if case is None:
            print(f"[skip] {name}: not in current spec")
            continue
        gold = torch.load(path, map_location="cpu")
        inp = build_inputs(case, device=dev)

        try:
            out, lse = run_backend(case, inp)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name:42s} {type(e).__name__}: {e}")
            n_fail += 1
            rows.append({"case": name, "status": "error", "error": str(e)})
            continue

        ref_out, ref_lse = reference_attention(case, inp)
        d_gold = _maxdiff(out, gold["out"])
        d_gold_lse = _lse_maxdiff(lse, gold["lse"])
        d_ref = _maxdiff(out, ref_out)
        d_ref_lse = _lse_maxdiff(lse, ref_lse)

        ok = d_gold < OUT_TOL and d_gold_lse < LSE_TOL
        n_pass += ok
        n_fail += not ok
        print(
            f"[{'ok ' if ok else 'BAD'}] {name:42s} "
            f"vs_gold out={d_gold:.4g} lse={d_gold_lse:.4g} | "
            f"vs_dense out={d_ref:.4g} lse={d_ref_lse:.4g}"
        )
        rows.append(
            {
                "case": name,
                "status": "ok" if ok else "FAIL",
                "vs_golden_out_maxabs": d_gold,
                "vs_golden_lse_maxabs": d_gold_lse,
                "vs_dense_out_maxabs": d_ref,
                "vs_dense_lse_maxabs": d_ref_lse,
            }
        )

    merge = check_merge_state(golden_dir)
    print("merge_state:", merge)
    if merge.get("status") == "ok":
        m_ok = merge["out_maxabs"] < OUT_TOL and merge["lse_maxabs"] < LSE_TOL
        n_pass += m_ok
        n_fail += not m_ok

    report = {
        "backend": BACKEND,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "passed": n_pass,
        "failed": n_fail,
        "merge_state": merge,
        "cases": rows,
    }
    out_path = golden_dir.parent / f"check_{BACKEND}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n{n_pass} passed, {n_fail} failed -> {out_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
