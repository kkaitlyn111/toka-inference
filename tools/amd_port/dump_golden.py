"""Record NVIDIA/flashinfer ground truth for the AMD port.

Run this on a CUDA box with the reference env (torch 2.9.0+cu128,
flashinfer 0.2.2.post1). It writes one .pt per case containing flashinfer's
output and LSE, plus a manifest with the environment fingerprint.

It also cross-checks flashinfer against the dense fp32 reference in
golden_spec.py. That check validates the *reference*, which is what the ROCm
backend will ultimately be measured against, so a failure here means the
harness is wrong -- not the port.

Usage:
    python -m tools.amd_port.dump_golden --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.amd_port.golden_spec import (
    CASES,
    AttnCase,
    build_inputs,
    reference_attention,
)


def run_flashinfer(case: AttnCase, inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    q = inp["q"]
    k_cache, v_cache = inp["k_cache"], inp["v_cache"]

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

    out, lse = w.run_return_lse(q=q, paged_kv_cache=(k_cache, v_cache))
    return out, lse


def dump_merge_state_golden(out_dir: Path) -> dict:
    """cascade.merge_state ground truth + the base-2 LSE convention probe."""
    from flashinfer import cascade

    g = torch.Generator(device="cpu").manual_seed(1234)
    t, h, d = 64, 32, 128
    oa = (torch.randn(t, h, d, generator=g) * 0.5).to(torch.bfloat16).cuda()
    ob = (torch.randn(t, h, d, generator=g) * 0.5).to(torch.bfloat16).cuda()
    la = (torch.randn(t, h, generator=g) * 2.0).float().cuda()
    lb = (torch.randn(t, h, generator=g) * 2.0).float().cuda()

    m_out, m_lse = cascade.merge_state(oa, la, ob, lb)
    torch.save({"out": m_out.cpu(), "lse": m_lse.cpu()}, out_dir / "merge_state.pt")

    # Which log base does merge_state use? Compare against both.
    w2 = torch.exp2(la - torch.logaddexp2(la, lb)).unsqueeze(-1)
    ref2 = (oa.float() * w2 + ob.float() * (1 - w2)).to(oa.dtype)
    we = torch.exp(la - torch.logaddexp(la, lb)).unsqueeze(-1)
    refe = (oa.float() * we + ob.float() * (1 - we)).to(oa.dtype)
    err2 = (m_out.float() - ref2.float()).abs().max().item()
    erre = (m_out.float() - refe.float()).abs().max().item()
    return {"merge_base2_maxabs": err2, "merge_basee_maxabs": erre}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="substring filter on case name")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    manifest = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "device": torch.cuda.get_device_name(0),
        "cases": {},
    }

    cases = [c for c in CASES if not args.only or args.only in c.name]
    n_fail = 0
    for case in cases:
        inp = build_inputs(case, device="cuda")
        try:
            out, lse = run_flashinfer(case, inp)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {case.name}: {type(e).__name__}: {e}", flush=True)
            manifest["cases"][case.name] = {
                "status": "flashinfer_unsupported",
                "error": f"{type(e).__name__}: {e}",
                "spec": case.to_json(),
            }
            continue

        ref_out, ref_lse = reference_attention(case, inp)
        d_out = (out.float() - ref_out.float().to(out.device)).abs().max().item()
        # Compare only where the reference LSE is finite: a fully-masked row is
        # -inf on both sides and (-inf) - (-inf) is nan.
        lse_c = lse.float().cpu()
        ref_lse_c = ref_lse.float().cpu()
        finite = torch.isfinite(ref_lse_c)
        d_lse = (
            (lse_c[finite] - ref_lse_c[finite]).abs().max().item()
            if bool(finite.any())
            else 0.0
        )

        torch.save(
            {"out": out.cpu(), "lse": lse.cpu(), "spec": case.to_json()},
            out_dir / f"{case.name}.pt",
        )
        # bf16 has ~3 decimal digits; 2e-2 absolute on activations of O(1) is
        # kernel-vs-dense-fp32 rounding, not a semantic difference.
        ok = d_out < 2e-2 and d_lse < 5e-2
        n_fail += not ok
        manifest["cases"][case.name] = {
            "status": "ok" if ok else "REFERENCE_MISMATCH",
            "fi_vs_dense_out_maxabs": d_out,
            "fi_vs_dense_lse_maxabs": d_lse,
            "spec": case.to_json(),
        }
        print(
            f"[{'ok ' if ok else 'BAD'}] {case.name:42s} "
            f"out={d_out:.4g} lse={d_lse:.4g}",
            flush=True,
        )

    manifest["merge_state"] = dump_merge_state_golden(out_dir)
    print("merge_state:", manifest["merge_state"], flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(cases)} cases to {out_dir}; {n_fail} reference mismatches")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
