"""Boundary probe for AITER's causal paged-prefill bug on gfx950.

With causal=True at head_dim 128, `aiter.mha_batch_prefill_func` deviates 0.17-0.35 from
the fp32 oracle (correct bf16 rounding is ~8e-3), on roughly two runs in three. head_dim
64 and 256 and all non-causal calls were clean. AITER guards a bug of this shape for ROCm
7.2 only (op_tests/test_batch_prefill.py::should_skip_rocm72_issue); it reproduces on 7.1.

Results depend on earlier calls in the process and the failure is intermittent, so each
config runs in its own subprocess, repeated.

    python -m tools.amd_port.aiter_causal_bug --one 64 8 128 7 1   # one config
    python -m tools.amd_port.aiter_causal_bug                      # the whole matrix
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# (num_qo_heads, num_kv_heads, head_dim, seq_len, causal)
# head_dim 128 + causal is the bad region; everything else was clean in every trial.
MATRIX = [
    (64, 8, 128, 7, 1),  # qwen3-32b shape             -> intermittently bad
    (32, 8, 128, 7, 1),  # llama-3.1-8b shape          -> intermittently bad
    (64, 16, 128, 7, 1),  # GQA 4                       -> intermittently bad
    (16, 8, 128, 7, 1),  # GQA 2                       -> intermittently bad
    (8, 8, 128, 7, 1),  # GQA 1 (MHA)                 -> intermittently bad
    (32, 8, 128, 129, 1),  # longer                      -> intermittently bad
    (64, 8, 128, 129, 1),  # longer                      -> intermittently bad
    (64, 8, 128, 64, 1),  # page-aligned                -> intermittently bad
    (64, 8, 128, 7, 0),  # non-causal                  -> ok
    (64, 8, 64, 7, 1),  # head_dim 64                 -> ok
    (64, 8, 256, 7, 1),  # head_dim 256                -> ok
    (64, 8, 64, 129, 1),  # head_dim 64, longer         -> ok
    (64, 8, 256, 129, 1),  # head_dim 256, longer        -> ok
]

TOL = 2e-2


def run_one(hq: int, hkv: int, d: int, n: int, causal: int) -> int:

    from tools.amd_port.golden_spec import (
        AttnCase,
        build_inputs,
        reference_attention,
    )
    from tools.amd_port.probe_aiter import run_aiter

    case = AttnCase(
        name="probe",
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_dim=d,
        seq_lens=(n,),
        qo_lens=(n,),
        causal=bool(causal),
        page_size=16,
    )
    inp = build_inputs(case, device="cuda")
    out, _ = run_aiter(case, inp)
    ref, _ = reference_attention(case, inp)
    dev = (out.float() - ref.float()).abs().max().item()
    bad = dev > TOL
    print(
        f"hq={hq:>3} hkv={hkv:>3} gqa={hq // hkv:>2} head_dim={d:>3} n={n:>4} "
        f"causal={causal} -> max|out| {dev:9.2e}  {'FAIL' if bad else 'ok'}"
    )
    return int(bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--one", nargs=5, type=int, metavar=("HQ", "HKV", "D", "N", "CAUSAL")
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="processes per config; the bug is intermittent, so >1 matters",
    )
    args = ap.parse_args()

    if args.one:
        return run_one(*args.one)

    n_bad = 0
    for cfg in MATRIX:
        fails = 0
        worst = ""
        for _ in range(args.repeat):
            # Separate process per run: see the module docstring.
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.amd_port.aiter_causal_bug",
                    "--one",
                    *[str(x) for x in cfg],
                ],
                cwd=str(REPO),
                capture_output=True,
                check=False,
                text=True,
            )
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("hq=")]
            if not line:
                print(f"{cfg} -> no output\n{r.stderr[-500:]}")
                continue
            if "FAIL" in line[0]:
                fails += 1
                worst = line[0]
            elif not worst:
                worst = line[0]
        n_bad += fails > 0
        print(f"{worst.split(' -> ')[0]} -> FAILED {fails}/{args.repeat} runs")
    print(f"\n{n_bad}/{len(MATRIX)} configurations wrong on at least one run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
