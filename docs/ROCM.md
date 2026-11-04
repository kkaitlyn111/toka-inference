# Tokasaurus on AMD (ROCm)

Runs tokasaurus on AMD MI350X (gfx950, CDNA4, ROCm 7.x) without changing the NVIDIA
code path.

## Design

flashinfer is CUDA-only (nvcc-compiled kernels, nvrtc JIT). Tokasaurus uses five of
its entry points: the paged prefill and decode wrappers (`plan` / `run` /
`run_return_lse`), `cascade.merge_state` (hydragen), and `sampling_from_probs`. Every
call site now imports them from a dispatch seam, `tokasaurus/model/attn_backend/`:

* **on CUDA** it re-exports the real flashinfer symbols, so NVIDIA behaviour is
  byte-for-byte what it was;
* **on ROCm** it substitutes a portable implementation, selected by `TOKA_ROCM_ATTN`:
  * `aiter` (default): AMD's AITER `mha_batch_prefill_func`, one kernel launch for the
    whole ragged batch (`aiter_attention.py`);
  * `reference`: a correctness-first, flash-style online-softmax implementation in
    plain torch with fp32 accumulation (`paged_attention.py`). It is kept on purpose:
    it is the numerical oracle the AITER path is validated against, and the fallback
    for shapes AITER rejects (head_dim > 256).

CUDA graphs are disabled on ROCm (`ServerConfig.finalize`), because the backend reads
page-table lengths on the host in `plan()`, and a replayed graph would silently use a
stale plan.

## Correctness

The question is not "does it match" but "does the port add error beyond the
platform's own floor". The floor was measured with stock HF transformers (SDPA, no
tokasaurus) on both machines, since cuBLAS and hipBLASLt choose different GEMM
reduction orders:

| measurement (Qwen3-4B-Instruct-2507, NVIDIA L40S vs MI350X) | worst abs logprob deviation |
|---|---|
| tokasaurus ROCm port vs tokasaurus NVIDIA | 0.243 |
| platform floor: stock HF on MI350X vs stock HF on L40S | 0.213 to 0.264 |
| NVIDIA vs NVIDIA (control) | 0.000 |

The NVIDIA-vs-NVIDIA control shows that the measurement can resolve a difference, so
0.243 is a real reading, and it sits inside the floor.

Also verified: 42/42 attention golden vectors (recorded from flashinfer on an L40S)
and 7/7 serving top-1 agreement.

## Performance

The reference backend was correct, but its throughput barely improved with batch size,
because it loops over sequences in Python and syncs to the host once per layer. AITER
fixed the scaling, not just the constant factor (Qwen3-4B, 64 generated tokens per
request, `tools/amd_port/bench_throughput.py`):

| concurrency | L40S + flashinfer | MI350X reference | MI350X + AITER |
|---|---|---|---|
| 1 | 60.3 tok/s | 13.3 | 40.6 |
| 8 | 425.6 | 29.5 | 314.8 |
| 32 | 1601.5 | 32.5 | 1054.5 |

Don't quote a single speedup ratio: it is dominated by how badly the reference path
happens to do in a given round, which was not stable between rounds. The stable claim
is that AITER scales with batch size and the reference path does not.

## Three traps

1. **AITER's causal paged prefill is silently wrong on gfx950 at head_dim 128.** The
   deviation is 0.17 to 0.35 against the fp32 oracle, where correct bf16 rounding
   is ~8e-3, and it is intermittent (about two runs in three in a fresh process).
   head_dim 64/256 and all non-causal calls (decode, hydragen) were clean. Causal
   prefill therefore runs on the reference path; `TOKA_ROCM_AITER_CAUSAL=1`
   re-enables AITER for re-testing once AMD ships a fix
   (`python -m tools.amd_port.aiter_causal_bug` reproduces it).
2. **The LSE base differs between implementations.** flashinfer uses base 2, while
   aten's flash attention and AITER return natural log. A natural-log LSE fed into
   `cascade.merge_state` doesn't crash; it silently skews hydragen's merge weights.
   The backend converts at the kernel boundary.
3. **AITER's V prefetch reads past the end of the page-index array.** The index
   array is padded by 128 entries, as AITER's own tests do; without the padding you
   get rare, input-dependent illegal memory accesses.

## Running it

```bash
pip install -r requirements-rocm.txt
pip install -e . --no-deps
# optional but needed for speed: build AITER from https://github.com/ROCm/aiter

python -c "import tokasaurus.model.attn_backend as b; print(b.BACKEND)"  # rocm-aiter
toka model=Qwen/Qwen3-4B-Instruct-2507 kv_cache_num_tokens=32768 max_seqs_per_forward=32
```

Harnesses (run from the repo root):

| script | purpose |
|---|---|
| `tools/amd_port/golden_spec.py` | 45 attention cases built byte-identically on any device from a CPU seed, plus the dense fp32 oracle |
| `tools/amd_port/dump_golden.py` | record flashinfer ground truth on a CUDA machine |
| `tools/amd_port/check_backend.py` | compare whichever backend is active against those recordings |
| `tools/amd_port/probe_aiter.py` | call AITER directly against the oracle and measure its LSE convention |
| `tools/amd_port/aiter_causal_bug.py` | boundary probe for trap 1 (one subprocess per config, repeated) |
| `tools/amd_port/platform_floor.py` | the stock-HF platform floor |
| `tools/amd_port/serve_e2e.sh`, `compare_e2e.py` | end-to-end greedy generations + top-20 logprobs, diffed across platforms |
| `tools/amd_port/bench_throughput.py` | the concurrency sweep above |

`tests/test_attn_backend.py` runs the ROCm wrappers on a CPU against the oracle.
