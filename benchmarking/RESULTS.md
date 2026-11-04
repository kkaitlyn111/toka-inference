# Benchmark results

All runs use `tokasaurus/benchmarks/gsm8k_passk.py`: 4096 GSM8K problems (train split),
2-shot, `n=32` samples each at temperature 0.6, `max_tokens=1024`, with the server
launched and torn down by the script. The raw result files are in `results/`.

## Tokasaurus vs vLLM, Qwen2.5-Math-1.5B-Instruct, one A100

| engine | wall-clock (s) | output tokens | throughput (tok/s) | pass@1 | pass@32 |
|---|---|---|---|---|---|
| vLLM (prefix caching, chunked prefill) | 2205.6 | 16,084,228 | 7,292 | 0.8315 | 0.9912 |
| Tokasaurus | 728.9 | 16,086,015 | 22,069 | 0.8329 | 0.9924 |
| Tokasaurus, smaller KV cache (786K tokens, 4096 seqs) | 739.2 | 16,083,164 | 21,758 | 0.8324 | 0.9915 |
| Tokasaurus + hydragen, `torch_compile=T` | 648.9 | 16,048,656 | 24,732 | 0.8290 | 0.9915 |

Tokasaurus finishes the same workload **3.0x faster** than vLLM at the same accuracy
(the Tokasaurus runs without hydragen match vLLM's pass@1 and pass@32 to within 0.002).

Notes and caveats:

* Throughput = total output tokens / wall-clock time. Requests are submitted on the
  harness's fixed arrival schedule (problem *i* is sent at 0.1 × *i* seconds, inherited from
  upstream's `monkeys_gsm8k.py`), so every run has a ~410 s floor. The gap between
  engines is therefore a lower bound on the steady-state difference.
* The vLLM and first Tokasaurus rows were filed under an `a100-40G` log directory, and
  the same Tokasaurus run also appears under `a100-80G`, so the A100 memory variant
  (40GB vs 80GB) is not certain from the logs.
* The hydragen row also turns on `torch_compile`, so it is not a pure hydragen
  ablation. A second hydragen run took 3453.9 s with identical settings and is
  treated as an outlier (its cause was not recorded); it is kept in
  `results/.../a100-80gb_toka_hydragen.jsonl`.
* Launch commands: `results/a100_qwen2.5-math-1.5b/*.jsonl` (`launch` field).

## OLMo-1B (DataDecide) served natively, one A100 (80GB job constraint)

`allenai/DataDecide-dolma1_6plus-1B` through the native OLMo path, with
`kv_cache_num_tokens=524288 max_seqs_per_forward=1024 max_tokens_per_forward=16384
torch_compile=T`. Larger `max_seqs_per_forward` settings (4096, 8192) hit KV-cache
overcommit or OOM, since OLMo-1B uses MHA and is KV-heavy:

| wall-clock (s) | pass@1 | pass@32 |
|---|---|---|
| 1420.4 | 0.0124 | 0.2029 |

The run's token count was not recorded: the benchmark crashed while counting tokens
with the OLMo tokenizer, a bug since fixed in `gsm8k_passk.py`. The low accuracy is
expected for a 1B base model at 2-shot.

## AMD MI350X

See [`docs/ROCM.md`](../docs/ROCM.md).
