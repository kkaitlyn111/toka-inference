# toka-inference

An extension of [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus), the
high-throughput LLM inference engine, that adds:

- an **AMD ROCm backend** (AITER kernels on MI350X, flashinfer unchanged on NVIDIA);
- **hybrid-attention architectures**: sliding-window (SWA) models now, gated-DeltaNet (GDN)
  hybrids coming soon;
- more **model families**: GLM, Qwen, Gemma, OLMo and Muse.

It also includes an engine benchmark harness.

## Supported models

| model | architecture | status | tested |
|---|---|---|---|
| Llama 3, Qwen 2, Qwen 3 | dense | supported (upstream) | upstream tests; ROCm backend on Qwen3-4B |
| OLMo / DataDecide (`hf_olmo`) | dense | supported | logprobs vs HF transformers on GPU |
| GLM-4.6 (355B) | MoE, FP8 | supported | parity vs HF on CPU; GPU parity pending |
| Gemma 4 31B | sliding-window + full attention | supported | parity vs HF on CPU; GPU parity pending |
| Muse Glimmer 30B | sliding-window + NoPE full attention | supported | parity vs HF on CPU; GPU parity pending |
| Qwen 3.5 9B and other gated-DeltaNet hybrids | linear-attention hybrid | coming soon | |
| gpt-oss 20B | MoE, attention sinks | coming soon | |

## What's added

- **Models:** native OLMo (`model/olmo.py`), GLM-4.6 with grouped-GEMM MoE dispatch and
  FP8 dequant at load (`model/glm4_moe.py`), and Gemma 4 / Muse Glimmer on a shared
  per-layer-type attention base (`model/sliding_window.py`). All of them work with the
  transformers version upstream pins.
- **AMD ROCm backend** (`model/attn_backend/`): every flashinfer call goes through one
  module. On CUDA it's flashinfer; on ROCm it's AMD's AITER kernels, with a portable
  reference implementation as the fallback and test oracle. Details and numbers in
  [`docs/ROCM.md`](docs/ROCM.md).
- **Benchmarks** (`benchmarks/gsm8k_passk.py`, `benchmarking/`): repeated-sampling
  GSM8K against any OpenAI-compatible server. On one A100, 4096 problems × 32 samples
  with Qwen2.5-Math-1.5B took 729 s on Tokasaurus vs 2206 s on vLLM, at the same pass@k
  ([`benchmarking/RESULTS.md`](benchmarking/RESULTS.md)).
- **Tests:** OLMo logprobs vs HF, model parity on tiny checkpoints, and CPU tests for the
  ROCm wrappers.

## Quickstart

```bash
pip install -e ".[dev,olmo]"

# serve a model
toka model=allenai/DataDecide-dolma1_6plus-1B
toka-ping prompt='Question: What is 12 * 7?\nAnswer:' max_tokens=32

# tests: CPU ones run anywhere, the rest need a GPU
pytest tests/test_attn_backend.py tests/test_olmo_config.py tests/test_model_parity.py
MODEL=allenai/DataDecide-dolma1_6plus-1B pytest tests/test_olmo_logprobs.py

# benchmark against vLLM
MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct VLLM_BIN=/path/to/vllm bash benchmarking/gsm8k_toka_vs_vllm.sh
```

On AMD: `pip install -r requirements-rocm.txt && pip install -e . --no-deps`, plus
[AITER](https://github.com/ROCm/aiter) for the fast path. See [`docs/ROCM.md`](docs/ROCM.md).

Upstream's full CLI docs are in [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md).
