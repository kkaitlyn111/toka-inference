# toka-inference

This repo extends [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus), a
high-throughput LLM inference engine, in a few practical uses. It adds compatibility with 1. AMD (ROCm attention
backend built on AITER kernels), 2. hybrid-attention architectures that mix full attention
with Gated DeltaNet (GDN) linear attention or sliding-window attention (SWA), and 3. native
support for the GLM, Qwen, Gemma, OLMo, Muse, gpt-oss etc. model families. Across these families, it supports inference with Cartridges (trained KV cache modules) and Attention Matching, along with capturing of GDN state.

## Models

| model | architecture |
|---|---|
| Llama 3, Qwen 2, Qwen 3 | dense |
| OLMo (`hf_olmo` checkpoints, incl. DataDecide) | dense |
| GLM-4.6 | MoE, FP8 weights |
| Qwen 3.5 | GDN + full attention |
| Gemma 4 | SWA + full attention |
| Muse Glimmer | SWA + NoPE full attention |
| gpt-oss | MoE, MXFP4 weights, attention sinks, SWA |

## Benchmark

On repeated sampling for GSM8K (Qwen2.5-Math-1.5B-Instruct, 4096 problems × 32 samples,
one A100), Tokasaurus finishes in 729 s and vLLM in 2206 s, at the same pass@k. Details
are in [`benchmarking/RESULTS.md`](benchmarking/RESULTS.md).

## Installation

```bash
pip install -e .
pip install -e ".[olmo]"   # for hf_olmo checkpoints
```

## Quickstart

```bash
toka model=Qwen/Qwen3-4B-Instruct-2507
toka-ping prompt='What is 12 * 7?' max_tokens=32 chat=True
```

The server exposes OpenAI-compatible chat and completions endpoints; see
[`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md) for the full set of options.

On AMD GPUs, install with `pip install -r requirements-rocm.txt && pip install -e . --no-deps`
and add [AITER](https://github.com/ROCm/aiter); [`docs/ROCM.md`](docs/ROCM.md) describes the
backend.
