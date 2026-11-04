# toka-inference

An extension of [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus), the
high-throughput LLM inference engine, with additional model support, an AMD ROCm backend,
and an engine benchmark harness. Built while serving models for cartridge (KV-cache
compaction) research.

## Supported models

| model | architecture | inference | cartridge training | hardware | verification |
|---|---|---|---|---|---|
| Llama 3, Qwen 2, Qwen 3 | dense | yes (upstream) | via Accordion | NVIDIA; AMD on Qwen3-4B | upstream tests; ROCm backend parity on Qwen3-4B |
| OLMo / DataDecide (`hf_olmo`) | dense | yes | no | NVIDIA | logprobs vs HF on GPU (A6000), GSM8K benchmark |
| GLM-4.6 (355B) | MoE, FP8 | yes | via Accordion | NVIDIA, AMD¹ | CPU-tested vs HF; GPU parity checks pending. Full model ran on MI350X (tp=4)¹ |
| Gemma 4 31B | sliding-window + full attention | yes | via Accordion | NVIDIA, AMD¹ | CPU-tested vs HF; GPU parity checks pending. Full-model parity on L40S/H200¹ |
| Muse Glimmer 30B | sliding-window + NoPE full attention | yes | via Accordion | NVIDIA | CPU-tested vs HF; GPU parity checks pending. Full-model parity on H200, 6 to 9223 tokens¹ |
| Qwen 3.5 9B and other gated-DeltaNet hybrids | linear-attention hybrid | coming soon | via Accordion | | |
| gpt-oss 20B | MoE, attention sinks | coming soon | via Accordion | | |

¹ Full-size results come from the original implementations in the lab's research codebase.
The versions here are ports onto upstream Tokasaurus, checked against HF transformers on
tiny random checkpoints (`tests/test_model_parity.py`). The CPU path matches HF to ~1e-6;
the GPU (flashinfer) runs of that check are pending.

Cartridge training is done in Accordion, the lab's training codebase, and is not part of
this repo.

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

## Credits

This builds on Tokasaurus by Juravsky et al. and keeps its Apache-2.0 license.

```bibtex
@misc{juravsky2025tokasaurus,
  author       = {Jordan Juravsky and Ayush Chakravarthy and Ryan Ehrlich and Sabri Eyuboglu and Bradley Brown and Joseph Shetaye and Christopher R{\'e} and Azalia Mirhoseini},
  title        = {Tokasaurus: An LLM Inference Engine for High-Throughput Workloads},
  year         = {2025},
  howpublished = {\url{https://scalingintelligence.stanford.edu/blogs/tokasaurus/}}
}
```
