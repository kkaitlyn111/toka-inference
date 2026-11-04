#!/bin/bash
# Repeated-sampling GSM8K: tokasaurus vs vLLM on the same model, problems and sampling
# settings. Each engine is launched, benchmarked and torn down by gsm8k_passk.py.
#
#   MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct ENGINES="toka vllm" bash benchmarking/gsm8k_toka_vs_vllm.sh
#
# Every setting below can be overridden from the environment. vLLM is usually
# installed in its own environment: point VLLM_BIN at that env's `vllm`.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}
ENGINES=${ENGINES:-"toka vllm"}       # any of: toka toka_hydragen vllm
N=${N:-32}                            # samples per problem
LIMIT=${LIMIT:-4096}                  # number of problems
PORT=${PORT:-10210}
OUT_DIR=${OUT_DIR:-benchmarking/results/$(basename "$MODEL")}
VLLM_BIN=${VLLM_BIN:-vllm}

# engine sizing: the settings behind the A100 results in benchmarking/RESULTS.md
KV_CACHE_TOKENS=${KV_CACHE_TOKENS:-1048576}
MAX_SEQS=${MAX_SEQS:-8192}
MAX_TOKENS_PER_FORWARD=${MAX_TOKENS_PER_FORWARD:-16384}
TORCH_COMPILE=${TORCH_COMPILE:-F}
VLLM_GPU_BLOCKS=${VLLM_GPU_BLOCKS:-98304}
VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-4096}

export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"

TOKA="toka model=$MODEL port=$PORT kv_cache_num_tokens=$KV_CACHE_TOKENS \
max_seqs_per_forward=$MAX_SEQS max_tokens_per_forward=$MAX_TOKENS_PER_FORWARD \
torch_compile=$TORCH_COMPILE .uvsh"

for engine in $ENGINES; do
    case $engine in
        toka) launch="$TOKA" ;;
        toka_hydragen) launch="$TOKA use_hydragen=T hydragen_min_group_size=129" ;;
        vllm) launch="$VLLM_BIN serve $MODEL --port $PORT \
--num-gpu-blocks-override $VLLM_GPU_BLOCKS --max-num-seqs $VLLM_MAX_SEQS \
--max-num-batched-tokens $MAX_TOKENS_PER_FORWARD --enable-prefix-caching \
--enable-chunked-prefill --disable-log-requests --uvicorn-log-level warning" ;;
        *) echo "unknown engine: $engine" >&2; exit 1 ;;
    esac

    echo "== $engine: $launch"
    python tokasaurus/benchmarks/gsm8k_passk.py model="$MODEL" limit="$LIMIT" n="$N" \
        port="$PORT" reps=1 launch="$launch" save_path="$OUT_DIR/$engine.jsonl"
done
