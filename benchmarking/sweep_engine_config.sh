#!/bin/bash
# Sweep tokasaurus' memory/batching knobs for one model on the GSM8K pass@k workload,
# to find the largest configuration that fits (KV-cache overcommit and OOM are the
# failure modes). One run per line of CONFIGS; results go to $OUT_DIR/<name>.jsonl.
#
#   MODEL=allenai/DataDecide-dolma1_6plus-1B bash benchmarking/sweep_engine_config.sh
#
# Under SLURM, run one config per array task:  sbatch --array=0-3 ... this script
set -euo pipefail

MODEL=${MODEL:-allenai/DataDecide-dolma1_6plus-1B}
N=${N:-32}
LIMIT=${LIMIT:-4096}
TORCH_COMPILE=${TORCH_COMPILE:-T}
OUT_DIR=${OUT_DIR:-benchmarking/results/sweep_$(basename "$MODEL")}

# name kv_cache_num_tokens max_seqs_per_forward max_tokens_per_forward
# Defaults are the OLMo-1B (MHA, so KV-heavy) sweep on an A100-80GB; "baseline" is the
# largest configuration that ran without KV-cache overcommit.
CONFIGS=${CONFIGS:-"
baseline   524288 1024 16384
reduced_kv 393216  800 16384
more_seqs  524288 1536 16384
balanced   458752 1200 16384
"}

export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"

run_config() {
    local name=$1 kv=$2 seqs=$3 tokens=$4
    local port
    port=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
    local launch="toka model=$MODEL port=$port kv_cache_num_tokens=$kv \
max_seqs_per_forward=$seqs max_tokens_per_forward=$tokens torch_compile=$TORCH_COMPILE .uvsh"
    echo "== $name: $launch"
    python tokasaurus/benchmarks/gsm8k_passk.py model="$MODEL" limit="$LIMIT" n="$N" \
        port="$port" reps=1 launch="$launch" save_path="$OUT_DIR/$name.jsonl"
}

mapfile -t lines < <(echo "$CONFIGS" | sed '/^\s*$/d')
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_config ${lines[$SLURM_ARRAY_TASK_ID]}
else
    for line in "${lines[@]}"; do run_config $line; done
fi
