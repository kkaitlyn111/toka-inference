#!/bin/bash
# End-to-end serving parity: start tokasaurus, run fixed greedy prompts, save outputs.
# Identical script on CUDA and ROCm -- only VENV and OUT differ, so a diff of the two
# JSON files is a direct platform comparison.
#
# usage: serve_e2e.sh <venv> <out.json> [model] [port]
set -uo pipefail

VENV=${1:?venv path}
OUT=${2:?output json}
MODEL=${3:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${4:-8231}

# Caches default under $SCRATCH (or /tmp); point HF_HOME at a pre-populated cache.
SCRATCH=${SCRATCH:-/tmp/$USER}
export HF_HOME=${HF_HOME:-$SCRATCH/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/triton-cache}
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$(dirname "$OUT")"

source "$VENV/bin/activate"
LOGDIR=${LOGDIR:-$SCRATCH/amd_port/logs}
mkdir -p "$LOGDIR"
SRVLOG="$LOGDIR/serve_$(basename "$OUT" .json).log"

echo "== launching $MODEL on port $PORT (venv=$VENV)"
toka model="$MODEL" port="$PORT" \
  kv_cache_num_tokens=32768 max_tokens_per_forward=4096 max_seqs_per_forward=32 max_topk_logprobs=30 \
  > "$SRVLOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/ping" >/dev/null 2>&1; then echo "server up after ${i}0s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; tail -40 "$SRVLOG"; exit 1; fi
  sleep 10
done

python - "$OUT" "$PORT" "$MODEL" <<'PY'
import json, sys, urllib.request

out_path, port, model = sys.argv[1], sys.argv[2], sys.argv[3]

# Fixed prompts spanning short/long, factual/generative. Greedy so the only source
# of divergence between platforms is the kernels.
PROMPTS = [
    "The capital of France is",
    "Q: What is 17 * 23? Think step by step.\nA:",
    "Write a Python function that reverses a linked list.\n\n```python\n",
    "List the first ten prime numbers, separated by commas:",
    "Explain in two sentences why the sky appears blue.",
    "Once upon a time in a distant galaxy, there lived",
]

results = []
for p in PROMPTS:
    body = json.dumps({
        "model": model, "prompt": p,
        "max_tokens": 64, "temperature": 0.0, "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    results.append({"prompt": p, "text": d["choices"][0]["text"]})
    print(f"[ok] {p[:40]!r} -> {d['choices'][0]['text'][:60]!r}", flush=True)

# Single-step top-k logprobs. Generated text amplifies any difference -- one
# near-tie logit flips a token and every later token diverges -- so the honest
# measure of the kernels is the one-step output distribution itself.
LONG = ("The following is a list of facts.\n"
        + "".join(f"Fact {i}: the value of item {i} is {(i * 7919) % 1000}.\n" for i in range(400))
        + "Question: what is the value of item 137?\nAnswer:")

logprobs = []
for name, p in [(f"p{i}", pr) for i, pr in enumerate(PROMPTS)] + [("long_4k", LONG)]:
    body = json.dumps({
        "model": model, "prompt": p,
        "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        lp = d["choices"][0].get("logprobs") or {}
        top = (lp.get("top_logprobs") or [{}])[0]
        logprobs.append({"name": name, "top1": d["choices"][0]["text"], "top_logprobs": top})
        print(f"[lp] {name}: top1={d['choices'][0]['text']!r} n={len(top)}", flush=True)
    except Exception as e:
        print(f"[lp] {name}: FAILED {type(e).__name__}: {e}", flush=True)
        logprobs.append({"name": name, "error": str(e)})

json.dump(
    {"model": model, "results": results, "logprobs": logprobs},
    open(out_path, "w"), indent=2,
)
print("wrote", out_path)
PY
RC=$?
kill $SRV 2>/dev/null
exit $RC
