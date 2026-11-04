"""Decode throughput against a running tokasaurus server, swept over concurrency.

The number that matters is the scaling (c=32 vs c=1): the reference backend loops over
sequences, so it flatlines, while a batched kernel should scale until it saturates.
Prompts are distinct per slot so nothing shares a prefix.

    python -m tools.amd_port.bench_throughput --port 8231 --concurrency 1,8,32
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _post(port: int, body: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _prompt(i: int, prompt_tokens: int) -> str:
    # Unique filler per slot so no two requests share a prefix (hydragen would
    # otherwise turn a concurrency sweep into a shared-prefix benchmark).
    filler = " ".join(f"w{i}x{j}" for j in range(prompt_tokens))
    return f"Item {i}. Context: {filler}\nContinue the story at length:"


def run_one(port: int, model: str, i: int, max_tokens: int, prompt_tokens: int) -> int:
    d = _post(
        port,
        {
            "model": model,
            "prompt": _prompt(i, prompt_tokens),
            "max_tokens": max_tokens,
            # Sampling (not greedy) so generation does not collapse into a
            # degenerate repeat, and ignore_eos so every request spends exactly
            # the same decode budget regardless of what it generates.
            "temperature": 1.0,
            "ignore_eos": True,
        },
    )
    usage = d.get("usage") or {}
    n = usage.get("completion_tokens")
    if n is None:
        # Fall back to a token-count estimate if the server omits usage.
        n = max_tokens
    return int(n)


def bench(
    port: int, model: str, concurrency: int, max_tokens: int, prompt_tokens: int
) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        counts = list(
            ex.map(
                lambda i: run_one(port, model, i, max_tokens, prompt_tokens),
                range(concurrency),
            )
        )
    dt = time.perf_counter() - t0
    total = sum(counts)
    return {
        "concurrency": concurrency,
        "seconds": round(dt, 2),
        "total_tokens": total,
        "tok_per_s": round(total / dt, 1),
        "tokens_per_request": counts[0] if len(set(counts)) == 1 else counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8231)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=200)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]

    if args.warmup:
        # First request pays torch.compile / kernel JIT; never time it.
        run_one(args.port, args.model, 999, 8, args.prompt_tokens)
        print("[warmup done]", flush=True)

    rows = []
    for c in levels:
        r = bench(args.port, args.model, c, args.max_tokens, args.prompt_tokens)
        rows.append(r)
        print(
            f"concurrency={r['concurrency']:>3}  "
            f"{r['tok_per_s']:>8.1f} tok/s  "
            f"({r['total_tokens']} tok in {r['seconds']}s)",
            flush=True,
        )

    base = rows[0]["tok_per_s"]
    for r in rows:
        r["scaling_vs_c1"] = round(r["tok_per_s"] / base, 2)
    print(
        "\nscaling vs concurrency=1: "
        + ", ".join(f"{r['concurrency']}x -> {r['scaling_vs_c1']}x" for r in rows)
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "rows": rows}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
