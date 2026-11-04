"""Repeated-sampling GSM8K throughput benchmark over the completions API.

Few-shot GSM8K problems are each sampled `n` times; the script reports pass@k and
end-to-end generation throughput. It talks to any OpenAI-compatible server, so the same
run compares engines: pass the server command as `launch=` (a `toka ...` or
`vllm serve ...` command line) and the script starts it, waits for it, and tears it down.
Completions rather than chat, so it also works for base models such as OLMo.

    python tokasaurus/benchmarks/gsm8k_passk.py model=Qwen/Qwen2.5-Math-1.5B-Instruct \
        limit=4096 n=32 launch="toka model=Qwen/Qwen2.5-Math-1.5B-Instruct" \
        save_path=results/toka.jsonl
"""

import json
import random
import re
import statistics
import time
from functools import partial

import pydra
from datasets import load_dataset
from openai import OpenAI
from tabulate import tabulate
from tqdm import tqdm

from tokasaurus.benchmarks.utils import (
    BaseConfig,
    launch_server,
    make_pass_at_k_table,
    maybe_save_results,
    parallelize,
    shuffle_and_limit,
)
from tokasaurus.model.loading import get_inner_tokenizer, load_tokenizer

DEFAULT_WORKERS = 20
# problem i is submitted at i * this many seconds (as in upstream's monkeys_gsm8k.py)
REQUEST_STAGGER_SECONDS = 0.1
# save every Nth problem's completions for spot checks
COMPLETION_SAMPLE_EVERY = 300


class ScriptConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.n = 512
        self.limit = 4096
        self.max_tokens = 1024  # OLMo/DataDecide models have a 2048-token context
        self.temperature = 0.6
        self.num_few_shot = 2
        self.stop_strings = ["Question:"]
        self.workers = 1024

    def finalize(self):
        self.ks = list(range(1, min(11, self.n + 1)))
        cur = 100
        while True:
            self.ks.append(cur)
            cur *= 10
            if cur > self.n:
                break

        if self.n not in self.ks:
            self.ks.append(self.n)


ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
INVALID_ANS_GSM8k = "[invalid]"
GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]


def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st


def extract_answer_gsm8k(completion):
    match = ANS_RE_GSM8k.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = filter_ignores(
            match_str,
            GSM8K_IGNORE_REGEXES,
        )
        return match_str
    else:
        return INVALID_ANS_GSM8k


def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    return extract_answer_gsm8k(model_completion) == gt_answer


def get_few_shot_prompt(item):
    few_shot_items = item["few_shot_items"]

    few_shot_pieces = []
    for f in few_shot_items:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/gsm8k/gsm8k.yaml
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)

    few_shot_prompt = "".join(few_shot_pieces)

    return few_shot_prompt


def run_inference(
    item,
    port: int,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_strings: list,
    n: int,
    workers: int,
):
    # making the ordering of requests to the server more consistent with multiple workers
    if workers != 0:
        index = item["shuffled_index"]
        time.sleep(REQUEST_STAGGER_SECONDS * index)

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key=api_key,
        max_retries=0,
        timeout=600.0,
    )
    few_shot_prompt = get_few_shot_prompt(item)
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:"

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_strings,
            n=n,
            logprobs=None,
        )
    except Exception as e:
        # openai exceptions don't pickle across the worker pool
        raise RuntimeError(f"api call failed: {type(e).__name__}: {e!s}") from e

    completions = [choice.text for choice in response.choices]
    assert len(completions) == n

    gt_answer = item["answer"]
    corrects = [is_correct_gsm8k(completion, gt_answer) for completion in completions]

    result = {
        "prompt": prompt,
        "completions": completions,
        "corrects": corrects,
    }

    return result


def run_eval(config: ScriptConfig, go_func, test_dataset: list[dict]):
    start = time.time()
    num_workers = config.workers if config.workers is not None else DEFAULT_WORKERS
    print(f"Using {num_workers} workers for {len(test_dataset)} items")
    results_list = parallelize(
        go_func,
        test_dataset,
        num_workers=num_workers,
        processes=True,
        allow_unordered=True,
    )
    end = time.time()

    elapsed = end - start
    print(f"Elapsed time: {elapsed} seconds")

    corrects_list = [result["corrects"] for result in results_list]
    table = make_pass_at_k_table(corrects_list, config.ks)

    print(tabulate(table, headers=["k", "pass@k"], tablefmt="github"))

    tokenizer = load_tokenizer(config.model)
    inner_tokenizer = get_inner_tokenizer(tokenizer)

    total_input_tokens = sum(
        len(inner_tokenizer.encode(result["prompt"]).ids) for result in results_list
    )
    encoded_outputs = [
        inner_tokenizer.encode_batch(result["completions"]) for result in results_list
    ]
    output_tokens_per_item = [
        sum(len(output) for output in outputs) for outputs in encoded_outputs
    ]
    total_output_tokens = sum(output_tokens_per_item)

    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    throughput = total_output_tokens / elapsed
    print(f"Throughput: {throughput:.2f} tokens/second")

    maybe_save_results(
        config,
        {
            "duration": elapsed,
            "pass_at_k": table,
            "launch": config.launch,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    )

    if (save_path := config.save_path) is not None:
        completions_save_path = save_path.with_suffix(".completions.jsonl")
        num_logged = 0
        with open(completions_save_path, "a") as f:
            for idx, result in enumerate(results_list):
                if idx % COMPLETION_SAMPLE_EVERY == 0:
                    line = json.dumps(
                        {
                            "prompt": result["prompt"],
                            "completions": result["completions"],
                            "corrects": result["corrects"],
                        }
                    )
                    f.write(line + "\n")
                    num_logged += 1
        print(f"Logged {num_logged} completions to {completions_save_path}")


def main(config: ScriptConfig):
    # Problems are drawn from the train split (7.5K items) so that limit=4096 is
    # possible; few-shot examples are sampled from the other selected problems.
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))

    random.seed(config.seed)
    train_dataset = shuffle_and_limit(train_dataset, config)
    print(f"Number of train items: {len(train_dataset)}")

    for i, data in enumerate(train_dataset):
        other_items = [item for j, item in enumerate(train_dataset) if j != i]
        sampled_items = random.sample(
            other_items, min(config.num_few_shot, len(other_items))
        )
        # only the needed fields: the items reference each other otherwise
        few_shot_items = [
            {"question": item["question"], "answer": item["answer"]}
            for item in sampled_items
        ]
        data["few_shot_items"] = few_shot_items

    inner_tokenizer = get_inner_tokenizer(load_tokenizer(config.model))
    prompt_lengths = []
    for data in train_dataset:
        prompt = get_few_shot_prompt(data) + f"Question: {data['question']}\nAnswer:"
        prompt_lengths.append(len(inner_tokenizer.encode(prompt).ids))

    prompt_lengths.sort()
    stats = {
        "min": prompt_lengths[0],
        "max": prompt_lengths[-1],
        "mean": statistics.mean(prompt_lengths),
        "median": statistics.median(prompt_lengths),
    }
    print("prompt length stats (tokens):")
    print(
        f"  Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}"
    )

    # plain values rather than the pydra config, which doesn't pickle
    go_func = partial(
        run_inference,
        port=config.port,
        api_key=config.api_key,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop_strings=config.stop_strings,
        n=config.n,
        workers=config.workers if config.workers is not None else 0,
    )

    if (save_path := config.save_path) is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        launch_command_save_path = save_path.with_suffix(".launch.txt")
        launch_command_save_path.write_text(str(config.launch))

    with launch_server(config):
        for _ in tqdm(range(config.reps)):
            run_eval(config, go_func, train_dataset)


if __name__ == "__main__":
    pydra.run(main)
