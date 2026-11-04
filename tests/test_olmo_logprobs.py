"""End-to-end OLMo check: tokasaurus' native OLMo path against the HF reference model.

Serves an hf_olmo checkpoint with tokasaurus, greedily generates from GSM8K questions,
then re-scores prompt + generation with ai2-olmo's HF model and compares, per token:

  * argmax agreement (TOKEN_MATCH_LIMIT),
  * max absolute logprob difference (MAX_ABS_TOL_LIMIT),
  * mean relative logprob difference (MEAN_REL_TOL_LIMIT).

Needs a GPU and the [olmo] extra:

    MODEL=allenai/DataDecide-dolma1_6plus-1B pytest tests/test_olmo_logprobs.py -s
"""

import ast
import os
import shlex

import pydra
import pytest
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from openai import OpenAI

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port

MODEL = os.environ.get("MODEL", "allenai/DataDecide-dolma1_6plus-1B")
OVERRIDES = os.environ.get("OVERRIDES", "dtype=float16")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "5"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "250"))
MEAN_REL_TOL_LIMIT = ast.literal_eval(os.environ.get("MEAN_REL_TOL_LIMIT", "0.1"))
MAX_ABS_TOL_LIMIT = ast.literal_eval(os.environ.get("MAX_ABS_TOL_LIMIT", "0.2"))
TOKEN_MATCH_LIMIT = ast.literal_eval(os.environ.get("TOKEN_MATCH_LIMIT", "0.95"))


def make_config():
    config = ServerConfig()
    config.model = MODEL
    # DataDecide models have a 2048-token context; keep the engine small.
    config.kv_cache_num_tokens = 1024
    config.max_num_tokens_per_request = 2048
    config.max_seqs_per_forward = 8
    config.max_tokens_per_forward = 512
    config.use_cudagraphs = False
    config.early_stopping_buffer_size = 256
    config.early_stopping_num_prediction_buckets = 128
    config.port = find_free_port()
    config.enable_chosen_logprobs = True
    config.max_topk_logprobs = 1

    if OVERRIDES:
        pydra.apply_overrides(config, shlex.split(OVERRIDES))

    return config


@pytest.fixture(scope="module")
def client():
    mp.set_start_method("spawn", force=True)
    config = make_config()
    with server_manager(config):
        yield OpenAI(api_key="beepboop", base_url=f"http://localhost:{config.port}/v1")


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    import hf_olmo  # noqa: F401  (registers the hf_olmo architecture with transformers)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    return model, tokenizer


PROMPTS = load_dataset("openai/gsm8k", "main", split="train")["question"][:NUM_PROMPTS]


@pytest.mark.parametrize("prompt_idx", range(len(PROMPTS)))
def test_logprobs_match_hf(client: OpenAI, hf_model_and_tokenizer, prompt_idx: int):
    prompt = PROMPTS[prompt_idx]
    response = client.completions.create(
        model="none",
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        logprobs=1,
    )
    model, tokenizer = hf_model_and_tokenizer

    for choice in response.choices:
        api_logprobs = choice.logprobs.token_logprobs
        seq_ids = tokenizer.convert_tokens_to_ids(choice.logprobs.tokens)

        input_ids = tokenizer.encode(prompt) + seq_ids
        with torch.inference_mode():
            logits = model(torch.tensor([input_ids], device="cuda:0")).logits

        # position t predicts token t+1, so the generated tokens are scored by the
        # len(seq_ids) positions ending one before the last
        hf_logprobs = torch.log_softmax(logits.float(), dim=-1)[
            0, -len(seq_ids) - 1 : -1
        ]

        token_matches, adiffs, rdiffs = [], [], []
        for token_id, hf_dist, api_logprob in zip(
            seq_ids, hf_logprobs, api_logprobs, strict=True
        ):
            hf_logprob = hf_dist[token_id].item()
            token_matches.append(hf_dist.argmax().item() == token_id)
            adiff = abs(api_logprob - hf_logprob)
            adiffs.append(adiff)
            rdiffs.append(2 * adiff / (abs(api_logprob) + abs(hf_logprob) + 1e-3))

        token_match_rate = sum(token_matches) / len(token_matches)
        max_adiff = max(adiffs)
        mean_rdiff = sum(rdiffs) / len(rdiffs)
        print(
            f"[prompt {prompt_idx}] {len(seq_ids)} tokens | token match {token_match_rate:.4f} "
            f"| max abs logprob diff {max_adiff:.4f} | mean rel diff {mean_rdiff:.4f}"
        )

        if TOKEN_MATCH_LIMIT is not None:
            assert token_match_rate >= TOKEN_MATCH_LIMIT
        if MEAN_REL_TOL_LIMIT is not None:
            assert mean_rdiff <= MEAN_REL_TOL_LIMIT
        if MAX_ABS_TOL_LIMIT is not None:
            assert max_adiff <= MAX_ABS_TOL_LIMIT
