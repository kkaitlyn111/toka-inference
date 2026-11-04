"""Shared spec for NVIDIA-vs-ROCm attention parity testing.

The whole point of this module is that BOTH platforms build byte-identical inputs
from a seed, so the golden files only need to carry OUTPUTS. Inputs are generated
on the CPU with an explicit torch.Generator and only then moved to the device --
CUDA and ROCm device RNGs do not agree, CPU RNG does.

Shapes mirror real model configs (read from their config.json files):

  qwen3-32b     64 layers, 64 qo / 8 kv heads, head_dim 128, full attention
  llama-3.1-8b  32 layers, 32 qo / 8 kv heads, head_dim 128
  gemma-4-31b   60 layers, 32 qo / 16 kv heads, head_dim 256 local / 512 global
  gpt-oss-20b   24 layers, 64 qo / 8 kv heads, head_dim 64

KV cache layout is flashinfer NHD: [num_pages, page_size, num_kv_heads, head_dim].
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

PAGE_SIZE = 16  # tokasaurus ServerConfig.page_size default


@dataclass(frozen=True)
class AttnCase:
    """One attention parity case. `name` is the golden-file key."""

    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    seq_lens: tuple[int, ...]  # KV length per sequence
    qo_lens: tuple[int, ...]  # query tokens per sequence (1 each => decode)
    causal: bool
    dtype: str = "bfloat16"
    page_size: int = PAGE_SIZE
    seed: int = 0

    @property
    def is_decode(self) -> bool:
        return all(q == 1 for q in self.qo_lens)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def to_json(self) -> dict:
        return asdict(self)


def _pages_for(seq_len: int, page_size: int) -> int:
    return max(1, math.ceil(seq_len / page_size))


def build_inputs(case: AttnCase, device: str = "cuda") -> dict:
    """Deterministically build the inputs for a case.

    Returns q, k_cache, v_cache and the paged-KV index structures that
    flashinfer's plan() consumes. Identical on any platform for a given case.
    """
    g = torch.Generator(device="cpu").manual_seed(case.seed)
    dt = case.torch_dtype
    ps = case.page_size

    # --- page table -------------------------------------------------------
    # Pages are handed out contiguously per sequence but the *page ids* are
    # shuffled, so a backend that assumes contiguous physical pages fails here.
    pages_per_seq = [_pages_for(n, ps) for n in case.seq_lens]
    total_pages = sum(pages_per_seq)
    # +37 slack pages so page ids are not a dense 0..N-1 range either.
    num_pages = total_pages + 37
    perm = torch.randperm(num_pages, generator=g)[:total_pages]

    kv_indptr = torch.zeros(len(case.seq_lens) + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.tensor(pages_per_seq, dtype=torch.int32).cumsum(0)
    kv_indices = perm.to(torch.int32)
    last_page_len = torch.tensor(
        [((n - 1) % ps) + 1 for n in case.seq_lens], dtype=torch.int32
    )

    qo_indptr = torch.zeros(len(case.qo_lens) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.tensor(case.qo_lens, dtype=torch.int32).cumsum(0)

    # --- tensors ----------------------------------------------------------
    # Scaled to a realistic activation magnitude: unit-normal q.k over head_dim
    # would saturate softmax and hide numeric differences.
    scale = 0.5
    num_q_tokens = int(qo_indptr[-1])
    q = (
        torch.randn(num_q_tokens, case.num_qo_heads, case.head_dim, generator=g) * scale
    ).to(dt)
    k_cache = (
        torch.randn(num_pages, ps, case.num_kv_heads, case.head_dim, generator=g)
        * scale
    ).to(dt)
    v_cache = (
        torch.randn(num_pages, ps, case.num_kv_heads, case.head_dim, generator=g)
        * scale
    ).to(dt)

    # Slots past a sequence's true length are garbage in a real server too;
    # fill them with a loud value so any backend that fails to mask shows up.
    for s in range(len(case.seq_lens)):
        pages = kv_indices[kv_indptr[s] : kv_indptr[s + 1]]
        tail = pages[-1].item()
        k_cache[tail, last_page_len[s] :] = 100.0
        v_cache[tail, last_page_len[s] :] = 100.0

    return {
        "q": q.to(device),
        "k_cache": k_cache.to(device),
        "v_cache": v_cache.to(device),
        "qo_indptr": qo_indptr.to(device),
        "kv_indptr": kv_indptr.to(device),
        "kv_indices": kv_indices.to(device),
        "last_page_len": last_page_len.to(device),
    }


def reference_attention(case: AttnCase, inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense fp32 reference: the definition of correct, on any device.

    Mirrors the semantics of `_dense_route_probs` in tokasaurus'
    attention_utils.py. Returns (out [T, Hq, D] in case dtype,
    lse [T, Hq] fp32 in BASE 2 -- flashinfer's convention).
    """
    ps = case.page_size
    q = inp["q"].float()
    k_cache = inp["k_cache"].float()
    v_cache = inp["v_cache"].float()
    # Bounds are read as Python ints (CPU), but the page gather must use an index
    # tensor on the same device as the cache.
    qo_indptr = inp["qo_indptr"].cpu()
    kv_indptr = inp["kv_indptr"].cpu()
    kv_indices = inp["kv_indices"].to(k_cache.device)
    last_page_len = inp["last_page_len"].cpu()

    hq, hkv, d = case.num_qo_heads, case.num_kv_heads, case.head_dim
    group = hq // hkv
    sm_scale = 1.0 / math.sqrt(d)
    log2e = 1.4426950408889634

    out = torch.zeros(q.shape[0], hq, d, device=q.device, dtype=torch.float32)
    lse = torch.full(
        (q.shape[0], hq), float("-inf"), device=q.device, dtype=torch.float32
    )

    for s in range(len(case.seq_lens)):
        pages = kv_indices[kv_indptr[s] : kv_indptr[s + 1]]
        n_keys = (len(pages) - 1) * ps + int(last_page_len[s])
        k = k_cache[pages].reshape(-1, hkv, d)[:n_keys]  # [L, Hkv, D]
        v = v_cache[pages].reshape(-1, hkv, d)[:n_keys]

        q0, q1 = int(qo_indptr[s]), int(qo_indptr[s + 1])
        qs = q[q0:q1].view(-1, hkv, group, d)  # [Q, Hkv, G, D]
        scores = torch.einsum("qhgd,lhd->qhgl", qs, k) * sm_scale

        if case.causal:
            # Queries are the LAST (q1-q0) positions of the sequence.
            n_q = q1 - q0
            qpos = torch.arange(n_keys - n_q, n_keys, device=q.device).view(-1, 1)
            kpos = torch.arange(n_keys, device=q.device).view(1, -1)
            mask = kpos > qpos  # [Q, L]
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))

        row_lse = torch.logsumexp(scores, dim=-1)  # [Q, Hkv, G]
        probs = torch.exp(scores - row_lse.unsqueeze(-1))
        o = torch.einsum("qhgl,lhd->qhgd", probs, v)

        out[q0:q1] = o.reshape(-1, hq, d)
        lse[q0:q1] = (row_lse * log2e).reshape(-1, hq)

    return out.to(case.torch_dtype), lse


# ---------------------------------------------------------------------------
# The case matrix.
# ---------------------------------------------------------------------------


def _decode(name, hq, hkv, d, seq_lens, seed=0):
    return AttnCase(
        name=name,
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_dim=d,
        seq_lens=tuple(seq_lens),
        qo_lens=tuple(1 for _ in seq_lens),
        causal=False,  # decode: single query attends to all of KV
        seed=seed,
    )


def _prefill(name, hq, hkv, d, qo_lens, seed=0, causal=True):
    # Prefill from empty: kv_len == qo_len per sequence.
    return AttnCase(
        name=name,
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_dim=d,
        seq_lens=tuple(qo_lens),
        qo_lens=tuple(qo_lens),
        causal=causal,
        seed=seed,
    )


# (num_qo_heads, num_kv_heads, head_dim, label)
MODEL_SHAPES = [
    (64, 8, 128, "qwen3_32b"),
    (32, 8, 128, "llama31_8b"),
    (32, 16, 256, "gemma4_local"),
    (32, 16, 512, "gemma4_global"),
    (64, 8, 64, "gptoss"),
]

CASES: list[AttnCase] = []

for hq, hkv, d, label in MODEL_SHAPES:
    # decode: single seq, batch, ragged lengths, page-aligned vs not, long
    CASES += [
        _decode(f"decode_{label}_bs1_short", hq, hkv, d, [7]),
        _decode(f"decode_{label}_bs1_aligned", hq, hkv, d, [64]),
        _decode(f"decode_{label}_bs4_ragged", hq, hkv, d, [7, 64, 129, 1]),
        _decode(
            f"decode_{label}_bs17_mixed", hq, hkv, d, [1 + 37 * i for i in range(17)]
        ),
        _decode(f"decode_{label}_bs1_long", hq, hkv, d, [4096]),
    ]
    # prefill: causal (true prefill) and non-causal (hydragen shared prefix)
    CASES += [
        _prefill(f"prefill_{label}_bs1_short", hq, hkv, d, [7]),
        _prefill(f"prefill_{label}_bs3_ragged", hq, hkv, d, [7, 64, 129]),
        _prefill(f"prefill_{label}_bs1_long", hq, hkv, d, [2048]),
        _prefill(f"hydragen_{label}_bs3", hq, hkv, d, [7, 64, 129], causal=False),
    ]

CASES_BY_NAME = {c.name: c for c in CASES}
