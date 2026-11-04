"""Paged attention on AMD AITER (MI350X / gfx950): the whole ragged batch in one kernel.

Uses `aiter.mha_batch_prefill_func`, which takes flashinfer's page table almost as is
(`flash_attn_varlen_func` needs page sizes that are multiples of 128; tokasaurus uses 16).
Decode runs through the same kernel with one query per sequence, so it also returns the
LSE that hydragen's merge needs. Two differences from flashinfer are handled here:

  1. the LSE comes back head-major and in natural log; flashinfer is [tokens, heads]
     and base 2;
  2. the kernel's V prefetch reads past the end of the page-index array, so it is
     padded (as AITER's own tests do), or rare illegal memory accesses follow.
"""

from __future__ import annotations

import torch
from torch import Tensor

LOG2_E = 1.4426950408889634

# padding AITER's tests use for the V prefetch (op_tests/test_batch_prefill.py)
PAGE_INDEX_PAD = 128


def _index(t: Tensor, device: torch.device | None = None) -> Tensor:
    """Contiguous int32. Page tables may arrive on the CPU and AITER needs them on the
    GPU, but plan() never sees a device tensor, so the move is deferred to
    `AiterPagedPlan.on()` on the first layer."""
    out = t.to(dtype=torch.int32).contiguous()
    return out if device is None else out.to(device=device)


class AiterPagedPlan:
    """A flashinfer page table digested for AITER once per forward step, so its two host
    syncs (max query / KV length) are amortized over all layers."""

    __slots__ = (
        "_device",
        "causal",
        "cu_seqlens_q",
        "kv_indptr",
        "kv_last_page_lens",
        "kv_page_indices",
        "max_seqlen_k",
        "max_seqlen_q",
        "num_q_tokens",
        "num_qo_heads",
        "sm_scale",
    )

    def __init__(
        self,
        kv_indptr: Tensor,
        kv_indices: Tensor,
        last_page_len: Tensor,
        page_size: int,
        num_qo_heads: int,
        qo_indptr: Tensor | None = None,
        causal: bool = False,
        sm_scale: float | None = None,
    ):
        device = kv_indices.device
        self._device = device
        b = kv_indptr.numel() - 1

        self.kv_indptr = _index(kv_indptr)
        self.kv_last_page_lens = _index(last_page_len)
        self.causal = causal
        self.sm_scale = sm_scale
        self.num_qo_heads = num_qo_heads

        # Pad the page-index array (see PAGE_INDEX_PAD). The pad value is 0, a
        # valid page id, so the stray prefetch stays inside the cache buffer.
        idx = _index(kv_indices)
        self.kv_page_indices = torch.cat(
            [idx, idx.new_zeros(PAGE_INDEX_PAD)]
        ).contiguous()

        indptr64 = self.kv_indptr.to(torch.int64)
        pages_per_seq = indptr64[1:] - indptr64[:-1]
        kv_lens = (pages_per_seq - 1).clamp(
            min=0
        ) * page_size + self.kv_last_page_lens.to(torch.int64)
        kv_lens = torch.where(pages_per_seq == 0, torch.zeros_like(kv_lens), kv_lens)
        self.max_seqlen_k = int(kv_lens.max().item()) if b > 0 else 0

        if qo_indptr is None:
            # Decode: exactly one query token per sequence.
            self.cu_seqlens_q = torch.arange(b + 1, dtype=torch.int32, device=device)
            self.max_seqlen_q = 1
            self.num_q_tokens = b
        else:
            self.cu_seqlens_q = _index(qo_indptr)
            qo64 = self.cu_seqlens_q.to(torch.int64)
            self.max_seqlen_q = int((qo64[1:] - qo64[:-1]).max().item()) if b > 0 else 0
            self.num_q_tokens = int(qo64[-1].item()) if b > 0 else 0

    def on(self, device: torch.device) -> None:
        """Move the index tensors to the compute device, once.

        Called from the first layer's run(); subsequent layers hit the no-op path.
        """
        if self._device == device:
            return
        self.kv_indptr = self.kv_indptr.to(device)
        self.kv_last_page_lens = self.kv_last_page_lens.to(device)
        self.kv_page_indices = self.kv_page_indices.to(device)
        self.cu_seqlens_q = self.cu_seqlens_q.to(device)
        self._device = device


def aiter_paged_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    plan: AiterPagedPlan,
    sm_scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Run AITER paged attention for one layer.

    q: [num_q_tokens, num_qo_heads, head_dim].
    Returns (out [num_q_tokens, num_qo_heads, head_dim] in q.dtype,
             lse [num_q_tokens, num_qo_heads] fp32, BASE 2).
    """
    import aiter

    num_q_tokens, hq, _ = q.shape
    if num_q_tokens == 0 or plan.max_seqlen_k == 0:
        return (
            torch.zeros_like(q),
            torch.full(
                (num_q_tokens, hq), float("-inf"), device=q.device, dtype=torch.float32
            ),
        )

    scale = sm_scale if sm_scale is not None else plan.sm_scale
    plan.on(q.device)

    out, lse = aiter.mha_batch_prefill_func(
        q.contiguous(),
        k_cache,
        v_cache,
        plan.cu_seqlens_q,
        plan.kv_indptr,
        plan.kv_page_indices,
        plan.max_seqlen_q,
        plan.max_seqlen_k,
        softmax_scale=scale,
        causal=plan.causal,
        kv_last_page_lens=plan.kv_last_page_lens,
        return_lse=True,
    )

    # AITER: [num_heads, total_q], natural log.
    # flashinfer / tokasaurus merge_state: [total_q, num_heads], base 2.
    lse = lse.transpose(0, 1).contiguous().float() * LOG2_E
    return out, lse
