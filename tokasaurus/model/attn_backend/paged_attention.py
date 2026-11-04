"""Device-agnostic paged attention, used as the ROCm replacement for flashinfer.

Implements exactly the semantics tokasaurus depends on:

  * paged KV cache in flashinfer's NHD layout: [num_pages, page_size, num_kv_heads, head_dim]
  * ragged batches described by (qo_indptr, kv_indptr, kv_indices, last_page_len)
  * GQA via head grouping (num_qo_heads must be a multiple of num_kv_heads)
  * causal masking aligned to the END of the KV (queries are the last qo_len positions)
  * **log-sum-exp returned in BASE 2**, matching flashinfer. This is load-bearing:
    hydragen merges shared-prefix and per-sequence attention with
    `cascade.merge_state` (logaddexp2/exp2), and a natural-log LSE silently skews
    the merge weights.

Attention is computed with an online (flash-style) softmax over KV chunks, so peak
memory is bounded by the chunk size rather than the sequence length, and the fp32
accumulation makes the result the numerical reference rather than an approximation
of one.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

LOG2_E = 1.4426950408889634

# KV positions processed per chunk. 2048 keeps the score tile small enough that a
# long-context decode does not materialize a [B, H, Q, kv_len] tensor.
DEFAULT_KV_CHUNK = 2048


def build_gather_index(
    kv_indptr: Tensor,
    kv_indices: Tensor,
    last_page_len: Tensor,
    page_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten the page table into per-sequence slot indices.

    Returns (gather_idx [B, Lmax], valid [B, Lmax], kv_lens [B]) where gather_idx
    indexes the *flattened* cache of shape [num_pages * page_size, ...]. Positions
    where `valid` is False are arbitrary and must be masked by the caller.
    """
    device = kv_indices.device
    b = kv_indptr.numel() - 1

    pages_per_seq = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int64)
    kv_lens = (pages_per_seq - 1).clamp(min=0) * page_size + last_page_len.to(
        torch.int64
    )
    # A sequence with zero pages has zero keys.
    kv_lens = torch.where(pages_per_seq == 0, torch.zeros_like(kv_lens), kv_lens)

    l_max = int(kv_lens.max().item()) if b > 0 else 0
    if l_max == 0:
        empty = torch.zeros((b, 0), dtype=torch.int64, device=device)
        return empty, empty.bool(), kv_lens

    pos = torch.arange(l_max, device=device).unsqueeze(0).expand(b, l_max)  # [B, Lmax]
    valid = pos < kv_lens.unsqueeze(1)

    # slot = page_id(page_table[seq, pos // page_size]) * page_size + pos % page_size
    page_slot = pos // page_size
    # Clamp so out-of-range lookups stay in bounds; they are masked out anyway.
    page_slot = torch.minimum(page_slot, (pages_per_seq - 1).clamp(min=0).unsqueeze(1))
    flat_page_pos = kv_indptr[:-1].to(torch.int64).unsqueeze(1) + page_slot
    flat_page_pos = flat_page_pos.clamp(min=0, max=max(kv_indices.numel() - 1, 0))
    page_ids = kv_indices.to(torch.int64)[flat_page_pos]  # [B, Lmax]
    gather_idx = page_ids * page_size + (pos % page_size)

    return gather_idx, valid, kv_lens


def paged_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    kv_indptr: Tensor,
    kv_indices: Tensor,
    last_page_len: Tensor,
    page_size: int,
    qo_indptr: Tensor | None = None,
    causal: bool = False,
    sm_scale: float | None = None,
    window_left: int = -1,
    kv_chunk: int = DEFAULT_KV_CHUNK,
) -> tuple[Tensor, Tensor]:
    """Paged attention over a ragged batch.

    q: [num_q_tokens, num_qo_heads, head_dim]. When `qo_indptr` is None every
    sequence contributes exactly one query (the decode case). `window_left` >= 0 limits
    each query to itself and the `window_left` keys before it (flashinfer semantics).

    Returns (out [num_q_tokens, num_qo_heads, head_dim] in q.dtype,
             lse [num_q_tokens, num_qo_heads] fp32, BASE 2).
    """
    device = q.device
    num_q_tokens, hq, d = q.shape
    hkv = k_cache.shape[2]
    assert hq % hkv == 0, f"num_qo_heads {hq} not divisible by num_kv_heads {hkv}"
    group = hq // hkv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    b = kv_indptr.numel() - 1
    if qo_indptr is None:
        qo_indptr = torch.arange(b + 1, device=device, dtype=torch.int32)

    out = torch.zeros((num_q_tokens, hq, d), device=device, dtype=q.dtype)
    lse = torch.full(
        (num_q_tokens, hq), float("-inf"), device=device, dtype=torch.float32
    )

    if num_q_tokens == 0 or kv_indices.numel() == 0:
        return out, lse

    gather_idx, _, kv_lens = build_gather_index(
        kv_indptr, kv_indices, last_page_len, page_size
    )
    if gather_idx.numel() == 0:
        return out, lse

    flat_k = k_cache.view(-1, hkv, d)
    flat_v = v_cache.view(-1, hkv, d)

    qo_indptr_c = qo_indptr.to(torch.int64).cpu()
    kv_lens_c = kv_lens.cpu()

    # Sequences are handled independently: their KV lengths differ by orders of
    # magnitude in a real batch, so padding them into one tensor wastes far more
    # work than the Python-level loop costs.
    for s in range(b):
        q0, q1 = int(qo_indptr_c[s]), int(qo_indptr_c[s + 1])
        n_q = q1 - q0
        n_keys = int(kv_lens_c[s])
        if n_q == 0 or n_keys == 0:
            continue

        qs = q[q0:q1].view(n_q, hkv, group, d).to(torch.float32)
        qs = qs.permute(1, 2, 0, 3)  # [Hkv, G, Q, D]

        m = torch.full(
            (hkv, group, n_q), float("-inf"), device=device, dtype=torch.float32
        )
        denom = torch.zeros((hkv, group, n_q), device=device, dtype=torch.float32)
        acc = torch.zeros((hkv, group, n_q, d), device=device, dtype=torch.float32)

        # Causal masking places the queries at the tail of the KV.
        q_pos = torch.arange(n_keys - n_q, n_keys, device=device).view(n_q, 1)

        for c0 in range(0, n_keys, kv_chunk):
            c1 = min(c0 + kv_chunk, n_keys)
            idx = gather_idx[s, c0:c1]
            k = flat_k[idx].to(torch.float32)  # [C, Hkv, D]
            v = flat_v[idx].to(torch.float32)

            # [Hkv, G, Q, C]
            scores = torch.einsum("hgqd,chd->hgqc", qs, k) * sm_scale

            k_pos = torch.arange(c0, c1, device=device).view(1, -1)
            if causal:
                scores = scores.masked_fill(
                    (k_pos > q_pos).view(1, 1, n_q, c1 - c0), float("-inf")
                )
            if window_left >= 0:
                outside = (k_pos < q_pos - window_left).view(1, 1, n_q, c1 - c0)
                scores = scores.masked_fill(outside, float("-inf"))

            m_new = torch.maximum(m, scores.amax(dim=-1))
            # Rows still entirely masked stay at -inf; exp(-inf - -inf) is nan, so
            # pin them to 0 and let the final denom == 0 guard produce a -inf LSE.
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(m - m_safe)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
            p = torch.exp(scores - m_safe.unsqueeze(-1))
            p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))

            denom = denom * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.einsum("hgqc,chd->hgqd", p, v)
            m = m_safe

        nonempty = denom > 0
        safe_denom = torch.where(nonempty, denom, torch.ones_like(denom))
        o = acc / safe_denom.unsqueeze(-1)
        o = torch.where(nonempty.unsqueeze(-1), o, torch.zeros_like(o))

        seq_lse = torch.where(
            nonempty,
            (m + torch.log(safe_denom)) * LOG2_E,
            torch.full_like(denom, float("-inf")),
        )

        out[q0:q1] = o.permute(2, 0, 1, 3).reshape(n_q, hq, d).to(q.dtype)
        lse[q0:q1] = seq_lse.permute(2, 0, 1).reshape(n_q, hq)

    return out, lse


def merge_state(
    v_a: Tensor, s_a: Tensor, v_b: Tensor, s_b: Tensor
) -> tuple[Tensor, Tensor]:
    """flashinfer `cascade.merge_state` equivalent.

    Combines two partial attention results given their BASE-2 log-sum-exps.
    Returns (merged_out in v_a.dtype, merged_lse fp32 base 2).
    """
    a32, b32 = v_a.float(), v_b.float()
    sa, sb = s_a.float(), s_b.float()

    log_z = torch.logaddexp2(sa, sb)
    w_a = torch.exp2(sa - log_z)
    w_b = torch.exp2(sb - log_z)

    # Both -inf (no keys on either side) => weights are nan; the result is zero.
    both_empty = ~torch.isfinite(log_z)
    w_a = torch.where(both_empty, torch.zeros_like(w_a), w_a)
    w_b = torch.where(both_empty, torch.zeros_like(w_b), w_b)

    while w_a.dim() < a32.dim():
        w_a = w_a.unsqueeze(-1)
        w_b = w_b.unsqueeze(-1)

    merged = a32 * w_a + b32 * w_b
    return merged.to(v_a.dtype), log_z


def sampling_from_probs(probs: Tensor, uniform: Tensor | None = None) -> Tensor:
    """Inverse-CDF categorical sampling, matching flashinfer's semantics.

    flashinfer >= 0.5 dropped the `uniform_samples` argument; both call shapes are
    accepted.
    """
    cdf = probs.float().cumsum(dim=-1)
    cdf = cdf / cdf[..., -1:].clamp(min=1e-20)
    if uniform is None:
        uniform = torch.rand(probs.shape[:-1], device=probs.device, dtype=torch.float32)
    u = uniform.reshape(*probs.shape[:-1], 1).to(cdf.dtype)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    return idx.squeeze(-1).clamp(max=probs.shape[-1] - 1).to(torch.int32)
