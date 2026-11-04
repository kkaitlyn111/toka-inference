"""The flashinfer entry points tokasaurus uses, dispatched by platform.

On CUDA these are flashinfer's own symbols, so the NVIDIA path is unchanged. flashinfer
is CUDA-only, so on ROCm (or with TOKA_PORTABLE_ATTN=1 anywhere, e.g. for CPU tests)
the paged prefill/decode wrappers, `cascade.merge_state` and `sampling_from_probs` are
reimplemented on AMD's AITER kernels or portable torch ops. LSEs are base 2, as in
flashinfer.
"""

from __future__ import annotations

import os

import torch

IS_ROCM = torch.version.hip is not None
USE_PORTABLE = IS_ROCM or os.environ.get("TOKA_PORTABLE_ATTN") == "1"


if not USE_PORTABLE:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        cascade,
        sampling_from_probs,
    )

    BACKEND = "flashinfer"

else:
    import warnings
    from types import SimpleNamespace

    from torch import Tensor

    from tokasaurus.model.attn_backend.paged_attention import (
        merge_state,
        paged_attention,
        sampling_from_probs,
    )

    # "aiter": one AITER launch per batch. "reference": the portable implementation,
    # which is also the oracle AITER is validated against and the fallback for
    # calls AITER can't serve.
    _MODE = (
        os.environ.get("TOKA_ROCM_ATTN", "aiter" if IS_ROCM else "reference")
        .strip()
        .lower()
    )
    if _MODE not in ("aiter", "reference"):
        raise ValueError(
            f"TOKA_ROCM_ATTN must be 'aiter' or 'reference', got {_MODE!r}"
        )

    # AITER's CK prefill kernel rejects head_dim > 256 (Gemma 4's global layers)
    AITER_MAX_HEAD_DIM = 256

    # AITER's causal paged prefill is intermittently wrong on gfx950 at head_dim 128
    # (0.17-0.35 max deviation vs ~8e-3 expected); see tools/amd_port/aiter_causal_bug.py.
    # Causal prefill therefore uses the reference path unless this is set.
    _ALLOW_CAUSAL = os.environ.get("TOKA_ROCM_AITER_CAUSAL", "0") == "1"

    _aiter_ok = False
    if _MODE == "aiter":
        try:
            from tokasaurus.model.attn_backend.aiter_attention import (
                AiterPagedPlan,
                aiter_paged_attention,
            )

            _aiter_ok = True
        except (
            ImportError,
            OSError,
            RuntimeError,
        ) as e:  # missing or unbuildable AITER
            warnings.warn(
                f"AITER unavailable ({e!r}); using the reference attention backend, "
                "which is correct but does not scale with batch size",
                RuntimeWarning,
                stacklevel=2,
            )

    if IS_ROCM:
        BACKEND = "rocm-aiter" if _aiter_ok else "rocm-reference"
    else:
        BACKEND = "portable-reference"

    cascade = SimpleNamespace(merge_state=merge_state)

    class _PlannedWrapper:
        """flashinfer's plan()/run() contract: plan() records the page table once per
        forward step, run() is called once per layer."""

        def __init__(
            self, workspace_buffer: Tensor | None = None, kv_layout: str = "NHD"
        ):
            assert kv_layout == "NHD", (
                f"only the NHD layout is implemented, got {kv_layout}"
            )
            # flashinfer's attribute name, read when recording cudagraphs
            self._float_workspace_buffer = workspace_buffer
            self._plan: dict | None = None

        @staticmethod
        def _unpack_cache(paged_kv_cache) -> tuple[Tensor, Tensor]:
            if isinstance(paged_kv_cache, (tuple, list)):
                return paged_kv_cache[0], paged_kv_cache[1]
            # fused [num_pages, 2, page_size, num_kv_heads, head_dim] layout
            return paged_kv_cache[:, 0], paged_kv_cache[:, 1]

        def run(self, q: Tensor, paged_kv_cache, *args, **kwargs):
            return self._dispatch(q, paged_kv_cache)[0]

        def run_return_lse(self, q: Tensor, paged_kv_cache, *args, **kwargs):
            return self._dispatch(q, paged_kv_cache)

        def _set_plan(self, logits_soft_cap: float | None = None, **plan) -> None:
            if logits_soft_cap:
                raise NotImplementedError("attention logits soft cap")
            # build the AITER digest here so its host syncs happen once per step,
            # not once per layer
            plan["aiter"] = None
            if (
                _aiter_ok
                and (_ALLOW_CAUSAL or not plan["causal"])
                and plan["window_left"] < 0  # sliding-window layers use the reference
                and plan["head_dim"] <= AITER_MAX_HEAD_DIM
            ):
                plan["aiter"] = AiterPagedPlan(
                    kv_indptr=plan["kv_indptr"],
                    kv_indices=plan["kv_indices"],
                    last_page_len=plan["last_page_len"],
                    page_size=plan["page_size"],
                    num_qo_heads=plan["num_qo_heads"],
                    qo_indptr=plan["qo_indptr"],
                    causal=plan["causal"],
                    sm_scale=plan["sm_scale"],
                )
            self._plan = plan

        def _dispatch(self, q: Tensor, paged_kv_cache) -> tuple[Tensor, Tensor]:
            if (p := self._plan) is None:
                raise RuntimeError("run() called before plan()")
            k_cache, v_cache = self._unpack_cache(paged_kv_cache)
            if p["aiter"] is not None:
                return aiter_paged_attention(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    plan=p["aiter"],
                    sm_scale=p["sm_scale"],
                )
            return paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                kv_indptr=p["kv_indptr"],
                kv_indices=p["kv_indices"],
                last_page_len=p["last_page_len"],
                page_size=p["page_size"],
                qo_indptr=p["qo_indptr"],
                causal=p["causal"],
                sm_scale=p["sm_scale"],
                window_left=p["window_left"],
            )

    class BatchPrefillWithPagedKVCacheWrapper(_PlannedWrapper):
        def plan(
            self,
            qo_indptr: Tensor,
            paged_kv_indptr: Tensor,
            paged_kv_indices: Tensor,
            paged_kv_last_page_len: Tensor,
            num_qo_heads: int,
            num_kv_heads: int,
            head_dim_qk: int | None = None,
            page_size: int | None = None,
            causal: bool = False,
            sm_scale: float | None = None,
            window_left: int = -1,
            logits_soft_cap: float | None = None,
            head_dim: int | None = None,
            **_,
        ):
            # flashinfer 0.2.0 (pinned here) calls it head_dim; later releases head_dim_qk
            head_dim_qk = head_dim_qk if head_dim_qk is not None else head_dim
            assert head_dim_qk is not None and page_size is not None
            self._set_plan(
                qo_indptr=qo_indptr,
                kv_indptr=paged_kv_indptr,
                kv_indices=paged_kv_indices,
                last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim_qk,
                page_size=page_size,
                causal=causal,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
            )

    class BatchDecodeWithPagedKVCacheWrapper(_PlannedWrapper):
        def __init__(
            self,
            workspace_buffer: Tensor | None = None,
            kv_layout: str = "NHD",
            use_cuda_graph: bool = False,
            **_,
        ):
            super().__init__(workspace_buffer, kv_layout)
            self.use_cuda_graph = use_cuda_graph

        def plan(
            self,
            indptr: Tensor,
            indices: Tensor,
            last_page_len: Tensor,
            num_qo_heads: int,
            num_kv_heads: int,
            head_dim: int,
            page_size: int,
            sm_scale: float | None = None,
            window_left: int = -1,
            logits_soft_cap: float | None = None,
            **_,
        ):
            self._set_plan(
                qo_indptr=None,  # decode: one query per sequence
                kv_indptr=indptr,
                kv_indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                page_size=page_size,
                causal=False,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
            )


__all__ = [
    "BACKEND",
    "IS_ROCM",
    "USE_PORTABLE",
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "cascade",
    "sampling_from_probs",
]
