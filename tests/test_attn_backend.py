"""CPU tests for the ROCm attention backend (tokasaurus/model/attn_backend/).

The ROCm branch of the backend is selected by `torch.version.hip`, and its reference
implementation is plain torch, so it can be exercised on any machine by importing the
package with `torch.version.hip` patched. This checks the wrappers against the dense
fp32 oracle in tools/amd_port/golden_spec.py, driven with the exact plan()/run()
keyword arguments upstream tokasaurus passes (flashinfer 0.2.0 naming).

    pytest tests/test_attn_backend.py
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.amd_port.golden_spec import (
    CASES,
    build_inputs,
    reference_attention,
)

PKG = "tokasaurus.model.attn_backend"


@pytest.fixture(scope="module")
def rocm_backend():
    """Import the backend's ROCm branch (reference mode) without touching other tests."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith(PKG)}
    for k in saved:
        del sys.modules[k]

    mp = pytest.MonkeyPatch()
    mp.setattr(torch.version, "hip", "7.1.0-fake", raising=False)
    mp.setenv("TOKA_ROCM_ATTN", "reference")
    try:
        backend = importlib.import_module(PKG)
        assert backend.BACKEND == "rocm-reference"  # hip patched, so the ROCm branch
        yield backend
    finally:
        mp.undo()
        for k in [k for k in sys.modules if k.startswith(PKG)]:
            del sys.modules[k]
        sys.modules.update(saved)


# Small shapes only: the long cases are slow on a CPU and add no coverage here.
CPU_CASES = [
    c
    for c in CASES
    if ("llama31_8b" in c.name or "gptoss" in c.name) and "long" not in c.name
]


def run_wrapper(backend, case, inp):
    """Plan + run exactly like LlamaForCausalLM.plan and tokasaurus_attention do."""
    kv = (inp["k_cache"], inp["v_cache"])
    common = dict(
        num_kv_heads=case.num_kv_heads,
        num_qo_heads=case.num_qo_heads,
        head_dim=case.head_dim,
        page_size=case.page_size,
        q_data_type=case.torch_dtype,
        kv_data_type=case.torch_dtype,
        non_blocking=False,
    )
    if all(n == 1 for n in case.qo_lens) and not case.causal:
        wrapper = backend.BatchDecodeWithPagedKVCacheWrapper(
            None, use_tensor_cores=True
        )
        wrapper.plan(
            indptr=inp["kv_indptr"],
            indices=inp["kv_indices"],
            last_page_len=inp["last_page_len"],
            **common,
        )
    else:
        wrapper = backend.BatchPrefillWithPagedKVCacheWrapper(None)
        wrapper.plan(
            qo_indptr=inp["qo_indptr"],
            paged_kv_indptr=inp["kv_indptr"],
            paged_kv_indices=inp["kv_indices"],
            paged_kv_last_page_len=inp["last_page_len"],
            causal=case.causal,
            **common,
        )
    return wrapper.run_return_lse(q=inp["q"], paged_kv_cache=kv)


@pytest.mark.parametrize("case", CPU_CASES, ids=lambda c: c.name)
def test_matches_dense_reference(rocm_backend, case):
    inp = build_inputs(case, device="cpu")
    out, lse = run_wrapper(rocm_backend, case, inp)
    ref_out, ref_lse = reference_attention(case, inp)

    # bf16 output: two correct implementations differ by rounding (~1e-2 here).
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


def test_merge_state_equals_attention_over_union(rocm_backend):
    """Hydragen's invariant: merging two partial results == attending over both."""
    torch.manual_seed(0)
    q = torch.randn(5, 4, 16)
    k_a, v_a = torch.randn(11, 4, 16), torch.randn(11, 4, 16)
    k_b, v_b = torch.randn(7, 4, 16), torch.randn(7, 4, 16)

    def attend(k, v):
        scores = torch.einsum("thd,lhd->thl", q, k) / 4.0
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.einsum("thl,lhd->thd", torch.softmax(scores, dim=-1), v)
        return out, lse * 1.4426950408889634  # base 2, flashinfer's convention

    out_a, lse_a = attend(k_a, v_a)
    out_b, lse_b = attend(k_b, v_b)
    out_ab, lse_ab = attend(torch.cat([k_a, k_b]), torch.cat([v_a, v_b]))

    merged, merged_lse = rocm_backend.cascade.merge_state(out_a, lse_a, out_b, lse_b)
    torch.testing.assert_close(merged, out_ab, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(merged_lse, lse_ab, atol=1e-5, rtol=1e-5)


def test_sampling_from_probs_inverse_cdf(rocm_backend):
    probs = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
    uniform = torch.tensor([0.25, 0.99])
    ids = rocm_backend.sampling_from_probs(probs, uniform)
    assert ids.tolist() == [1, 1]
    assert ids.dtype == torch.int32
