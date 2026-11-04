"""Natively ported models (Gemma 4, Muse Glimmer, GLM-4.6) against transformers 5.15
on tiny random checkpoints.

Each check loads a fixture from tests/fixtures/tiny_models (written by
tools/model_parity/make_tiny_checkpoints.py), runs prefill and a decode step through
tokasaurus' paged attention, and compares full next-token distributions to HF. On a
CPU-only machine it uses the portable attention backend in fp32; with a GPU it also
runs the flashinfer path in bf16. Each check runs in its own process because the
attention backend is chosen at import time.

    pytest tests/test_sliding_window_models.py -s
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "tiny_models"
BACKENDS = ["portable"] + (["flashinfer"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model", ["gemma4", "muse_glimmer", "glm4_moe"])
def test_matches_hf_reference(model, backend):
    model_dir = FIXTURES / model
    if not (model_dir / "reference.pt").exists():
        pytest.skip(
            f"missing fixture {model_dir}; see tools/model_parity/make_tiny_checkpoints.py"
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.model_parity.check_tiny_checkpoint",
            str(model_dir),
            "--backend",
            backend,
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        check=False,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    print(lines[-1] if lines else result.stderr[-2000:])
    assert result.returncode == 0, result.stderr[-4000:] or result.stdout
    assert json.loads(lines[-1])["decode_step_argmax_match"]
