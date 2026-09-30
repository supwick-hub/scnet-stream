"""With the real Core ML model: skipped unless it is already on this machine.

Set SCNET_STREAM_MODEL to a .mlmodelc, .mlpackage or the release .zip, or let a previous run
have downloaded it into the cache. Nothing is downloaded here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scnet_stream.audio import ArraySource
from scnet_stream.model import MODEL_VERSION, cache_dir

from .conftest import music

pytestmark = pytest.mark.real_model


def _model_path():
    if env := os.environ.get("SCNET_STREAM_MODEL"):
        return Path(env)
    cached = cache_dir() / f"scnet-coreml-{MODEL_VERSION}" / "scnet.mlmodelc"
    return cached if cached.is_dir() else None


@pytest.fixture(scope="module")
def model_path():
    pytest.importorskip("coremltools")
    path = _model_path()
    if path is None or not path.exists():
        pytest.skip("real model not present")
    return path


def test_stems_add_back_up_to_the_mix(model_path, tmp_path):
    sf = pytest.importorskip("soundfile")
    from scnet_stream import separate

    mix = music(20.0, seed=1) * 0.5
    paths = separate(ArraySource(mix), tmp_path, model_path=model_path, float32=True)
    stems = [sf.read(str(p), dtype="float32")[0] for p in paths.values()]
    for s in stems:
        assert s.shape == mix.shape and np.isfinite(s).all()
    residual = sum(stems) - mix
    snr = 10 * np.log10(np.sum(mix**2) / np.sum(residual**2))
    # SCNet does not enforce that stems sum to the mix, but a working model comes close.
    assert snr > 10, snr
