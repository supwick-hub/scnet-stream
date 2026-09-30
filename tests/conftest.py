from __future__ import annotations

import numpy as np
import pytest

from scnet_stream.core import SOURCES

SR = 44100


def music(seconds: float, seed: int = 0, bpm: float = 124.0) -> np.ndarray:
    """Synthetic stereo float32: a kick, a bassline, a chord and a little noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 60.0 / bpm
    pos = (t % beat) / beat
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-pos * 12)
    bass = 0.3 * np.sin(2 * np.pi * 110 * t) * (np.floor(t / (4 * beat)) % 2)
    chord = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63))
    mono = 0.5 * kick + bass + chord + 0.02 * rng.standard_normal(n)
    return np.stack([mono, 0.9 * mono], axis=1).astype(np.float32)


def pass_through(spectrum: np.ndarray) -> np.ndarray:
    """A stand-in network that hands the mix back as every stem."""
    return np.repeat(spectrum[:, None], len(SOURCES), axis=1)


class FakeCompiledModel:
    """Replaces coremltools' CompiledMLModel: pass-through, one input, one output."""

    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, inputs):
        (spectrum,) = inputs.values()
        return {"out": pass_through(spectrum)}


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """`separate` with a pass-through network, and no download."""
    ct = pytest.importorskip("coremltools")
    from scnet_stream import pipeline

    compiled = tmp_path / "model" / "scnet.mlmodelc"
    compiled.mkdir(parents=True)
    monkeypatch.setattr(ct.models, "CompiledMLModel", FakeCompiledModel)
    import scnet_stream.model as model_mod

    monkeypatch.setattr(model_mod, "ensure_model", lambda *a, **k: compiled)
    return pipeline
