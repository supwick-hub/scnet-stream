"""The NumPy STFT pair against itself and, when torch is installed, against the PyTorch steps the
Core ML export was made with."""

from __future__ import annotations

import numpy as np
import pytest

from scnet_stream.core import SEGMENT
from scnet_stream.spectral import istft, spectrum_to_waveform, stft, waveform_to_spectrum

from .conftest import music


def test_stft_then_istft_is_the_identity():
    x = music(3.0)[: 1024 * 100].T.copy()
    y = istft(stft(x))
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-5


def test_the_spectrum_round_trips_through_every_source():
    x = music(SEGMENT / 44100)[:SEGMENT].T[None].copy()
    spec, mean, std, padding = waveform_to_spectrum(x)
    assert spec.dtype == np.float32
    assert spec.shape[:2] == (1, 4) and spec.shape[2] == 2049
    assert (SEGMENT + padding) % 1024 == 0 and (SEGMENT + padding) // 1024 % 2 == 1
    stems = spectrum_to_waveform(np.repeat(spec[:, None], 4, axis=1), mean, std, padding)
    assert stems.shape == (1, 4, 2, SEGMENT)
    assert np.max(np.abs(stems - x[:, None])) < 1e-4


def _torch_reference(waveform):
    """scnet-executorch's steps outside the network (MIT, Copyright (c) demixr)."""
    import torch

    hop = 1024
    padding = hop - waveform.shape[-1] % hop
    if (waveform.shape[-1] + padding) // hop % 2 == 0:
        padding += hop
    padded = torch.nn.functional.pad(waveform, (0, padding))
    batch, channels, length = padded.shape
    spectrum = torch.stft(padded.reshape(-1, length), n_fft=4096, hop_length=hop, win_length=4096,
                          center=True, normalized=True, return_complex=True)
    spectrum = torch.view_as_real(spectrum)
    f, t = spectrum.shape[1:3]
    spectrum = spectrum.permute(0, 3, 1, 2).reshape(batch, channels * 2, f, t)
    mean = spectrum.mean(dim=(1, 2, 3), keepdim=True)
    std = spectrum.std(dim=(1, 2, 3), keepdim=True)
    return (spectrum - mean) / (1e-5 + std), mean, std, padding


def _torch_inverse(spectrum, mean, std, padding):
    import torch

    batch, sources, channels, f, frames = spectrum.shape
    spectrum = spectrum * std[:, None] + mean[:, None]
    spectrum = spectrum.reshape(-1, 2, f, frames).permute(0, 2, 3, 1)
    waveform = torch.istft(torch.view_as_complex(spectrum.contiguous()), n_fft=4096,
                           hop_length=1024, win_length=4096, center=True, normalized=True)
    return waveform.reshape(batch, sources, channels // 2, -1)[..., :-padding]


def test_matches_the_pytorch_reference():
    torch = pytest.importorskip("torch")
    x = music(SEGMENT / 44100, seed=3)[:SEGMENT].T[None].copy()
    ours, mean, std, padding = waveform_to_spectrum(x)
    theirs, t_mean, t_std, t_padding = _torch_reference(torch.from_numpy(x))
    assert padding == t_padding
    assert ours.shape == tuple(theirs.shape)
    assert np.max(np.abs(ours - theirs.numpy())) < 1e-4
    assert abs(float(std.ravel()[0]) - float(t_std)) / float(t_std) < 1e-5

    rng = np.random.default_rng(0)
    out = (ours[:, None] * rng.uniform(0.5, 1.5, (1, 4, 1, 1, 1))).astype(np.float32)
    mine = spectrum_to_waveform(out, mean, std, padding)
    ref = _torch_inverse(torch.from_numpy(out), t_mean, t_std, t_padding).numpy()
    assert mine.shape == ref.shape
    assert np.max(np.abs(mine - ref)) < 1e-4
