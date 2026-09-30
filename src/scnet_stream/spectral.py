"""The STFT and its inverse around the network, in NumPy.

The Core ML export takes a normalised spectrogram, not audio, so this is the half of SCNet that runs
outside Core ML. It reproduces the steps of the scnet-executorch export (MIT, Copyright (c) demixr),
which follow official SCNet, including its quirks:

- `torch.stft(..., center=True, normalized=True)` with no window, i.e. a rectangular one;
- right-padding so the frame count is odd;
- channels laid out as [L real, L imag, R real, R imag];
- one mean and one (unbiased) standard deviation over the whole segment's spectrogram.

The reference implementation is PyTorch; this one needs no PyTorch. `tests/test_spectral.py`
checks the two against each other when torch is installed.
"""

from __future__ import annotations

import numpy as np

N_FFT = 4096
HOP = 1024
_OVERLAPS = N_FFT // HOP  # 4 frames cover every sample
_SCALE = np.float32(N_FFT**-0.5)  # normalized=True
_INV_SCALE = np.float32(N_FFT**0.5)


def stft(x: np.ndarray) -> np.ndarray:
    """(..., L) float32 -> (..., N_FFT // 2 + 1, frames) complex64.

    `torch.stft(x, 4096, 1024, 4096, center=True, normalized=True, return_complex=True)`.
    """
    pad = N_FFT // 2
    x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT, axis=-1)[..., ::HOP, :]
    spec = np.fft.rfft(frames, axis=-1).astype(np.complex64, copy=False) * _SCALE
    return np.swapaxes(spec, -1, -2)


def istft(spec: np.ndarray) -> np.ndarray:
    """(..., N_FFT // 2 + 1, frames) complex -> (..., HOP * (frames - 1)) float32.

    `torch.istft(spec, 4096, 1024, 4096, center=True, normalized=True)`: overlap-add, divided by
    the window envelope, which for a rectangular window is the count of frames over each sample.
    """
    n_frames = spec.shape[-1]
    frames = np.fft.irfft(np.swapaxes(spec, -1, -2), n=N_FFT, axis=-1).astype(np.float32, copy=False)
    frames *= _INV_SCALE
    lead = frames.shape[:-2]
    pieces = frames.reshape(*lead, n_frames, _OVERLAPS, HOP)
    out = np.zeros((*lead, n_frames + _OVERLAPS - 1, HOP), dtype=np.float32)
    envelope = np.zeros(n_frames + _OVERLAPS - 1, dtype=np.float32)
    for k in range(_OVERLAPS):
        out[..., k : k + n_frames, :] += pieces[..., :, k, :]
        envelope[k : k + n_frames] += 1
    out = out.reshape(*lead, -1)
    envelope = np.repeat(envelope, HOP)
    start = N_FFT // 2
    end = start + HOP * (n_frames - 1)
    return out[..., start:end] / envelope[start:end]


def waveform_to_spectrum(waveform: np.ndarray):
    """(batch, channels, length) float32 -> (normalised spectrum, mean, std, padding).

    The spectrum is (batch, channels * 2, frequency, frames) float32, the network's input.
    """
    length = waveform.shape[-1]
    padding = HOP - length % HOP
    if (length + padding) // HOP % 2 == 0:
        padding += HOP
    padded = np.pad(waveform, ((0, 0), (0, 0), (0, padding)))
    batch, channels, padded_length = padded.shape
    spec = stft(padded.reshape(-1, padded_length))
    freq, frames = spec.shape[-2:]
    real = np.stack([spec.real, spec.imag], axis=1).reshape(batch, channels * 2, freq, frames)
    mean = real.mean(axis=(1, 2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = real.std(axis=(1, 2, 3), keepdims=True, ddof=1, dtype=np.float64).astype(np.float32)
    return (real - mean) / (np.float32(1e-5) + std), mean, std, padding


def spectrum_to_waveform(spectrum: np.ndarray, mean, std, padding: int) -> np.ndarray:
    """The network's output (batch, sources, channels * 2, frequency, frames) -> waveforms
    (batch, sources, channels, length), undoing `waveform_to_spectrum`."""
    batch, sources, channels2, freq, frames = spectrum.shape
    spec = spectrum * std[:, None] + mean[:, None]
    spec = spec.reshape(-1, 2, freq, frames)
    complex_spec = (spec[:, 0] + 1j * spec[:, 1]).astype(np.complex64, copy=False)
    waveform = istft(complex_spec).reshape(batch, sources, channels2 // 2, -1)
    return waveform[..., :-padding]
