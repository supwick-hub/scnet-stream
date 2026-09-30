"""The streamed loop against the whole-array loop it replaces, and SCNet's arithmetic around the
network: anything that drops a segment, double-counts an overlap or forgets to denormalise shows
up as a stem that is not the mix."""

from __future__ import annotations

import numpy as np
import pytest

from scnet_stream.audio import ArraySource
from scnet_stream.core import (
    SEGMENT,
    SOURCES,
    mono_stats,
    separated_regions,
    stride_for,
    triangle,
)
from scnet_stream.spectral import spectrum_to_waveform, waveform_to_spectrum

from .conftest import pass_through

# Treats each source differently, so a mixed-up source shows.
GAINS = np.array([1.0, 0.5, -0.25, 2.0], dtype=np.float32)[None, :, None, None, None]


def gained(spectrum):
    return pass_through(spectrum) * GAINS


def whole_array(mix: np.ndarray, predict, mean: float, std: float, overlap: float) -> np.ndarray:
    """The straightforward runner: the whole track normalised, a track-length accumulator."""
    f_mean, f_std = np.float32(mean), np.float32(std)
    x = (np.ascontiguousarray(mix.T) - f_mean) / (f_std + np.float32(1e-8))
    length = x.shape[-1]
    stride = stride_for(overlap)
    weight = triangle()
    out = np.zeros((len(SOURCES), 2, length), dtype=np.float32)
    total = np.zeros(length, dtype=np.float32)
    for offset in range(0, length, stride):
        chunk = x[:, offset : offset + SEGMENT]
        n = chunk.shape[-1]
        if n < SEGMENT:
            chunk = np.pad(chunk, ((0, 0), (0, SEGMENT - n)))
        spectrum, s_mean, s_std, padding = waveform_to_spectrum(chunk[None])
        stems = spectrum_to_waveform(predict(spectrum), s_mean, s_std, padding)[0]
        out[..., offset : offset + n] += weight[:n] * stems[..., :n]
        total[offset : offset + n] += weight[:n]
    return out / total * f_std + f_mean


def _noise(seconds_in_segments: float, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SEGMENT * seconds_in_segments)
    return (rng.standard_normal((n, 2)) * 0.1 + 0.01).astype(np.float32)


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("segments", [0.4, 2.3])
def test_streamed_is_the_whole_array_loop(overlap, segments):
    mix = _noise(segments)
    mean, std, length = mono_stats(ArraySource(mix, block_frames=100_000))
    assert length == len(mix)
    whole = whole_array(mix, gained, mean, std, overlap)
    regions = list(
        separated_regions(
            ArraySource(mix, block_frames=100_000), gained, mean, std, length, overlap=overlap
        )
    )
    streamed = np.concatenate(regions, axis=-1)
    assert streamed.shape == whole.shape
    # Same operations in the same order: expected to be bit-identical, not merely close.
    assert np.array_equal(streamed, whole)


def test_mono_stats_are_the_whole_array_stats():
    mix = _noise(1.7, seed=9)
    mean, std, n = mono_stats(ArraySource(mix, block_frames=77_777))
    mono = mix.mean(axis=1, dtype=np.float32).astype(np.float64)
    assert n == len(mix)
    assert abs(mean - mono.mean()) < 1e-9
    assert abs(std - mono.std(ddof=1)) / std < 1e-9


def test_a_pass_through_network_gives_back_the_mix():
    mix = _noise(2.5)
    mean, std, length = mono_stats(ArraySource(mix))
    out = np.concatenate(list(separated_regions(ArraySource(mix), pass_through, mean, std, length)), -1)
    for stem in out:
        assert np.max(np.abs(stem.T - mix)) < 1e-4


def test_progress_is_monotonic_and_ends_at_one():
    mix = _noise(2.5)
    mean, std, length = mono_stats(ArraySource(mix))
    seen: list[float] = []
    for _ in separated_regions(ArraySource(mix), pass_through, mean, std, length, progress=seen.append):
        pass
    assert seen == sorted(seen) and 0 < seen[0] < 1 and seen[-1] == 1.0
    assert len(seen) == len(range(0, length, stride_for(0.25)))


def test_a_short_second_pass_is_an_error():
    mix = _noise(1.5)
    mean, std, length = mono_stats(ArraySource(mix))
    with pytest.raises(ValueError, match="second pass"):
        list(separated_regions(ArraySource(mix[:1000]), pass_through, mean, std, length))


def test_overlap_out_of_range_is_refused():
    with pytest.raises(ValueError):
        stride_for(0.95)
