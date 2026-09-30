"""SCNet's chunked inference as a stream.

This is official SCNet's `apply_model` with `split=True, shifts=0`: the whole track normalised by
one mean and standard deviation, then cut into 7.8 s segments that overlap, each weighted by a
triangle and overlap-added. A straightforward runner keeps a (4, 2, track length) accumulator and
the track itself; here the accumulators are one segment long, and every stretch of output is handed
on as soon as no later segment can touch it. The segments, weights and float32 arithmetic are the
same, in the same order, so the output is the same numbers (`tests/test_core.py`).

The price is reading the input twice: once for the statistics, once through the network.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Protocol

import numpy as np

from .spectral import spectrum_to_waveform, waveform_to_spectrum

SAMPLE_RATE = 44100
SEGMENT = 343980  # 7.8 s: the input length the Core ML export was traced with
OVERLAP = 0.25
# The export's output order.
SOURCES = ("drums", "bass", "other", "vocals")


class Source(Protocol):
    """Stereo float32 audio at 44.1 kHz as blocks shaped (frames, 2). `blocks()` may be called
    again for another pass and must yield the same samples each time."""

    def blocks(self) -> Iterable[np.ndarray]: ...


def mono_stats(source: Source) -> tuple[float, float, int]:
    """(mean, unbiased std, frames) of the mono fold, in one streamed pass.

    Blocks are combined in float64 with Chan et al.'s pairwise update, which agrees with the
    whole-array float32 computation to about 1e-7 relative.
    """
    n = 0
    mean = 0.0
    m2 = 0.0
    for block in source.blocks():
        mono = block.mean(axis=1, dtype=np.float32).astype(np.float64)
        k = mono.shape[0]
        if k == 0:
            continue
        block_mean = float(mono.mean())
        block_m2 = float(np.square(mono - block_mean).sum())
        delta = block_mean - mean
        total = n + k
        mean += delta * k / total
        m2 += block_m2 + delta * delta * n * k / total
        n = total
    std = (m2 / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std, n


def triangle(segment: int = SEGMENT) -> np.ndarray:
    half = segment // 2
    weight = np.concatenate([np.arange(1, half + 1), np.arange(segment - half, 0, -1)])
    weight = weight.astype(np.float32)
    return weight / weight.max()


def stride_for(overlap: float) -> int:
    if not 0.0 <= overlap <= 0.9:
        raise ValueError(f"overlap must be between 0 and 0.9; got {overlap}")
    return int((1 - overlap) * SEGMENT)


Predict = Callable[[np.ndarray], np.ndarray]


def separated_regions(
    source: Source,
    predict: Predict,
    mean: float,
    std: float,
    length: int,
    *,
    overlap: float = OVERLAP,
    progress: Callable[[float], None] | None = None,
) -> Iterator[np.ndarray]:
    """Yields (4, 2, n) float32 stems for successive stretches of the input, in order, each final.

    `predict(spectrum) -> spectrum` runs the network on one segment: (1, 4, F, T) in,
    (1, 4 sources, 4, F, T) out. `progress` gets the fraction of the input through the network
    after each segment.
    """
    stride = stride_for(overlap)
    weight = triangle()
    f_mean = np.float32(mean)
    f_std = np.float32(std)
    out = np.zeros((len(SOURCES), 2, SEGMENT), dtype=np.float32)
    total = np.zeros(SEGMENT, dtype=np.float32)
    reader = iter(source.blocks())
    buf = np.empty((0, 2), dtype=np.float32)
    buf_start = 0
    for offset in range(0, length, stride):
        need = min(offset + SEGMENT, length)
        while buf_start + len(buf) < need:
            try:
                block = next(reader)
            except StopIteration:
                raise ValueError(
                    f"the source ended at frame {buf_start + len(buf)} on its second pass; "
                    f"the first pass counted {length}"
                ) from None
            buf = np.concatenate([buf, np.asarray(block, dtype=np.float32)])
        chunk = np.ascontiguousarray(buf[offset - buf_start : need - buf_start].T)
        chunk = (chunk - f_mean) / (f_std + np.float32(1e-8))
        n = chunk.shape[-1]
        if n < SEGMENT:
            chunk = np.pad(chunk, ((0, 0), (0, SEGMENT - n)))
        spectrum, s_mean, s_std, padding = waveform_to_spectrum(chunk[None])
        predicted = np.asarray(predict(spectrum), dtype=np.float32)
        stems = spectrum_to_waveform(predicted, s_mean, s_std, padding)[0]
        out[..., :n] += weight[:n] * stems[..., :n]
        total[:n] += weight[:n]
        # Everything before the next segment's start has had every contribution it will get.
        final = stride if offset + stride < length else n
        yield out[..., :final] / total[:final] * f_std + f_mean
        if progress is not None:
            progress(min(1.0, (offset + n) / length))
        out[..., : SEGMENT - final] = out[..., final:]
        out[..., SEGMENT - final :] = 0
        total[: SEGMENT - final] = total[final:]
        total[SEGMENT - final :] = 0
        drop = offset + final - buf_start
        buf = buf[drop:]
        buf_start += drop
