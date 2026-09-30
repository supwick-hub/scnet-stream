"""`separate`: a file in, four stem WAVs out, memory flat in the track's length."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np

from .audio import FfmpegSource
from .core import OVERLAP, SAMPLE_RATE, SOURCES, Predict, Source, mono_stats, separated_regions
from .wav import WavWriter

# progress(stage, fraction): stage is "download", "load", "read" or "separate".
Progress = Callable[[str, float], None]
Input = Union[str, Path, Source]


def separate_source(
    source: Source,
    out_dir: str | Path,
    predict: Predict,
    *,
    overlap: float = OVERLAP,
    float32: bool = False,
    progress: Progress | None = None,
) -> dict[str, Path]:
    """`separate` with the network given: any `predict(spectrum) -> spectrum` will do."""
    out_dir = Path(out_dir)
    estimate = getattr(source, "estimated_frames", lambda: None)()

    def read_progress(frames: int) -> None:
        if progress is not None and estimate:
            progress("read", min(1.0, frames / estimate))

    counted = _Counting(source, read_progress)
    mean, std, length = mono_stats(counted)
    if progress is not None:
        progress("read", 1.0)
    stage = None if progress is None else (lambda f: progress("separate", f))
    writers = {
        name: WavWriter(out_dir / f"{name}.wav", SAMPLE_RATE, 2, float32=float32)
        for name in SOURCES
    }
    try:
        for region in separated_regions(
            source, predict, mean, std, length, overlap=overlap, progress=stage
        ):
            for i, name in enumerate(SOURCES):
                writers[name].write(np.ascontiguousarray(region[i].T))
        if length == 0 and progress is not None:
            progress("separate", 1.0)
        for writer in writers.values():
            writer.commit()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    return {name: writers[name].path for name in ("vocals", "drums", "bass", "other")}


class _Counting:
    """A source that reports frames read, for the first pass's progress."""

    def __init__(self, source: Source, report: Callable[[int], None]):
        self._source = source
        self._report = report

    def blocks(self):
        n = 0
        for block in self._source.blocks():
            n += len(block)
            self._report(n)
            yield block


def separate(
    input_path: Input,
    out_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    device: str = "gpu",
    overlap: float = OVERLAP,
    float32: bool = False,
    progress: Progress | None = None,
) -> dict[str, Path]:
    """Separate `input_path` into `out_dir/{vocals,drums,bass,other}.wav`; returns name -> path.

    `input_path` is anything ffmpeg can decode (or an object with a re-iterable `blocks()` of
    stereo float32 at 44.1 kHz). It is read twice, a block at a time -- once for the mix's mean and
    standard deviation, once through the network -- and each stem is written as it is finished, so
    memory does not grow with the track's length.

    `model_path`: a `.mlmodelc`, `.mlpackage` or the release `.zip`; by default the pinned release
    is downloaded on first use. `device`: "gpu", "ane" (the Neural Engine; its first load
    prepares the model, which can take around twenty minutes once), "cpu" or "all".
    `overlap`: between segments, 0.25 as in SCNet. `float32`: write 32-bit float WAVs instead of
    dithered 16-bit. `progress(stage, fraction)`, if given, is called as each stage advances.
    """
    from .core import stride_for
    from .model import ensure_model, load

    stride_for(overlap)  # validate before any download
    source = input_path if hasattr(input_path, "blocks") else FfmpegSource(input_path)
    dl = None if progress is None else (lambda f: progress("download", f))
    compiled = ensure_model(model_path, progress=dl)
    if progress is not None:
        progress("load", 0.0)
    predict = load(compiled, device)
    if progress is not None:
        progress("load", 1.0)
    return separate_source(
        source, out_dir, predict, overlap=overlap, float32=float32, progress=progress
    )
