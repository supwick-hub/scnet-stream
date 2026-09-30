"""Time and peak memory of a separation, on synthetic audio generated here.

    python benchmarks/bench.py make 16 /tmp/synth-16.m4a      # a 16-minute synthetic AAC file
    python benchmarks/bench.py run /tmp/synth-16.m4a out/     # streamed, as `separate` does
    python benchmarks/bench.py whole /tmp/synth-16.m4a out/   # a straightforward whole-array runner

`run` and `whole` print one JSON line: seconds per stage and the process's peak phys_footprint.
Run each in a fresh process, with MallocLargeCache=0 so the footprint is live memory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scnet_stream import core, pipeline
from scnet_stream.audio import FfmpegSource
from scnet_stream.macos import peak_footprint_bytes
from scnet_stream.model import ensure_model, load
from scnet_stream.pipeline import separate_source
from scnet_stream.wav import WavWriter

SR = 44100
TIMES: dict[str, float] = {}


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMES[name] = TIMES.get(name, 0.0) + time.perf_counter() - t

    return wrapper


def make(minutes: float, dest: Path) -> None:
    """Kick, bass, chords, hats and a vibrato lead, 124 BPM, rendered a block at a time."""
    rng = np.random.default_rng(0)
    wav = dest.with_suffix(".tmp.wav")
    writer = WavWriter(wav, SR, 2)
    total = int(minutes * 60 * SR)
    beat = 60 / 124
    for start in range(0, total, 1 << 18):
        t = np.arange(start, min(start + (1 << 18), total)) / SR
        pos = (t % beat) / beat
        bar = np.floor(t / (4 * beat))
        kick = np.sin(2 * np.pi * 50 * t * (1 + 0.5 * np.exp(-pos * 30))) * np.exp(-pos * 10)
        root = np.array([55.0, 65.41, 49.0, 73.42])[(bar % 4).astype(int)]
        bass = 0.25 * np.sign(np.sin(2 * np.pi * root * t)) * (pos < 0.6)
        chord = sum(0.05 * np.sin(2 * np.pi * root * m * t) for m in (4, 5, 6))
        hats = 0.05 * rng.standard_normal(len(t)) * np.exp(-((t + beat / 2) % beat) / beat * 40)
        lead = 0.12 * np.sin(2 * np.pi * root * 8 * t + 3 * np.sin(2 * np.pi * 5.5 * t))
        lead *= (bar % 8) >= 4
        mono = 0.5 * kick + bass + chord + hats + lead
        writer.write(np.stack([mono, 0.9 * mono + 0.1 * lead], axis=1) * 0.5)
    writer.commit()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", "256k", str(dest)], check=True)  # fmt: skip
    wav.unlink()



def run(src: Path, out: Path, device: str, whole: bool) -> dict:
    t0 = time.perf_counter()
    compiled = ensure_model()
    predict = timed("network", load(compiled, device))
    TIMES["load"] = time.perf_counter() - t0
    core.waveform_to_spectrum = timed("stft", core.waveform_to_spectrum)
    core.spectrum_to_waveform = timed("istft", core.spectrum_to_waveform)
    pipeline.mono_stats = timed("first_pass", pipeline.mono_stats)
    WavWriter.write = timed("write", WavWriter.write)
    source = FfmpegSource(src)
    t1 = time.perf_counter()
    if whole:
        _whole(source, out, predict)
    else:
        separate_source(source, out, predict)
    TIMES["total"] = time.perf_counter() - t1
    return {"seconds": {k: round(v, 1) for k, v in TIMES.items()},
            "peak_mb": round((peak_footprint_bytes() or 0) / (1 << 20))}  # fmt: skip


def _whole(source, out: Path, predict) -> None:
    """The straightforward runner: decode everything, track-length accumulators, write at the end."""
    mix = np.concatenate(list(source.blocks()))
    mono = mix.mean(axis=1)
    mean, std = np.float32(mono.mean()), np.float32(mono.std(ddof=1))
    x = (mix.T - mean) / (std + np.float32(1e-8))
    length = x.shape[-1]
    stride = core.stride_for(core.OVERLAP)
    weight = core.triangle()
    acc = np.zeros((4, 2, length), dtype=np.float32)
    total = np.zeros(length, dtype=np.float32)
    for offset in range(0, length, stride):
        chunk = x[:, offset : offset + core.SEGMENT]
        n = chunk.shape[-1]
        if n < core.SEGMENT:
            chunk = np.pad(chunk, ((0, 0), (0, core.SEGMENT - n)))
        spec, s_mean, s_std, padding = core.waveform_to_spectrum(chunk[None])
        stems = core.spectrum_to_waveform(np.asarray(predict(spec)), s_mean, s_std, padding)[0]
        acc[..., offset : offset + n] += weight[:n] * stems[..., :n]
        total[offset : offset + n] += weight[:n]
    stems = acc / total * std + mean
    for i, name in enumerate(core.SOURCES):
        w = WavWriter(out / f"{name}.wav", SR, 2)
        w.write(stems[i].T)
        w.commit()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(float(sys.argv[2]), Path(sys.argv[3]))
    else:
        device = sys.argv[4] if len(sys.argv) > 4 else "gpu"
        print(json.dumps(run(Path(sys.argv[2]), Path(sys.argv[3]), device, cmd == "whole")))
