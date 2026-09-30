"""A WAV writer that takes a block at a time: 16-bit PCM with TPDF dither, or 32-bit float.

Written to a temporary name and renamed into place on `commit`, so a reader never sees half a
file; the RIFF sizes are patched in at the end, so the length need not be known up front.

**Why dither.** Converting float to 16-bit by truncation (which libsndfile does) adds half an LSB
of DC and turns the quantisation error into distortion correlated with the signal, audible on
fades and reverb tails. Rounding removes the DC but not the correlation; TPDF dither -- the sum of
two uniform variates, 1 LSB peak to peak -- makes the error independent of the signal, for about
4.8 dB more noise at around -90 dBFS.

**Deterministic.** The dither is seeded from the file's name and drawn from two independent
streams, one per variate, so the bytes depend only on the name and the samples -- not on how the
audio was cut into blocks. Separating the same input twice gives the same files.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import struct
from pathlib import Path

import numpy as np

FULL_SCALE_16 = 32767
INT16_MIN = -32768
_PCM = 1
_IEEE_FLOAT = 3
_MAX_DATA_BYTES = 0xFFFFFFFF - 64  # RIFF sizes are 32-bit
# Bounds the float64 temporaries of the dither: 2^18 stereo frames is ~4 MB each.
WRITE_BLOCK = 1 << 18


class Ditherer:
    def __init__(self, seed: str):
        key = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
        a, b = np.random.SeedSequence(key).spawn(2)
        self._a = np.random.default_rng(a)
        self._b = np.random.default_rng(b)

    def quantise(self, frames: np.ndarray) -> np.ndarray:
        """float in [-1, 1] -> int16, TPDF-dithered, rounded to nearest, clipped."""
        scaled = np.asarray(frames, dtype=np.float64) * FULL_SCALE_16
        dither = self._a.random(scaled.shape) - self._b.random(scaled.shape)
        # Clipped rather than wrapped: dither can push a full-scale sample past the range, and a
        # wrapped peak is a click.
        quantised = np.floor(scaled + dither + 0.5)
        return np.clip(quantised, INT16_MIN, FULL_SCALE_16).astype("<i2")


class WavWriter:
    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        channels: int = 2,
        *,
        float32: bool = False,
        seed: str | None = None,
    ):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.float32 = float32
        self.frames = 0
        self._sample_bytes = 4 if float32 else 2
        self._dither = None if float32 else Ditherer(seed if seed is not None else self.path.name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(6)}.part")
        self._file = open(self._tmp, "wb")
        self._file.write(self._header(0))

    def _header(self, frames: int) -> bytes:
        block_align = self.channels * self._sample_bytes
        data_bytes = frames * block_align
        fmt_tag = _IEEE_FLOAT if self.float32 else _PCM
        fmt = struct.pack(
            "<HHIIHH", fmt_tag, self.channels, self.sample_rate,
            self.sample_rate * block_align, block_align, 8 * self._sample_bytes,
        )  # fmt: skip
        chunks = b""
        if self.float32:
            # A non-PCM format carries cbSize and a fact chunk.
            fmt += struct.pack("<H", 0)
            chunks += b"fmt " + struct.pack("<I", len(fmt)) + fmt
            chunks += b"fact" + struct.pack("<II", 4, frames)
        else:
            chunks += b"fmt " + struct.pack("<I", len(fmt)) + fmt
        chunks += b"data" + struct.pack("<I", data_bytes)
        return b"RIFF" + struct.pack("<I", 4 + len(chunks) + data_bytes) + b"WAVE" + chunks

    def write(self, frames: np.ndarray) -> None:
        """`frames`: float, (n, channels), nominally in [-1, 1]."""
        frames = np.asarray(frames)
        if frames.ndim != 2 or frames.shape[1] != self.channels:
            raise ValueError(f"expected (n, {self.channels}); got {frames.shape}")
        added = (self.frames + len(frames)) * self.channels * self._sample_bytes
        if added > _MAX_DATA_BYTES:
            raise ValueError("WAV data would exceed 4 GB; use 16-bit output or a shorter input")
        for start in range(0, len(frames), WRITE_BLOCK):
            block = frames[start : start + WRITE_BLOCK]
            if self._dither is not None:
                data = self._dither.quantise(block)
            else:
                data = np.asarray(block, dtype="<f4")
            self._file.write(np.ascontiguousarray(data).tobytes())
            self.frames += len(block)

    def commit(self) -> Path:
        self._file.seek(0)
        self._file.write(self._header(self.frames))
        self._file.close()
        os.replace(self._tmp, self.path)
        return self.path

    def abort(self) -> None:
        if not self._file.closed:
            self._file.close()
        self._tmp.unlink(missing_ok=True)
