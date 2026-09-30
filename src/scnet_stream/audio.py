"""Audio in, a block at a time, from ffmpeg.

Each call to `blocks()` starts a fresh decode. ffmpeg is deterministic, so every pass sees the same
samples, and a decode runs at several hundred times realtime, so reading a file twice is cheaper
than holding it.

ffmpeg drops the encoder delay that AAC and MP3 files start with (from the MP4 edit list,
`iTunSMPB`, or the LAME/Xing header), so the stems line up sample for sample with what ffmpeg
decodes -- which is what most players and DAWs show. A decoder that kept those samples would see
every stem 20-50 ms late.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from .core import SAMPLE_RATE

# ~6 s of stereo float32: 2 MB a block.
BLOCK_FRAMES = 1 << 18


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FileNotFoundError(f"{tool} not found on PATH; scnet-stream decodes with ffmpeg")
    return path


class FfmpegSource:
    """Any file ffmpeg can read, as stereo float32 at 44.1 kHz. After a complete pass, `frames`
    is the decoded length."""

    def __init__(self, path: str | Path, block_frames: int = BLOCK_FRAMES):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.block_frames = block_frames
        self.frames: int | None = None

    def _command(self) -> list[str]:
        return [
            _require("ffmpeg"), "-nostdin", "-v", "error",
            "-i", str(self.path),
            "-vn", "-sn", "-dn",
            "-f", "f32le", "-ac", "2", "-ar", str(SAMPLE_RATE),
            "-",
        ]  # fmt: skip

    def blocks(self) -> Iterator[np.ndarray]:
        cmd = self._command()
        frame_bytes = 8
        with tempfile.TemporaryFile() as stderr:
            # stderr to a file: a pipe nobody reads can fill and stall ffmpeg.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr)
            total = 0
            carry = b""
            try:
                while True:
                    raw = proc.stdout.read(self.block_frames * frame_bytes)
                    if not raw:
                        break
                    raw = carry + raw
                    usable = len(raw) - len(raw) % frame_bytes
                    carry = raw[usable:]
                    block = np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, 2).copy()
                    total += len(block)
                    yield block
                returncode = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
            if returncode != 0:
                stderr.seek(0)
                message = stderr.read().decode("utf-8", "replace").strip()
                raise RuntimeError(f"ffmpeg failed on {self.path} ({returncode}): {message}")
        self.frames = total

    def estimated_frames(self) -> int | None:
        """The container's stated duration, in frames: good enough for a progress bar."""
        try:
            out = subprocess.run(
                [_require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(self.path)],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()  # fmt: skip
            return int(float(out) * SAMPLE_RATE)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None


class ArraySource:
    """An array already in memory, (frames, 2) float32, served as blocks."""

    def __init__(self, frames: np.ndarray, block_frames: int = BLOCK_FRAMES):
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != 2:
            raise ValueError(f"expected (frames, 2) stereo; got {frames.shape}")
        self._frames = frames
        self.frames = len(frames)
        self.block_frames = block_frames

    def blocks(self) -> Iterator[np.ndarray]:
        for start in range(0, self.frames, self.block_frames):
            yield self._frames[start : start + self.block_frames]

    def estimated_frames(self) -> int:
        return self.frames
