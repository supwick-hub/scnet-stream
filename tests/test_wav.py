from __future__ import annotations

import wave

import numpy as np
import pytest

from scnet_stream.wav import WavWriter

from .conftest import music


def _write(path, frames, step, **kw):
    w = WavWriter(path, 44100, 2, **kw)
    for start in range(0, len(frames), step):
        w.write(frames[start : start + step])
    return w.commit()


def test_16_bit_round_trip_is_within_dither(tmp_path):
    frames = music(2.0) * 0.5
    path = _write(tmp_path / "a.wav", frames, 10_000)
    with wave.open(str(path)) as r:
        assert (r.getnchannels(), r.getsampwidth(), r.getframerate()) == (2, 2, 44100)
        data = np.frombuffer(r.readframes(r.getnframes()), "<i2").reshape(-1, 2)
    assert data.shape == frames.shape
    err = data / 32767 - frames
    assert np.max(np.abs(err)) <= 1.5 / 32767  # half an LSB of rounding plus one of dither
    assert abs(err.mean()) < 1e-5  # no DC


def test_float32_round_trip_is_exact(tmp_path):
    sf = pytest.importorskip("soundfile")
    frames = music(1.0) * 1.3  # float keeps values past full scale
    path = _write(tmp_path / "f.wav", frames, 3_333, float32=True)
    back, rate = sf.read(str(path), dtype="float32")
    info = sf.info(str(path))
    assert rate == 44100 and info.subtype == "FLOAT"
    assert np.array_equal(back, frames)


def test_16_bit_reads_back_in_soundfile(tmp_path):
    sf = pytest.importorskip("soundfile")
    frames = music(1.0) * 0.5
    path = _write(tmp_path / "a.wav", frames, 50_000)
    back, rate = sf.read(str(path), dtype="float32")
    assert back.shape == frames.shape and rate == 44100


def test_bytes_do_not_depend_on_block_size(tmp_path):
    frames = music(2.0) * 0.5
    a = _write(tmp_path / "x" / "s.wav", frames, 12_345).read_bytes()
    b = _write(tmp_path / "y" / "s.wav", frames, len(frames)).read_bytes()
    assert a == b


def test_full_scale_clips_rather_than_wraps(tmp_path):
    frames = np.full((1000, 2), 1.0, dtype=np.float32)
    frames[500:] = -1.0
    path = _write(tmp_path / "c.wav", frames, 1000)
    with wave.open(str(path)) as r:
        data = np.frombuffer(r.readframes(r.getnframes()), "<i2").reshape(-1, 2)
    assert data[:500].min() > 32000 and data[500:].max() < -32000


def test_abort_leaves_nothing(tmp_path):
    w = WavWriter(tmp_path / "gone.wav", 44100, 2)
    w.write(np.zeros((100, 2), np.float32))
    w.abort()
    assert list(tmp_path.iterdir()) == []
