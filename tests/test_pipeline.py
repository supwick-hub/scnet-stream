"""`separate` end to end through ffmpeg, with a pass-through network standing in for Core ML."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from scnet_stream.core import SEGMENT

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _read(path):
    import wave

    with wave.open(str(path)) as r:
        return np.frombuffer(r.readframes(r.getnframes()), "<i2").reshape(-1, 2) / 32767


def _synthetic_file(path, seconds: float, codec: str = "pcm_s16le"):
    """A two-tone signal written by ffmpeg itself: no real audio anywhere."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=44100:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=44100:duration={seconds}",
         "-filter_complex", "[0][1]join=inputs=2:channel_layout=stereo",
         "-c:a", codec, str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_each_stem_is_the_mix_through_a_pass_through_network(fake_model, tmp_path):
    src = _synthetic_file(tmp_path / "in.wav", 2.5 * SEGMENT / 44100)
    seen = []
    paths = fake_model.separate(src, tmp_path / "out", progress=lambda s, f: seen.append((s, f)))
    assert list(paths) == ["vocals", "drums", "bass", "other"]
    mix = _read(src)
    for path in paths.values():
        stem = _read(path)
        assert stem.shape == mix.shape
        assert np.sqrt(np.mean((stem - mix) ** 2)) < 1e-4
    stages = [s for s, _ in seen]
    assert stages.index("load") < stages.index("read") < stages.index("separate")
    assert seen[-1] == ("separate", 1.0)
    assert not list((tmp_path / "out").glob(".*")), "temporary files left behind"


def test_decodes_aac(fake_model, tmp_path):
    src = _synthetic_file(tmp_path / "in.m4a", 10.0, codec="aac")
    paths = fake_model.separate(src, tmp_path / "out", float32=True)
    import soundfile as sf

    stem, rate = sf.read(str(paths["vocals"]), dtype="float32")
    assert rate == 44100 and abs(len(stem) - 441000) < 2048


def test_cli(fake_model, tmp_path, capsys):
    from scnet_stream.cli import main

    src = _synthetic_file(tmp_path / "in.wav", 3.0)
    assert main([str(src), str(tmp_path / "out")]) == 0
    out = capsys.readouterr()
    assert "vocals.wav" in out.out and "Separating" in out.err
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "bass.wav", "drums.wav", "other.wav", "vocals.wav"
    ]


def test_cli_reports_a_missing_file(fake_model, tmp_path, capsys):
    from scnet_stream.cli import main

    assert main([str(tmp_path / "nope.wav"), str(tmp_path / "out")]) == 1
    assert "nope.wav" in capsys.readouterr().err


def test_a_failed_decode_leaves_no_stems(fake_model, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio" * 100)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        fake_model.separate(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
