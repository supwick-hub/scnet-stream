from __future__ import annotations

import hashlib

import pytest

from scnet_stream import model


def test_download_checks_the_hash(tmp_path):
    src = tmp_path / "asset.zip"
    src.write_bytes(b"x" * 3_000_000)
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    seen = []
    dest = model.download(tmp_path / "cache" / "a.zip", src.as_uri(), good, seen.append)
    assert dest.read_bytes() == src.read_bytes()
    assert seen and seen[-1] == 1.0

    with pytest.raises(RuntimeError, match="SHA-256"):
        model.download(tmp_path / "cache" / "b.zip", src.as_uri(), "0" * 64)
    assert not list((tmp_path / "cache").glob("b.zip*"))


def test_cache_dir_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SCNET_STREAM_CACHE", str(tmp_path))
    assert model.cache_dir() == tmp_path


def test_unknown_model_path_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.ensure_model(tmp_path / "missing.mlmodelc")


def test_unknown_device_is_refused(tmp_path):
    pytest.importorskip("coremltools")
    with pytest.raises(ValueError, match="device"):
        model.load(tmp_path, "tpu")
