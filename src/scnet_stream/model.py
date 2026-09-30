"""The Core ML model: fetched once from the scnet-executorch release, verified, compiled, cached.

No weights ship with this package. The first run downloads the pinned release asset, refuses it
unless its SHA-256 matches, unpacks it and compiles it to `.mlmodelc`, which loads in about a
second from then on.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

MODEL_VERSION = "v0.1.2"
MODEL_URL = (
    "https://github.com/demixr/scnet-executorch/releases/download/"
    f"{MODEL_VERSION}/scnet_coreml.mlpackage.zip"
)
MODEL_SHA256 = "d15357c0abc901defb76282178966b0c41cc3e7b9ad34a8a0fd4598b11043c2f"
DEFAULT_INPUT_NAME = "mix_spec"

COMPUTE_UNITS = {"gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE", "cpu": "CPU_ONLY", "all": "ALL"}


def cache_dir() -> Path:
    """`$SCNET_STREAM_CACHE`, else the platform's user cache directory."""
    if env := os.environ.get("SCNET_STREAM_CACHE"):
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "scnet-stream"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "scnet-stream"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    dest: Path,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Fetch `url` to `dest`, refusing it unless its SHA-256 is `expected_sha256`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
        length = int(response.headers.get("Content-Length") or 0)
        got = 0
        while block := response.read(1 << 20):
            out.write(block)
            got += len(block)
            if progress is not None and length > 0:
                progress(min(1.0, got / length))
    digest = sha256(partial)
    if digest != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded model has SHA-256 {digest}, not the pinned {expected_sha256}")
    os.replace(partial, dest)
    return dest


def _compile(package: Path, folder: Path) -> Path:
    import coremltools as ct

    spec = ct.models.MLModel(str(package), skip_model_load=True).get_spec()
    (folder / "io.json").write_text(json.dumps({"input": spec.description.input[0].name}))
    compiled = folder / "scnet.mlmodelc"
    staging = folder / "scnet.staging.mlmodelc"  # compile_model wants the extension
    shutil.rmtree(staging, ignore_errors=True)
    ct.utils.compile_model(str(package), str(staging))
    shutil.rmtree(compiled, ignore_errors=True)
    os.replace(staging, compiled)
    return compiled


def _from_archive(archive: Path, folder: Path) -> Path:
    unpacked = folder / "unpacked"
    shutil.rmtree(unpacked, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpacked)
    try:
        package = next(unpacked.rglob("*.mlpackage"))
        return _compile(package, folder)
    finally:
        shutil.rmtree(unpacked, ignore_errors=True)


def ensure_model(
    model_path: str | Path | None = None,
    *,
    cache: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """The compiled `.mlmodelc` to load.

    `model_path` may be a compiled `.mlmodelc`, an `.mlpackage`, or the release's `.zip`; a package
    or zip is compiled into the cache. With no `model_path`, the pinned release is downloaded into
    the cache on first use (`progress` gets the download's fraction).
    """
    cache = cache or cache_dir()
    if model_path is not None:
        path = Path(model_path).expanduser()
        if path.suffix == ".mlmodelc" and path.is_dir():
            return path
        if path.suffix == ".mlpackage" and path.is_dir():
            key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
            folder = cache / f"local-{key}"
            folder.mkdir(parents=True, exist_ok=True)
            return _compile(path, folder)
        if path.suffix == ".zip" and path.is_file():
            folder = cache / f"zip-{sha256(path)[:16]}"
            compiled = folder / "scnet.mlmodelc"
            if compiled.is_dir() and (folder / "io.json").is_file():
                return compiled
            folder.mkdir(parents=True, exist_ok=True)
            return _from_archive(path, folder)
        raise FileNotFoundError(f"not a .mlmodelc, .mlpackage or .zip: {path}")

    folder = cache / f"scnet-coreml-{MODEL_VERSION}"
    compiled = folder / "scnet.mlmodelc"
    if compiled.is_dir() and (folder / "io.json").is_file():
        return compiled
    archive = folder / "scnet_coreml.mlpackage.zip"
    if not archive.is_file() or sha256(archive) != MODEL_SHA256:
        download(archive, progress=progress)
    return _from_archive(archive, folder)


def input_name(compiled: Path) -> str:
    io = compiled.parent / "io.json"
    if io.is_file():
        return json.loads(io.read_text())["input"]
    return DEFAULT_INPUT_NAME


def load(compiled: Path, device: str = "gpu"):
    """A `predict(spectrum) -> spectrum` function for the compiled model on `device`."""
    import coremltools as ct
    import numpy as np

    try:
        units = getattr(ct.ComputeUnit, COMPUTE_UNITS[device])
    except KeyError:
        raise ValueError(f"device must be one of {sorted(COMPUTE_UNITS)}; got {device!r}") from None
    model = ct.models.CompiledMLModel(str(compiled), compute_units=units)
    name = input_name(compiled)

    def predict(spectrum: np.ndarray) -> np.ndarray:
        outputs = model.predict({name: spectrum})
        if len(outputs) != 1:
            raise RuntimeError(f"expected one output from the model; got {sorted(outputs)}")
        return next(iter(outputs.values()))

    return predict
