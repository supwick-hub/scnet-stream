# Changelog

## 0.1.0 — unreleased

First version.

- `separate()` and the `scnet-stream` command: four stems from any file ffmpeg can decode, with
  memory that does not grow with the track's length.
- The STFT pair in NumPy; no PyTorch needed at runtime.
- The pinned Core ML model from scnet-executorch v0.1.2, downloaded on first use and checked
  against its SHA-256; or a local `.mlmodelc`, `.mlpackage` or release `.zip`.
- 16-bit WAV with TPDF dither (deterministic), or 32-bit float.
- `scnet_stream.macos`: an opt-in re-exec with `MallocLargeCache=0`, and footprint readers.
