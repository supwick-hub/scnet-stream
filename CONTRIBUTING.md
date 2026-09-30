# Contributing

Thanks for looking. This is maintained on a best-effort basis: issues and pull requests are
welcome, and replies may take a while.

## Setting up

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
```

The tests need ffmpeg on `PATH` and do not need the model: a pass-through network stands in for
Core ML. `tests/test_real_model.py` runs only when the model is already present (set
`SCNET_STREAM_MODEL`, or run the CLI once so it is cached); `tests/test_spectral.py` checks the
NumPy STFT against PyTorch only when torch is installed.

## Ground rules

- **Memory stays flat.** Nothing may hold an array the length of the track. `tests/test_memory.py`
  guards this; please keep it passing.
- **Output stays the same.** A change to the chunking or overlap-add should keep
  `tests/test_core.py` bit-identical to the whole-array loop, or say clearly why not.
- **No real audio in the repository.** Tests and benchmarks generate their signals.
- Keep dependencies to NumPy and coremltools.
