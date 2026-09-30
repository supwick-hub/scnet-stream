"""`scnet-stream in.m4a out/`"""

from __future__ import annotations

import argparse
import sys
import time

_LABELS = {
    "download": "Downloading model",
    "load": "Loading model",
    "read": "Reading",
    "separate": "Separating",
}


class _Printer:
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.stage = None
        self.started = time.monotonic()

    def __call__(self, stage: str, fraction: float) -> None:
        if self.quiet:
            return
        if stage != self.stage:
            if self.stage is not None:
                sys.stderr.write("\n")
            self.stage = stage
            self.started = time.monotonic()
        elapsed = time.monotonic() - self.started
        sys.stderr.write(f"\r{_LABELS.get(stage, stage)}: {fraction:6.1%}  {elapsed:6.1f} s")
        sys.stderr.flush()

    def done(self) -> None:
        if not self.quiet and self.stage is not None:
            sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scnet-stream",
        description="Separate a track into vocals, drums, bass and other with SCNet on Core ML.",
    )
    parser.add_argument("input", help="any file ffmpeg can decode")
    parser.add_argument("out_dir", help="directory for vocals.wav, drums.wav, bass.wav, other.wav")
    parser.add_argument("--device", choices=["gpu", "ane", "cpu", "all"], default="gpu",
                        help="Core ML compute units (default: gpu). The first 'ane' load prepares "
                        "the model for the Neural Engine, which can take around twenty minutes.")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="overlap between 7.8 s segments (default: 0.25)")
    parser.add_argument("--float32", action="store_true",
                        help="write 32-bit float WAVs instead of dithered 16-bit")
    parser.add_argument("--model", dest="model_path",
                        help="a local .mlmodelc, .mlpackage or release .zip instead of the download")
    parser.add_argument("--no-malloc-large-cache", action="store_true",
                        help="macOS: re-exec with MallocLargeCache=0 so freed memory is returned")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.no_malloc_large_cache:
        from .macos import relaunch_without_large_malloc_cache

        relaunch_without_large_malloc_cache()

    from .pipeline import separate

    printer = _Printer(args.quiet)
    started = time.monotonic()
    try:
        paths = separate(
            args.input, args.out_dir, model_path=args.model_path, device=args.device,
            overlap=args.overlap, float32=args.float32, progress=printer,
        )  # fmt: skip
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        printer.done()
        print(f"scnet-stream: {e}", file=sys.stderr)
        return 1
    printer.done()
    if not args.quiet:
        for path in paths.values():
            print(path)
        print(f"done in {time.monotonic() - started:.1f} s", file=sys.stderr)
    return 0
