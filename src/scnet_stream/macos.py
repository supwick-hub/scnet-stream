"""macOS memory helpers.

**The large-allocation cache.** macOS's malloc keeps freed large blocks in a per-process cache,
still dirty, and counts them in the process's footprint (what Activity Monitor shows). Allocating
and freeing arrays of 60, 65, ... 175 MB in turn left about 2.8 GB resident with nothing alive,
measured on macOS 26.2. A long-lived process that frees a big array per job therefore grows to the
sum of the distinct sizes it has ever freed, not to its peak.

`MallocLargeCache=0` in the environment turns the cache off. libmalloc reads it once, at process
start, so setting it from inside Python does nothing: the process has to be started with it, or
re-exec itself. scnet-stream's own arrays are bounded and small, so a single `separate` call does
not need this; a long-running host that also does whole-array work might.

Footprint, not RSS: memory Core ML and Metal hold on the GPU's behalf counts in `phys_footprint`
and not in `ps`'s RSS.
"""

from __future__ import annotations

import ctypes
import os
import sys

_RUSAGE_INFO_V4 = 4
# Offsets into rusage_info_v4, in 64-bit words after the 16-byte uuid.
_PHYS_FOOTPRINT = 7
_LIFETIME_MAX_PHYS_FOOTPRINT = 28


def relaunch_without_large_malloc_cache() -> None:
    """Re-exec this process with `MallocLargeCache=0`, once; a no-op elsewhere or if already set.

    Call it first thing in a script's entry point, before any heavy import: `os.execv` replaces the
    process (same pid) and starts it again from the top.
    """
    if sys.platform != "darwin" or os.environ.get("MallocLargeCache") == "0":
        return
    env = {**os.environ, "MallocLargeCache": "0"}
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def _rusage_word(index: int, pid: int | None) -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buf = ctypes.create_string_buffer(512)
        if libproc.proc_pid_rusage(ctypes.c_int(pid or os.getpid()), ctypes.c_int(_RUSAGE_INFO_V4), buf):
            return None
        start = 16 + 8 * index
        return int.from_bytes(buf.raw[start : start + 8], "little")
    except OSError:
        return None


def footprint_bytes(pid: int | None = None) -> int | None:
    """A process's current phys_footprint (this one by default); None off macOS."""
    return _rusage_word(_PHYS_FOOTPRINT, pid)


def peak_footprint_bytes(pid: int | None = None) -> int | None:
    """A process's lifetime peak phys_footprint; None off macOS."""
    return _rusage_word(_LIFETIME_MAX_PHYS_FOOTPRINT, pid)
