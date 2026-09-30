"""Peak memory does not grow with the track's length.

Runs in a fresh process per length, with a pass-through network, on synthetic audio generated
block by block (nothing the track's length exists anywhere). On macOS it measures the lifetime
peak phys_footprint with the malloc large cache off, so the figure is live memory; elsewhere,
ru_maxrss.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    import numpy as np
    from scnet_stream.core import SOURCES
    from scnet_stream.pipeline import separate_source
    from scnet_stream.macos import peak_footprint_bytes

    minutes, out = float(sys.argv[1]), sys.argv[2]

    class Synthetic:
        block = 1 << 18
        def blocks(self):
            total = int(minutes * 60 * 44100)
            for start in range(0, total, self.block):
                t = np.arange(start, min(start + self.block, total)) / 44100
                x = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 440.5 * t)
                yield np.stack([x, 0.8 * x], axis=1).astype(np.float32)

    def predict(spectrum):
        return np.repeat(spectrum[:, None], len(SOURCES), axis=1)

    separate_source(Synthetic(), out, predict)
    peak = peak_footprint_bytes()
    if peak is None:
        scale = 1 if sys.platform == "darwin" else 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    print(json.dumps({"peak": peak}))
    """
)


def _peak(minutes: float, out) -> int:
    env = {**os.environ, "MallocLargeCache": "0"}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(minutes), str(out)],
        capture_output=True, text=True, env=env, check=True,
    )  # fmt: skip
    return json.loads(result.stdout.strip().splitlines()[-1])["peak"]


@pytest.mark.slow
def test_peak_memory_is_flat_in_length(tmp_path):
    one = _peak(1, tmp_path / "one")
    eight = _peak(8, tmp_path / "eight")
    mb = 1 << 20
    print(f"peak: 1 min {one / mb:.0f} MB, 8 min {eight / mb:.0f} MB")
    # 8 minutes of stereo float32 alone is 169 MB, and a (4, 2, n) accumulator 677 MB: a runner
    # that held either would blow through this.
    assert eight - one < 60 * mb
