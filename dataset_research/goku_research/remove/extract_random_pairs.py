#!/usr/bin/env python3
"""Extract the selected GOKU remove roles."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import extract_random_pairs as common  # noqa: E402


common.ROOT = ROOT
common.MANIFEST_PATH = ROOT / "random_pairs.csv"
common.DEFAULT_VIDEO_TAR = Path(
    "/tmp/goku_remove_acquire/remove/remove.videos.part32.tar"
)


if __name__ == "__main__":
    common.main()
