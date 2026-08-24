#!/usr/bin/env python3
"""Reproduce the GOKU remove part32 random sample."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import sample_random_pairs as common  # noqa: E402


common.ROOT = ROOT
common.DEFAULT_METADATA_TAR = Path("/tmp/goku_remove_acquire/remove/remove.jsons.tar")
common.DEFAULT_VIDEO_TAR = Path(
    "/tmp/goku_remove_acquire/remove/remove.videos.part32.tar"
)
common.MEMBERS_PATH = ROOT / "shard_members.txt"
common.POPULATION_PATH = ROOT / "shard_population.csv"
common.OUTPUT_PATH = ROOT / "random_pairs.csv"
common.TASK_CATEGORY = "remove"
common.SUBPOPULATION = "remove_part32_complete_cases"
common.SAMPLE_ID_PREFIX = "goku-remove"
common.SHARD_SEED = 20260823
common.PAIR_SEED = 20260823
common.EXPECTED_METADATA_SHA256 = (
    "8f66847becc20c2912f51e392693d777f71b9776dde00f6e15721d9d70266e0c"
)
common.EXPECTED_VIDEO_SHA256 = (
    "67a6cf4c932106a766b505d9207a95e9e395742c614a24d9d5b8a2bdb29d0906"
)


if __name__ == "__main__":
    common.main()
