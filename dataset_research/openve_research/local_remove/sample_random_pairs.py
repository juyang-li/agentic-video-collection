#!/usr/bin/env python3
"""Reproduce the OpenVE local_remove sample backed by local_add assets."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import sample_random_pairs as common  # noqa: E402


common.ROOT = ROOT
common.METADATA_PATH = ROOT.parent / "local_remove.csv"
common.MEMBERS_PATH = ROOT / "prefix_members.txt"
common.POPULATION_PATH = ROOT / "downloadable_population.csv"
common.OUTPUT_PATH = ROOT / "random_pairs.csv"
common.EXPECTED_METADATA_SHA256 = (
    "227694ee059bf198ff26b65359c36a03ab31903665b61d9814ba99553e80f5c3"
)
common.DEFAULT_PREFIX_PATH = Path("/tmp/openve_local_add_prefix_2g.tar.gz")
common.DEFAULT_SEED = 20260825
common.CATEGORY = "local_remove"
common.ASSET_PATH_PREFIX = "local_add"
common.ARCHIVE_DIRECTORY = "local_add"
common.SUBPOPULATION = "local_remove_pairs_in_local_add_assets_fixed_prefix"
common.SAMPLE_ID_PREFIX = "openve-remove"
common.TASK_OVERRIDE = "remove"


if __name__ == "__main__":
    common.main()
