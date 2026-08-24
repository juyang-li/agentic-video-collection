#!/usr/bin/env python3
"""Reproduce the OpenVE local_add fixed-prefix random sample."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import sample_random_pairs as common  # noqa: E402


common.ROOT = ROOT
common.METADATA_PATH = ROOT.parent / "local_add.csv"
common.MEMBERS_PATH = ROOT / "prefix_members.txt"
common.POPULATION_PATH = ROOT / "downloadable_population.csv"
common.OUTPUT_PATH = ROOT / "random_pairs.csv"
common.EXPECTED_METADATA_SHA256 = (
    "8ea018c7b4d8559a503ca341a5d150b77cf407f657149ced4c9b7593fac387eb"
)
common.DEFAULT_PREFIX_PATH = Path("/tmp/openve_local_add_prefix_2g.tar.gz")
common.DEFAULT_SEED = 20260824
common.CATEGORY = "local_add"
common.ASSET_PATH_PREFIX = "local_add"
common.ARCHIVE_DIRECTORY = "local_add"
common.SUBPOPULATION = "local_add_assets_fixed_prefix"
common.SAMPLE_ID_PREFIX = "openve-add"
common.TASK_OVERRIDE = "add"


if __name__ == "__main__":
    common.main()
