#!/usr/bin/env python3
"""Prepare H.264 comparisons and audit files for sampled replacements."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from openve_research import prepare_random_pairs as common  # noqa: E402


def main() -> None:
    common.ROOT = ROOT
    common.MANIFEST_PATH = ROOT / "random_pairs.csv"
    common.EXAMPLES_DIR = ROOT / "examples"
    common.COMPARISON_DIR = common.EXAMPLES_DIR / "comparison"
    common.CONTACT_DIR = common.EXAMPLES_DIR / "contact_sheets"
    common.INVENTORY_PATH = ROOT / "general_replacement_inventory.csv"
    common.AUDIT_PATH = ROOT / "general_replacement_audit.csv"
    common.TASKS = ("replace",)
    common.EXPECTED_SAMPLE_SIZE = 10
    rows = common.collect_rows(rebuild=True)
    common.write_inventory(rows)
    written = common.write_audit(rows, reset=False)
    print(f"inventory_rows={len(rows)}")
    print(f"audit_status={'regenerated' if written else 'preserved'}")


if __name__ == "__main__":
    main()
