#!/usr/bin/env python3
"""Prepare GOKU comparisons using the shared OpenVE review renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from openve_research import prepare_random_pairs as common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reset-audit", action="store_true")
    args = parser.parse_args()
    common.ROOT = ROOT
    common.MANIFEST_PATH = ROOT / "random_pairs.csv"
    common.EXAMPLES_DIR = ROOT / "examples"
    common.COMPARISON_DIR = common.EXAMPLES_DIR / "comparison"
    common.CONTACT_DIR = common.EXAMPLES_DIR / "contact_sheets"
    common.INVENTORY_PATH = ROOT / "goku_inventory.csv"
    common.AUDIT_PATH = ROOT / "goku_audit.csv"
    common.TASKS = ("alter", "swap", "other")
    rows = common.collect_rows(args.rebuild)
    common.write_inventory(rows)
    written = common.write_audit(rows, args.reset_audit)
    print(f"inventory_rows={len(rows)}")
    print(f"audit_status={'regenerated' if written else 'preserved'}")


if __name__ == "__main__":
    main()
