#!/usr/bin/env python3
"""Atomically update the ablation queue status file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--state", required=True)
    parser.add_argument("--active-run", default="")
    parser.add_argument("--completed-run", action="append", default=[])
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    current = {}
    if args.path.exists():
        current = json.loads(args.path.read_text(encoding="utf-8"))
    completed = list(current.get("completed_runs", []))
    for run in args.completed_run:
        if run and run not in completed:
            completed.append(run)
    payload = {
        "state": args.state,
        "active_run": args.active_run or None,
        "completed_runs": completed,
        "message": args.message or None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.path.with_suffix(args.path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.path)


if __name__ == "__main__":
    main()
