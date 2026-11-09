#!/usr/bin/env python3
"""Create a Bernini task file that excludes already completed outputs."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.inputs.read_text(encoding="utf-8"))
    pending = [task for task in tasks if not Path(task["output"]).is_file() or Path(task["output"]).stat().st_size == 0]
    args.output.write_text(json.dumps(pending, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(tasks), "pending": len(pending), "complete": len(tasks) - len(pending)}))


if __name__ == "__main__":
    main()
