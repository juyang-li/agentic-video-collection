#!/usr/bin/env python3
"""Create a config containing only missing Lance outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_outputs import probe_aligned_output


def quarantine_path(output: Path) -> Path:
    """Return a collision-free path next to an invalid generated output."""
    candidate = output.with_suffix(output.suffix + ".invalid")
    index = 1
    while candidate.exists():
        candidate = output.with_suffix(output.suffix + f".invalid.{index}")
        index += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("pending_config", type=Path)
    parser.add_argument(
        "--requeue-invalid",
        action="store_true",
        help="Quarantine an existing output that fails full validation and regenerate it.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pending = {}
    completed = 0
    for key, value in config.items():
        output = args.output_dir / f"{int(key):06d}.mp4"
        if output.exists():
            try:
                probe_aligned_output(output, value)
            except Exception as error:
                if not args.requeue_invalid:
                    raise
                quarantine = quarantine_path(output)
                output.replace(quarantine)
                pending[key] = value
                print(
                    f"requeued_invalid={output} quarantine={quarantine} "
                    f"error={type(error).__name__}: {error}"
                )
            else:
                completed += 1
                continue
        if not output.exists():
            pending[key] = value
    args.pending_config.parent.mkdir(parents=True, exist_ok=True)
    args.pending_config.write_text(
        json.dumps(pending, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"completed={completed} pending={len(pending)} config={args.pending_config}")


if __name__ == "__main__":
    main()
