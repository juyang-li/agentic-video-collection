#!/usr/bin/env python3
"""Score provenance-routed edit pairs with the local VEFX-Reward model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_ROOT = PROJECT_ROOT / "1_specialist_filtering"
VEFX_REPO = ROOT / "vendor/VEFX-Bench"
DEFAULT_INPUT = ROOT / "artifacts/model_bakeoff_v1/current_scope_63.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/model_bakeoff_v1/vefx_4b_scores.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/model_bakeoff_v1/vefx_4b_summary.json"


FIELDS = [
    "model_id",
    "sample_id",
    "dataset",
    "branch",
    "operation",
    "human_status",
    "cohort",
    "source_provenance",
    "target_provenance",
    "generated_side",
    "audit_relation",
    "reference_training_role",
    "generated_training_role",
    "audit_instruction",
    "instruction_transform",
    "IF",
    "RQ",
    "EE",
    "Overall",
    "status",
    "error",
    "elapsed_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_media(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SPECIALIST_ROOT / path


def strip_leading_command(instruction: str, operation: str) -> str:
    commands = {
        "add": r"add|insert|place|put",
        "remove": r"remove|delete|erase",
    }
    command = commands[operation]
    value = re.sub(
        rf"^\s*(?:please\s+|could\s+you\s+|can\s+you\s+)?(?:{command})\b\s*",
        "",
        instruction.strip(),
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return value or instruction.strip()


def invert_replace_instruction(instruction: str) -> tuple[str, str]:
    """Invert known replacement templates without guessing the target noun phrase.

    EgoEdit descriptions often append long appearance and motion clauses to the
    object after ``with``/``into``.  Trying to swap both free-form noun phrases can
    therefore turn those clauses into a misleading review instruction.  In the
    inverse direction the current reference object is already visible; naming the
    original source object is sufficient and keeps the instruction unambiguous.
    """
    value = instruction.strip()
    patterns = (
        (
            r"^\s*(?:(?:seamlessly|gently)\s+)?"
            r"(?:replace|transform|turn|morph|convert)\s+(.+?)\s+"
            r"(?:with|into)\s+.+$",
            "inverse_replace_source_object_only",
        ),
        (
            r"^\s*(?:(?:seamlessly|gently)\s+)?"
            r"swap(?:\s+out)?\s+(.+?)\s+for\s+.+$",
            "inverse_swap_source_object_only",
        ),
    )
    for pattern, transform in patterns:
        match = re.match(pattern, value, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        source_object = match.group(1).strip().rstrip(" ,.;:!?")
        if not source_object or len(source_object) > 240:
            break
        return (
            "Replace the current reference object with the original source "
            f"object: {source_object}.",
            transform,
        )
    raise ValueError(f"cannot safely invert replacement instruction: {instruction!r}")


def route(row: dict[str, str]) -> dict[str, str]:
    source_provenance = row.get("source_provenance", "").strip().lower()
    target_provenance = row.get("target_provenance", "").strip().lower()
    operation = row.get("operation", "").strip().lower()
    instruction = row.get("instruction", "").strip()
    if (
        source_provenance == "generation_side_unresolved"
        and target_provenance == "generation_side_unresolved"
    ):
        return {
            "generated_side": "UNRESOLVED",
            "audit_relation": "training_direction_generation_side_unresolved",
            "reference_training_role": "",
            "generated_training_role": "",
            "audit_left_training_role": "source",
            "audit_right_training_role": "target",
            "artifact_checked_roles": "source+target",
            "instruction_following_direction": "source_to_target",
            "content_preservation_direction": "source_to_target",
            "audit_instruction": instruction,
            "instruction_transform": "identity_generation_side_unresolved",
        }
    if source_provenance == "original" and target_provenance == "edited":
        return {
            "generated_side": "TARGET",
            "audit_relation": "forward",
            "reference_training_role": "source",
            "generated_training_role": "target",
            "audit_left_training_role": "source",
            "audit_right_training_role": "target",
            "artifact_checked_roles": "target",
            "instruction_following_direction": "reference_to_generated",
            "content_preservation_direction": "reference_to_generated",
            "audit_instruction": instruction,
            "instruction_transform": "identity",
        }
    if source_provenance == "edited" and target_provenance == "original":
        if operation in {"add", "remove"}:
            inverse = "remove" if operation == "add" else "add"
            description = strip_leading_command(instruction, operation)
            audit_instruction = f"{inverse.capitalize()} {description}"
            transform = f"inverse_{operation}_to_{inverse}"
        elif operation == "replace":
            audit_instruction, transform = invert_replace_instruction(instruction)
        else:
            raise ValueError(f"unsupported operation: {operation!r}")
        return {
            "generated_side": "SOURCE",
            "audit_relation": "inverse",
            "reference_training_role": "target",
            "generated_training_role": "source",
            "audit_left_training_role": "target",
            "audit_right_training_role": "source",
            "artifact_checked_roles": "source",
            "instruction_following_direction": "reference_to_generated",
            "content_preservation_direction": "reference_to_generated",
            "audit_instruction": audit_instruction,
            "instruction_transform": transform,
        }
    if source_provenance == "edited" and target_provenance == "edited":
        return {
            "generated_side": "BOTH",
            "audit_relation": "training_direction_both_generated",
            "reference_training_role": "source",
            "generated_training_role": "target",
            "audit_left_training_role": "source",
            "audit_right_training_role": "target",
            "artifact_checked_roles": "source+target",
            "instruction_following_direction": "source_to_target",
            "content_preservation_direction": "source_to_target",
            "audit_instruction": instruction,
            "instruction_transform": "identity_both_generated",
        }
    raise ValueError(
        f"unsupported provenance source={source_provenance!r}, "
        f"target={target_provenance!r}"
    )


def append_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def previous_successes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {row["sample_id"] for row in read_csv(path) if row.get("status") == "pass"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default="xiangbog/VEFX-Reward-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-frame-pixels", type=int, default=399360)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()
    elif args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume or --overwrite: {args.output}")
    rows = read_csv(args.input)
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row["sample_id"] in wanted]
        missing = wanted - {row["sample_id"] for row in rows}
        if missing:
            raise ValueError(f"missing sample ids: {sorted(missing)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    succeeded = previous_successes(args.output) if args.resume else set()
    rows = [row for row in rows if row["sample_id"] not in succeeded]

    sys.path.insert(0, str(VEFX_REPO))
    import torch
    from vefx_reward import VEFXReward

    model = VEFXReward(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        fps=args.fps,
        max_frame_pixels=args.max_frame_pixels,
    )
    run_started = time.monotonic()
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        result = {field: "" for field in FIELDS}
        result.update({
            "model_id": args.model,
            "sample_id": row["sample_id"],
            "dataset": row.get("dataset", ""),
            "branch": row.get("branch", ""),
            "operation": row.get("operation", ""),
            "human_status": row.get("human_status", ""),
            "cohort": row.get("cohort", ""),
            "source_provenance": row.get("source_provenance", ""),
            "target_provenance": row.get("target_provenance", ""),
        })
        try:
            audit = route(row)
            result.update(audit)
            reference = resolve_media(row[f"{audit['reference_training_role']}_local"])
            generated = resolve_media(row[f"{audit['generated_training_role']}_local"])
            if not reference.is_file() or not generated.is_file():
                raise FileNotFoundError(f"missing media: {reference}, {generated}")
            scores = model.score(
                original_video=str(reference),
                edited_video=str(generated),
                instruction=audit["audit_instruction"],
            )
            result.update({key: str(scores[key]) for key in ("IF", "RQ", "EE", "Overall")})
            result["status"] = "pass"
        except Exception as error:
            result["status"] = "error"
            result["error"] = f"{type(error).__name__}: {error}"[:2000]
        result["elapsed_seconds"] = f"{time.monotonic() - started:.6f}"
        append_csv(args.output, result)
        print(
            f"vefx={index}/{len(rows)} sample_id={row['sample_id']} "
            f"status={result['status']} IF={result['IF']} RQ={result['RQ']} "
            f"EE={result['EE']} seconds={result['elapsed_seconds']}",
            flush=True,
        )

    all_rows = read_csv(args.output)
    summary = {
        "model_id": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(all_rows),
        "status_counts": dict(Counter(row["status"] for row in all_rows)),
        "route_counts": dict(Counter(row["audit_relation"] for row in all_rows)),
        "fps": args.fps,
        "max_frame_pixels": args.max_frame_pixels,
        "new_run_wall_seconds": time.monotonic() - run_started,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
