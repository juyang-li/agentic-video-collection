#!/usr/bin/env python3
"""Compare continuous32 and 16+resume32 general-edit LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from safetensors.torch import load_file
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


EXPECTED_TARGET_MODULES = 252


def compare_values(
    left: Any,
    right: Any,
    path: str,
    report: dict[str, Any],
    atol: float,
    rtol: float,
) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        report["tensor_count"] += 1
        if left.shape != right.shape or left.dtype != right.dtype:
            report["mismatches"].append(
                f"{path}: {tuple(left.shape)}/{left.dtype} != "
                f"{tuple(right.shape)}/{right.dtype}"
            )
            return
        difference = 0.0
        if left.numel():
            difference = (left.float() - right.float()).abs().max().item()
            report["max_abs_difference"] = max(
                report["max_abs_difference"], float(difference)
            )
        if torch.equal(left, right):
            report["exact_tensor_count"] += 1
        elif not torch.allclose(left, right, atol=atol, rtol=rtol):
            report["mismatches"].append(
                f"{path}: tensor differs beyond tolerance (max_abs={difference})"
            )
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not np.array_equal(left, right):
            report["mismatches"].append(f"{path}: numpy array differs")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            report["mismatches"].append(f"{path}: mapping keys differ")
            return
        for key in sorted(left, key=str):
            compare_values(left[key], right[key], f"{path}.{key}", report, atol, rtol)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            report["mismatches"].append(f"{path}: sequence lengths differ")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(
                left_item, right_item, f"{path}[{index}]", report, atol, rtol
            )
        return
    if left != right:
        report["mismatches"].append(f"{path}: {left!r} != {right!r}")


def comparison(left: Any, right: Any, atol: float, rtol: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "tensor_count": 0,
        "exact_tensor_count": 0,
        "max_abs_difference": 0.0,
        "mismatches": [],
    }
    compare_values(left, right, "root", report, atol, rtol)
    report["within_tolerance"] = not report["mismatches"]
    return report


def load_optimizer(checkpoint: Path, destination: Path) -> Any:
    dcp_to_torch_save(checkpoint / "optimizer", destination)
    return torch.load(destination, map_location="cpu", weights_only=False)


def require_complete(checkpoint: Path) -> dict[str, Any]:
    marker = checkpoint / "checkpoint_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("world_size") != 4:
        raise ValueError(f"invalid checkpoint marker: {marker}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    markers = {
        label: require_complete(path)
        for label, path in (
            ("continuous", args.continuous),
            ("resumed", args.resumed),
            ("prefix", args.prefix),
        )
    }
    if markers["continuous"]["completed_steps"] != 32:
        raise ValueError("continuous checkpoint must be step 32")
    if markers["resumed"]["completed_steps"] != 32:
        raise ValueError("resumed checkpoint must be step 32")
    if markers["prefix"]["completed_steps"] != 16:
        raise ValueError("resume prefix checkpoint must be step 16")

    adapters = {
        label: load_file(str(path / "adapter.safetensors"), device="cpu")
        for label, path in (
            ("continuous", args.continuous),
            ("resumed", args.resumed),
            ("prefix", args.prefix),
        )
    }
    lora_b_names = sorted(
        name for name in adapters["prefix"] if name.endswith("lora_B")
    )
    nonzero_lora_b = sum(
        int(torch.count_nonzero(adapters["prefix"][name]).item() > 0)
        for name in lora_b_names
    )
    result: dict[str, Any] = {
        "schema_version": "lance_general_edit_resume_smoke_audit_v1",
        "continuous": str(args.continuous.resolve()),
        "resumed": str(args.resumed.resolve()),
        "prefix": str(args.prefix.resolve()),
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "adapter": comparison(
            adapters["continuous"], adapters["resumed"], args.atol, args.rtol
        ),
        "prefix_lora_B": {
            "tensor_count": len(lora_b_names),
            "nonzero_tensor_count": nonzero_lora_b,
        },
    }
    del adapters

    scheduler = {
        label: torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
        for label, path in (("continuous", args.continuous), ("resumed", args.resumed))
    }
    result["scheduler"] = comparison(
        scheduler["continuous"], scheduler["resumed"], args.atol, args.rtol
    )
    trainer = {
        label: json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        for label, path in (("continuous", args.continuous), ("resumed", args.resumed))
    }
    result["trainer_state"] = comparison(
        trainer["continuous"], trainer["resumed"], args.atol, args.rtol
    )
    rng = {
        label: {
            path.name: torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(checkpoint.glob("rng.rank-*.pt"))
        }
        for label, checkpoint in (
            ("continuous", args.continuous),
            ("resumed", args.resumed),
        )
    }
    result["rng"] = comparison(rng["continuous"], rng["resumed"], args.atol, args.rtol)

    with tempfile.TemporaryDirectory(prefix="general-edit-resume-audit-") as temp_dir:
        temporary = Path(temp_dir)
        optimizer = {
            "continuous": load_optimizer(args.continuous, temporary / "continuous.pt"),
            "resumed": load_optimizer(args.resumed, temporary / "resumed.pt"),
        }
        result["optimizer"] = comparison(
            optimizer["continuous"], optimizer["resumed"], args.atol, args.rtol
        )

    passed = (
        result["adapter"]["within_tolerance"]
        and result["optimizer"]["within_tolerance"]
        and result["scheduler"]["within_tolerance"]
        and result["trainer_state"]["within_tolerance"]
        and result["rng"]["within_tolerance"]
        and len(lora_b_names) == EXPECTED_TARGET_MODULES
        and nonzero_lora_b == EXPECTED_TARGET_MODULES
    )
    result["status"] = "passed" if passed else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
