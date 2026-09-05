#!/usr/bin/env python3
"""Verify representative trainable updates and exact frozen-parameter stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


TRAINABLE_KEYS = [
    "language_model.model.layers.0.self_attn.q_norm_moe_gen.weight",
    "language_model.model.layers.0.self_attn.q_proj_moe_gen.bias",
    "vae2llm.bias",
    "llm2vae.bias",
]
FROZEN_KEYS = [
    "language_model.model.layers.0.self_attn.q_norm.weight",
    "vit_model.blocks.0.norm1.weight",
]


def allowed_trainable(name: str) -> bool:
    normalized = name.replace("._fsdp_wrapped_module", "").replace(
        "._checkpoint_wrapped_module", ""
    )
    return (
        "_moe_gen" in normalized
        or normalized.startswith("language_model.model.norm_moe_gen")
        or normalized.startswith("time_embedder.")
        or normalized.startswith("vae2llm.")
        or normalized.startswith("llm2vae.")
    )


def compare(base_path: Path, final_path: Path, key: str) -> dict:
    with safe_open(base_path, framework="pt", device="cpu") as base_file:
        base = base_file.get_tensor(key)
    with safe_open(final_path, framework="pt", device="cpu") as final_file:
        final = final_file.get_tensor(key)
    if base.shape != final.shape:
        raise ValueError(f"shape changed for {key}: {base.shape} -> {final.shape}")
    delta = (base.float() - final.float()).abs()
    return {
        "base_dtype": str(base.dtype),
        "final_dtype": str(final.dtype),
        "elements": base.numel(),
        "changed_elements": int((base.float() != final.float()).sum()),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--trainable-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trainable_names = [
        line.removesuffix(": True")
        for line in args.trainable_report.read_text(encoding="utf-8").splitlines()
        if line.endswith(": True")
    ]
    if len(trainable_names) != 513:
        raise ValueError(f"expected 513 trainable tensors, found {len(trainable_names)}")
    unexpected = [name for name in trainable_names if not allowed_trainable(name)]
    if unexpected:
        raise ValueError(f"unexpected trainable tensors: {unexpected[:10]}")

    comparisons = {
        key: compare(args.base, args.final, key)
        for key in [*TRAINABLE_KEYS, *FROZEN_KEYS]
    }
    for key in TRAINABLE_KEYS:
        if comparisons[key]["changed_elements"] == 0:
            raise ValueError(f"trainable tensor did not update: {key}")
    for key in FROZEN_KEYS:
        if comparisons[key]["changed_elements"] != 0:
            raise ValueError(f"frozen tensor changed: {key}")

    report = {
        "status": "ok",
        "trainable_tensor_count": len(trainable_names),
        "trainable_scope": "generation MoE + time_embedder + vae2llm + llm2vae",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
