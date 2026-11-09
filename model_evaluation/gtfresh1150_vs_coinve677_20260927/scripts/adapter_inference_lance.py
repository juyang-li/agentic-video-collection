#!/usr/bin/env python3
"""Run Lance single-turn inference with either legacy or PEFT DataMake LoRA."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def pop_argument(arguments: list[str], name: str) -> str:
    indexes = [index for index, value in enumerate(arguments) if value == name]
    if len(indexes) != 1:
        raise ValueError(f"{name} must be provided exactly once")
    index = indexes[0]
    if index + 1 >= len(arguments):
        raise ValueError(f"{name} requires a value")
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def main() -> None:
    arguments = list(sys.argv[1:])
    lance_root = Path(pop_argument(arguments, "--datamake_lance_root")).resolve()
    checkpoint = Path(pop_argument(arguments, "--datamake_lora_checkpoint")).resolve()
    sys.path.insert(0, str(lance_root))

    config = json.loads((checkpoint / "datamake_lora_config.json").read_text())
    schema = config["schema_version"]
    if schema == "datamake_lance_generation_lora_v1":
        from train.datamake_generation_lora import install_lora_adapter_inference_loader
        loader = install_lora_adapter_inference_loader
    elif schema == "datamake_lance_branch_lora_v1":
        from train.datamake_branch_lora import install_branch_adapter_inference_loader
        loader = install_branch_adapter_inference_loader
    elif schema == "datamake_lance_peft_branch_v1":
        sys.path.insert(0, str(lance_root.parent / "runtime/peft_0_15_2"))
        from train.datamake_peft import install_inference_loader
        loader = install_inference_loader
    else:
        raise ValueError(f"unsupported DataMake LoRA schema: {schema}")

    if not loader(str(checkpoint)):
        raise FileNotFoundError(f"incomplete adapter checkpoint: {checkpoint}")
    sys.argv = [sys.argv[0], *arguments]
    from inference_lance import main as inference_main
    inference_main()


if __name__ == "__main__":
    main()
