#!/usr/bin/env python3
"""Prefetch the two implicit official IVEBench models into the project cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import clip
import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "models" / "builtin"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cache = CACHE.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    clip_root = cache / "clip"
    torchhub_root = cache / "torchhub"
    clip_root.mkdir(exist_ok=True)
    torchhub_root.mkdir(exist_ok=True)
    clip_model, _ = clip.load("ViT-B/32", device="cpu", download_root=str(clip_root))
    del clip_model
    torch.hub.set_dir(str(torchhub_root))
    dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=True)
    del dino_model

    clip_files = sorted(path for path in clip_root.iterdir() if path.is_file())
    dino_files = sorted(path for path in torchhub_root.rglob("*.pth") if path.is_file())
    if len(clip_files) != 1 or not dino_files:
        raise RuntimeError("implicit model cache is incomplete")
    result = {
        "clip": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in clip_files],
        "dino": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in dino_files],
        "torchhub_dir": str(torchhub_root),
    }
    output = cache / "prefetch_manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
