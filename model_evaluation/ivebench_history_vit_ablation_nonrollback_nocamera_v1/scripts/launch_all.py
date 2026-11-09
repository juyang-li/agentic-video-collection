#!/usr/bin/env python3
"""Run normal inference, verify it, then run all History-ViT ablations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
SCRIPTS = Path(__file__).resolve().parent


def main() -> int:
    environment = dict(os.environ)
    subprocess.run([str(PYTHON), str(SCRIPTS / "launch_normal.py")], env=environment, check=True)
    subprocess.run([str(PYTHON), str(SCRIPTS / "launch_ablation.py")], env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
