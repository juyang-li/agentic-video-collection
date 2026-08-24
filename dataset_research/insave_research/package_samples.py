#!/usr/bin/env python3
"""Package the two reviewed H.264 sample sets without external zip tools."""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
COMMON = ["README.md", "insave_report.md"]
PACKAGES = {
    "insave_remove_random_pairs_15_h264.zip": [
        *COMMON,
        "random_pairs.csv",
        "insave_inventory.csv",
        "insave_audit.csv",
        "insave_audio_inventory.csv",
        "examples/comparison",
    ],
    "insave_add_random_pairs_15_h264.zip": [
        *COMMON,
        "random_add_pairs.csv",
        "insave_add_inventory.csv",
        "insave_add_audit.csv",
        "insave_add_audio_inventory.csv",
        "examples_add/comparison",
    ],
}


def files_for(entry: str) -> list[Path]:
    path = ROOT / entry
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return [path]


def main() -> None:
    for archive_name, entries in PACKAGES.items():
        destination = ROOT / archive_name
        temporary = destination.with_suffix(".zip.part")
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for entry in entries:
                for path in files_for(entry):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    archive.write(path, path.relative_to(ROOT))
        temporary.replace(destination)
        print(f"wrote={destination.name} bytes={destination.stat().st_size}")


if __name__ == "__main__":
    main()
