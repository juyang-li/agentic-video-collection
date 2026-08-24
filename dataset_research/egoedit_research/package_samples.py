#!/usr/bin/env python3
"""Package the EgoEdit H.264 sample set without external zip tools."""

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "egoedit_replace_random_pairs_15_h264.zip"
ENTRIES = [
    "README.md",
    "egoedit_report.md",
    "random_replace_pairs.csv",
    "egoedit_inventory.csv",
    "egoedit_audit.csv",
    "examples/comparison",
]


def files_for(entry: str) -> list[Path]:
    path = ROOT / entry
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return [path]


def main() -> None:
    temporary = OUTPUT.with_suffix(".zip.part")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for entry in ENTRIES:
            for path in files_for(entry):
                if not path.is_file():
                    raise FileNotFoundError(path)
                archive.write(path, path.relative_to(ROOT))
    temporary.replace(OUTPUT)
    print(f"wrote={OUTPUT.name} bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
