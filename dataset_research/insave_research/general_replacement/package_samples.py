#!/usr/bin/env python3
"""Package the reviewed replacement sample without raw duplicate videos."""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
INSAVE_ROOT = ROOT.parent
DESTINATION = ROOT / "insave_general_replacement_random_pairs_10_h264.zip"
ENTRIES = (
    "README.md",
    "random_pairs.csv",
    "general_replacement_inventory.csv",
    "general_replacement_audit.csv",
    "examples/comparison",
    "examples/contact_sheets",
)


def files_for(entry: str) -> list[Path]:
    path = ROOT / entry
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return [path]


def main() -> None:
    temporary = DESTINATION.with_suffix(".zip.part")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for entry in ENTRIES:
            for path in files_for(entry):
                if not path.is_file():
                    raise FileNotFoundError(path)
                archive.write(path, path.relative_to(ROOT))
        report = INSAVE_ROOT / "insave_report.md"
        if not report.is_file():
            raise FileNotFoundError(report)
        archive.write(report, "insave_report.md")
    temporary.replace(DESTINATION)
    print(f"wrote={DESTINATION.name} bytes={DESTINATION.stat().st_size}")


if __name__ == "__main__":
    main()
