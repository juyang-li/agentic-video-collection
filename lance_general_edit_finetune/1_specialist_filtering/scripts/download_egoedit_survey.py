#!/usr/bin/env python3
"""Range-extract EgoEdit survey videos from the authorized remote ZIPs."""

from __future__ import annotations

import argparse
import binascii
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "dataset_research"))


def crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value = binascii.crc32(chunk, value)
    return value & 0xFFFFFFFF


def safe_output(relative: str) -> Path:
    output = (ROOT / relative).resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError(f"output escapes specialist root: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/survey_manifest.csv")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "dataset_research/egoedit_research/zip_members.csv",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    try:
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        from remote_zip_extract import Entry, extract_member
    except ImportError as error:
        raise SystemExit(
            "requests and huggingface_hub are required in the acquisition environment"
        ) from error

    class HfArchive:
        def __init__(self, filename: str, size: int) -> None:
            self.filename = filename
            self.size = size
            self.url = hf_hub_url(
                "liguang0115/EgoEdit",
                filename,
                repo_type="dataset",
                revision="a9bd3faafe51a581b359acd7c6ad5c432508ff60",
            )

        def read(self, offset: int, length: int) -> bytes:
            headers = build_hf_headers(token=True)
            headers["Range"] = f"bytes={offset}-{offset + length - 1}"
            response = requests.get(self.url, headers=headers, timeout=180)
            if response.status_code != 206 or len(response.content) != length:
                raise RuntimeError(
                    f"Range failed for {self.filename}: status={response.status_code} "
                    f"bytes={len(response.content)}/{length}"
                )
            return response.content

    entries: dict[str, tuple[object, object]] = {}
    archive_cache: dict[str, object] = {}
    with args.index.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            archive = archive_cache.setdefault(
                row["archive"], HfArchive(row["archive"], int(row["archive_size"]))
            )
            entries[row["basename"]] = (
                archive,
                Entry(
                    name=row["name"],
                    method=int(row["method"]),
                    flags=int(row["flags"]),
                    crc32=int(row["crc32"], 16),
                    compressed_size=int(row["compressed_size"]),
                    uncompressed_size=int(row["uncompressed_size"]),
                    local_offset=int(row["local_offset"]),
                ),
            )

    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["dataset"] == "egoedit"]
    wanted: dict[str, list[Path]] = {}
    for row in rows:
        for role in ("source", "target"):
            wanted.setdefault(row[f"{role}_locator"], []).append(
                safe_output(row[f"{role}_local"])
            )

    def download(basename: str, outputs: list[Path]) -> str:
        if basename not in entries:
            raise KeyError(f"EgoEdit member is not indexed: {basename}")
        archive, entry = entries[basename]
        output = next(
            (
                path
                for path in outputs
                if path.is_file() and path.stat().st_size == entry.uncompressed_size
            ),
            outputs[0],
        )
        if (
            output.is_file()
            and output.stat().st_size == entry.uncompressed_size
            and crc32_file(output) == entry.crc32
        ):
            action = "preserved"
        else:
            extract_member(archive, entry, output)
            action = "downloaded"
        for duplicate in outputs:
            if duplicate == output:
                continue
            if duplicate.is_file() and duplicate.stat().st_size == entry.uncompressed_size:
                continue
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            temporary = duplicate.with_suffix(duplicate.suffix + ".part")
            shutil.copyfile(output, temporary)
            temporary.replace(duplicate)
        return f"{action} {basename} outputs={len(outputs)}"

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, name, paths) for name, paths in wanted.items()]
        for future in as_completed(futures):
            completed += 1
            print(f"[{completed}/{len(wanted)}] {future.result()}", flush=True)
    print(f"done unique_videos={len(wanted)}")


if __name__ == "__main__":
    main()
