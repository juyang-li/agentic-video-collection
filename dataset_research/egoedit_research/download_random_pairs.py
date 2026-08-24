#!/usr/bin/env python3
"""Range-extract only the sampled EgoEdit MP4s from authorized HF ZIPs."""

from __future__ import annotations

import binascii
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers
import requests


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from remote_zip_extract import Entry, extract_member  # noqa: E402


MANIFEST = ROOT / "random_replace_pairs.csv"
INDEX = ROOT / "zip_members.csv"
REPO_ID = "liguang0115/EgoEdit"
REVISION = "a9bd3faafe51a581b359acd7c6ad5c432508ff60"
WORKERS = 4


class HfArchive:
    def __init__(self, filename: str, size: int) -> None:
        self.filename = filename
        self.size = size
        self.url = hf_hub_url(
            REPO_ID, filename, repo_type="dataset", revision=REVISION
        )

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("requested range falls outside archive")
        headers = build_hf_headers(token=True)
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        response = requests.get(self.url, headers=headers, timeout=180)
        if response.status_code != 206:
            raise RuntimeError(
                f"Range failed for {self.filename}: HTTP {response.status_code}"
            )
        if len(response.content) != length:
            raise RuntimeError(
                f"short Range for {self.filename}: "
                f"{len(response.content)} != {length}"
            )
        return response.content


def crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value = binascii.crc32(chunk, value)
    return value & 0xFFFFFFFF


def load_index() -> dict[str, tuple[HfArchive, Entry]]:
    result: dict[str, tuple[HfArchive, Entry]] = {}
    archives: dict[str, HfArchive] = {}
    with INDEX.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            archive_name = row["archive"]
            archive = archives.setdefault(
                archive_name,
                HfArchive(archive_name, int(row["archive_size"])),
            )
            basename = row["basename"]
            if basename in result:
                raise ValueError(f"duplicate member basename: {basename}")
            result[basename] = (
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
    return result


def main() -> None:
    index = load_index()
    with MANIFEST.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    wanted: list[tuple[str, str, str, Path, HfArchive, Entry]] = []
    for row in rows:
        for role in ("source", "target"):
            basename = row[f"{role}_member_basename"]
            if basename not in index:
                raise KeyError(f"member is not indexed: {basename}")
            archive, entry = index[basename]
            output = ROOT / row[f"{role}_file"]
            wanted.append((row["sample_id"], role, basename, output, archive, entry))

    def download(item: tuple[str, str, str, Path, HfArchive, Entry]) -> str:
        sample_id, role, basename, output, archive, entry = item
        if (
            output.is_file()
            and output.stat().st_size == entry.uncompressed_size
            and crc32_file(output) == entry.crc32
        ):
            return f"preserved {sample_id}:{role} {basename}"
        extract_member(archive, entry, output)
        return f"downloaded {sample_id}:{role} {basename}"

    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(download, item) for item in wanted]
        for future in as_completed(futures):
            completed += 1
            print(f"[{completed}/{len(wanted)}] {future.result()}", flush=True)
    print(
        f"done roles={len(wanted)} unique_members="
        f"{len({item[2] for item in wanted})}"
    )


if __name__ == "__main__":
    main()
