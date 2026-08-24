#!/usr/bin/env python3
"""Fetch only EgoEdit ZIP central directories and build a member index."""

from __future__ import annotations

import csv
from pathlib import Path
import struct
import sys

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers
import requests


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from remote_zip_extract import parse_central_directory  # noqa: E402


REPO_ID = "liguang0115/EgoEdit"
REVISION = "a9bd3faafe51a581b359acd7c6ad5c432508ff60"
INDEX_DIR = ROOT / "zip_indexes"
OUTPUT = ROOT / "zip_members.csv"
ARCHIVES = {
    "videos/videos_001.zip": 32_461_006_641,
    "videos/videos_002.zip": 32_504_621_669,
    "videos/videos_003.zip": 33_853_775_397,
    "videos/videos_004.zip": 32_015_482_184,
    "videos/videos_005.zip": 33_053_028_680,
    "videos/videos_006.zip": 32_600_501_741,
    "videos/videos_007.zip": 33_220_309_780,
    "videos/videos_008.zip": 15_306_228_921,
}
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"


def remote_range(filename: str, start: int, end: int) -> bytes:
    url = hf_hub_url(
        REPO_ID, filename, repo_type="dataset", revision=REVISION
    )
    headers = build_hf_headers(token=True)
    headers["Range"] = f"bytes={start}-{end}"
    response = requests.get(url, headers=headers, timeout=120)
    if response.status_code != 206:
        raise RuntimeError(
            f"Range failed for {filename}: HTTP {response.status_code}"
        )
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"short Range for {filename}: {len(response.content)} != {expected}"
        )
    return response.content


def central_directory(filename: str, archive_size: int) -> tuple[bytes, int, int]:
    tail_size = min(128 * 1024, archive_size)
    tail_start = archive_size - tail_size
    tail = remote_range(filename, tail_start, archive_size - 1)
    eocd_pos = tail.rfind(EOCD_SIGNATURE)
    if eocd_pos < 0 or eocd_pos + 22 > len(tail):
        raise ValueError(f"EOCD missing in {filename}")
    fields = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
    entry_count = fields[4]
    cd_size = fields[5]
    cd_offset = fields[6]
    if entry_count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator_pos = tail.rfind(ZIP64_LOCATOR_SIGNATURE, 0, eocd_pos)
        if locator_pos < 0 or locator_pos + 20 > len(tail):
            raise ValueError(f"ZIP64 locator missing in {filename}")
        locator = struct.unpack_from("<4sLQL", tail, locator_pos)
        zip64_offset = locator[2]
        record = remote_range(filename, zip64_offset, zip64_offset + 55)
        values = struct.unpack_from("<4sQ2H2L4Q", record)
        if values[0] != ZIP64_EOCD_SIGNATURE:
            raise ValueError(f"bad ZIP64 EOCD in {filename}")
        entry_count = values[7]
        cd_size = values[8]
        cd_offset = values[9]
    data = remote_range(filename, cd_offset, cd_offset + cd_size - 1)
    return data, int(entry_count), int(cd_offset)


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for archive_rank, (filename, archive_size) in enumerate(ARCHIVES.items(), 1):
        central, expected_entries, cd_offset = central_directory(
            filename, archive_size
        )
        central_path = INDEX_DIR / f"videos_{archive_rank:03d}.central.bin"
        central_path.write_bytes(central)
        entries = parse_central_directory(central_path)
        if len(entries) != expected_entries:
            raise ValueError(
                f"entry mismatch for {filename}: {len(entries)} != {expected_entries}"
            )
        for entry in entries:
            if entry.name.lower().endswith(".mp4"):
                rows.append(
                    {
                        "archive": filename,
                        "archive_size": archive_size,
                        "central_offset": cd_offset,
                        "name": entry.name,
                        "basename": Path(entry.name).name,
                        "method": entry.method,
                        "flags": entry.flags,
                        "crc32": f"{entry.crc32:08x}",
                        "compressed_size": entry.compressed_size,
                        "uncompressed_size": entry.uncompressed_size,
                        "local_offset": entry.local_offset,
                    }
                )
        print(
            f"[{archive_rank}/{len(ARCHIVES)}] {filename} "
            f"entries={len(entries)} central_bytes={len(central)}",
            flush=True,
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unique = len({row["basename"] for row in rows})
    print(f"mp4_members={len(rows)} unique_basenames={unique}")


if __name__ == "__main__":
    main()
