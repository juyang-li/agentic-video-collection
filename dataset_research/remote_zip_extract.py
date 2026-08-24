#!/usr/bin/env python3
"""List and selectively extract members from very large remote ZIP archives.

The central directory is supplied as a small local range download.  Archive
parts are described by a JSON list of {"url": ..., "size": ...} objects.
Only byte ranges needed by the requested members are downloaded.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import json
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


@dataclass(frozen=True)
class Entry:
    name: str
    method: int
    flags: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def parse_zip64_extra(extra: bytes, values: list[tuple[str, int]]) -> dict[str, int]:
    pos = 0
    while pos + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4 : pos + 4 + size]
        pos += 4 + size
        if field_id != 0x0001:
            continue
        result: dict[str, int] = {}
        body_pos = 0
        for key, width in values:
            if body_pos + width > len(body):
                raise ValueError("truncated ZIP64 extra field")
            fmt = "<Q" if width == 8 else "<I"
            result[key] = struct.unpack_from(fmt, body, body_pos)[0]
            body_pos += width
        return result
    raise ValueError("ZIP64 extra field is missing")


def parse_central_directory(path: Path) -> list[Entry]:
    data = path.read_bytes()
    entries: list[Entry] = []
    pos = 0
    while data[pos : pos + 4] == CENTRAL_SIGNATURE:
        if pos + 46 > len(data):
            raise ValueError("truncated central directory header")
        values = struct.unpack_from("<4s6H3I5H2I", data, pos)
        flags = values[3]
        method = values[4]
        crc32 = values[7]
        compressed_size = values[8]
        uncompressed_size = values[9]
        name_len, extra_len, comment_len = values[10:13]
        disk_start = values[13]
        local_offset = values[16]
        name_start = pos + 46
        name_bytes = data[name_start : name_start + name_len]
        extra_start = name_start + name_len
        extra = data[extra_start : extra_start + extra_len]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)

        zip64_fields: list[tuple[str, int]] = []
        if uncompressed_size == 0xFFFFFFFF:
            zip64_fields.append(("uncompressed_size", 8))
        if compressed_size == 0xFFFFFFFF:
            zip64_fields.append(("compressed_size", 8))
        if local_offset == 0xFFFFFFFF:
            zip64_fields.append(("local_offset", 8))
        if disk_start == 0xFFFF:
            zip64_fields.append(("disk_start", 4))
        if zip64_fields:
            zip64 = parse_zip64_extra(extra, zip64_fields)
            uncompressed_size = zip64.get("uncompressed_size", uncompressed_size)
            compressed_size = zip64.get("compressed_size", compressed_size)
            local_offset = zip64.get("local_offset", local_offset)

        entries.append(
            Entry(
                name=name,
                method=method,
                flags=flags,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
            )
        )
        pos = extra_start + extra_len + comment_len
    if not entries:
        raise ValueError("no central directory entries found")
    return entries


class RemoteArchive:
    def __init__(self, parts: list[dict[str, object]]) -> None:
        self.parts = parts
        self.starts: list[int] = []
        offset = 0
        for part in parts:
            self.starts.append(offset)
            offset += int(part["size"])
        self.size = offset

    @staticmethod
    def _range(url: str, start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError(f"server ignored Range for {url}: HTTP {response.status}")
            return response.read()

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("requested range falls outside archive")
        chunks: list[bytes] = []
        remaining = length
        cursor = offset
        for index, part_start in enumerate(self.starts):
            part_size = int(self.parts[index]["size"])
            part_end = part_start + part_size
            if cursor >= part_end or remaining == 0:
                continue
            local_start = max(cursor, part_start) - part_start
            take = min(remaining, part_size - local_start)
            chunks.append(
                self._range(str(self.parts[index]["url"]), local_start, local_start + take - 1)
            )
            cursor += take
            remaining -= take
        if remaining:
            raise RuntimeError(f"short remote read: {remaining} bytes missing")
        return b"".join(chunks)


def extract_member(archive: RemoteArchive, entry: Entry, output: Path) -> None:
    header = archive.read(entry.local_offset, 30)
    if header[:4] != LOCAL_SIGNATURE:
        raise ValueError(f"bad local header for {entry.name}")
    values = struct.unpack("<4s5H3I2H", header)
    name_len, extra_len = values[-2:]
    data_offset = entry.local_offset + 30 + name_len + extra_len
    compressed = archive.read(data_offset, entry.compressed_size)
    if entry.method == 0:
        raw = compressed
    elif entry.method == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"unsupported ZIP method {entry.method} for {entry.name}")
    if len(raw) != entry.uncompressed_size:
        raise ValueError(f"size mismatch for {entry.name}")
    if binascii.crc32(raw) & 0xFFFFFFFF != entry.crc32:
        raise ValueError(f"CRC mismatch for {entry.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--central-directory", type=Path, required=True)
    parser.add_argument("--list-csv", type=Path)
    parser.add_argument("--extract-list", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    entries = parse_central_directory(args.central_directory)
    if args.list_csv:
        args.list_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.list_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["name", "method", "flags", "crc32", "compressed_size", "uncompressed_size", "local_offset"]
            )
            for entry in entries:
                writer.writerow(
                    [
                        entry.name,
                        entry.method,
                        entry.flags,
                        f"{entry.crc32:08x}",
                        entry.compressed_size,
                        entry.uncompressed_size,
                        entry.local_offset,
                    ]
                )

    if args.extract_list:
        if not args.output_dir:
            parser.error("--output-dir is required with --extract-list")
        selected = [line.strip() for line in args.extract_list.read_text().splitlines() if line.strip()]
        by_name = {entry.name: entry for entry in entries}
        missing = [name for name in selected if name not in by_name]
        if missing:
            raise KeyError(f"members not found: {missing[:5]}")
        parts = json.loads(args.parts.read_text())
        archive = RemoteArchive(parts)
        for rank, name in enumerate(selected, 1):
            print(f"[{rank}/{len(selected)}] {name}", flush=True)
            extract_member(archive, by_name[name], args.output_dir / name)


if __name__ == "__main__":
    main()
