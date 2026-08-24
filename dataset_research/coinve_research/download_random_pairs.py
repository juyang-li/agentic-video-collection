#!/usr/bin/env python3
"""Extract selected MP4 members from remote uncompressed CoinVE tar shards."""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
REPO_ID = "FireCRT/CoinVE-200K"
REVISION = "7989adca1088928414242c57dd62ce485f740889"
BLOCK_SIZE = 512
PROBE_SIZE = 8 * 1024 * 1024
FINAL_SCAN_SIZE = 32 * 1024 * 1024
MAX_PROBE_SIZE = 128 * 1024 * 1024
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class WantedMember:
    member: str
    destination: Path
    sample_id: str
    role: str


@dataclass(frozen=True)
class TarEntry:
    name: str
    offset: int
    size: int

    @property
    def data_offset(self) -> int:
        return self.offset + BLOCK_SIZE

    @property
    def next_offset(self) -> int:
        return self.data_offset + math.ceil(self.size / BLOCK_SIZE) * BLOCK_SIZE


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
    )
    session = requests.Session()
    session.headers.update({"Accept-Encoding": "identity"})
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=2))
    return session


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def parse_octal(field: bytes) -> int:
    if field and field[0] & 0x80:
        return int.from_bytes(field, byteorder="big", signed=True)
    stripped = field.rstrip(b"\0 ").lstrip(b" ")
    return int(stripped or b"0", 8)


def parse_header(block: bytes) -> tuple[str, int, str]:
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"short tar header: {len(block)} bytes")
    if block == bytes(BLOCK_SIZE):
        return "", 0, "end"
    if block[257:262] != b"ustar":
        raise ValueError("missing tar magic")

    stored_checksum = parse_octal(block[148:156])
    checksum_block = block[:148] + (b" " * 8) + block[156:]
    if sum(checksum_block) != stored_checksum:
        raise ValueError("invalid tar header checksum")

    name = block[0:100].split(b"\0", 1)[0].decode("utf-8", errors="surrogateescape")
    prefix = block[345:500].split(b"\0", 1)[0].decode(
        "utf-8", errors="surrogateescape"
    )
    if prefix:
        name = f"{prefix}/{name}"
    size = parse_octal(block[124:136])
    typeflag = block[156:157].decode("ascii", errors="replace") or "0"
    return name.removeprefix("./"), size, typeflag


class RemoteTar:
    def __init__(self, shard_path: str) -> None:
        self.shard_path = shard_path
        self.session = make_session()
        encoded_path = quote(shard_path, safe="/")
        resolver_url = (
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
            f"{REVISION}/{encoded_path}?download=true"
        )
        response = self.session.head(
            resolver_url, allow_redirects=True, timeout=(30, 60)
        )
        response.raise_for_status()
        self.url = response.url
        self.size = int(response.headers["Content-Length"])
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError(f"server does not advertise byte ranges: {shard_path}")

    def read_range(self, start: int, end: int) -> bytes:
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(30, 120),
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(
                    f"range request failed for {self.shard_path}: "
                    f"status={response.status_code}"
                )
            data = response.content
        finally:
            response.close()
        expected = end - start + 1
        if len(data) != expected:
            raise RuntimeError(
                f"short range for {self.shard_path}: expected={expected} got={len(data)}"
            )
        return data

    def close(self) -> None:
        self.session.close()


def scan_regular_headers(data: bytes, base_offset: int) -> list[TarEntry]:
    entries: list[TarEntry] = []
    for relative_offset in range(0, len(data) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block = data[relative_offset : relative_offset + BLOCK_SIZE]
        try:
            name, size, typeflag = parse_header(block)
        except ValueError:
            continue
        if name and typeflag in ("0", "\0", "7"):
            entries.append(
                TarEntry(
                    name=name,
                    offset=base_offset + relative_offset,
                    size=size,
                )
            )
    names = [entry.name for entry in entries]
    if names != sorted(names):
        raise RuntimeError("tar members are not sorted; binary range lookup is unsafe")
    return entries


def locate_member(remote: RemoteTar, target: str) -> tuple[TarEntry, int]:
    low = 0
    high = remote.size
    transferred = 0

    for _ in range(48):
        span = high - low
        if span <= 0:
            break
        if span <= FINAL_SCAN_SIZE:
            data = remote.read_range(low, high - 1)
            transferred += len(data)
            entries = scan_regular_headers(data, low)
            for entry in entries:
                if entry.name == target:
                    return entry, transferred
            break

        midpoint = low + span // 2
        probe_size = PROBE_SIZE
        entries: list[TarEntry] = []
        while probe_size <= MAX_PROBE_SIZE and not entries:
            start = max(low, midpoint - probe_size // 2)
            start = start // BLOCK_SIZE * BLOCK_SIZE
            end = min(high, start + probe_size)
            end = end // BLOCK_SIZE * BLOCK_SIZE
            if end <= start:
                break
            data = remote.read_range(start, end - 1)
            transferred += len(data)
            entries = scan_regular_headers(data, start)
            probe_size *= 2
        if not entries:
            raise RuntimeError(
                f"could not find a tar header near offset {midpoint} in {remote.shard_path}"
            )

        for entry in entries:
            if entry.name == target:
                return entry, transferred

        lower = [entry for entry in entries if entry.name < target]
        upper = [entry for entry in entries if entry.name > target]
        old_bounds = (low, high)
        if lower:
            low = max(low, lower[-1].next_offset)
        if upper:
            high = min(high, upper[0].offset)
        if (low, high) == old_bounds:
            break

    raise FileNotFoundError(f"member not found in {remote.shard_path}: {target}")


def extract_shard(shard_path: str, wanted: list[WantedMember]) -> dict[str, int]:
    pending = {item.member: item for item in wanted if not item.destination.is_file()}
    preserved = len(wanted) - len(pending)
    if not pending:
        log(f"[{shard_path}] preserved={preserved} scan=skipped")
        return {"extracted": 0, "preserved": preserved, "bytes": 0}

    remote = RemoteTar(shard_path)
    extracted = 0
    transferred = 0
    started = time.monotonic()
    log(
        f"[{shard_path}] scan_start size={remote.size} "
        f"targets={len(pending)} preserved={preserved}"
    )
    try:
        for name, item in list(pending.items()):
            entry, lookup_bytes = locate_member(remote, name)
            transferred += lookup_bytes
            payload = remote.read_range(
                entry.data_offset, entry.data_offset + entry.size - 1
            )
            transferred += len(payload)
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            part_path = item.destination.with_name(item.destination.name + ".part")
            part_path.write_bytes(payload)
            part_path.replace(item.destination)
            pending.pop(name)
            extracted += 1
            log(
                f"[{shard_path}] extracted {item.sample_id} {item.role} "
                f"member={name} bytes={entry.size} lookup_mb={lookup_bytes / 1e6:.1f}"
            )
    finally:
        remote.close()

    if pending:
        names = ", ".join(sorted(pending))
        raise FileNotFoundError(f"members not found in {shard_path}: {names}")
    elapsed = time.monotonic() - started
    log(
        f"[{shard_path}] scan_done extracted={extracted} preserved={preserved} "
        f"transferred_mb={transferred / 1e6:.1f} elapsed_s={elapsed:.1f}"
    )
    return {"extracted": extracted, "preserved": preserved, "bytes": transferred}


def load_jobs(sample_ids: set[str] | None) -> dict[str, list[WantedMember]]:
    grouped: dict[str, list[WantedMember]] = defaultdict(list)
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if sample_ids:
        known = {row["sample_id"] for row in rows}
        unknown = sample_ids - known
        if unknown:
            raise ValueError(f"unknown sample ids: {sorted(unknown)}")
        rows = [row for row in rows if row["sample_id"] in sample_ids]

    for row in rows:
        for role in ("source", "target"):
            destination = (ROOT / row[f"{role}_file"]).resolve()
            if ROOT.resolve() not in destination.parents:
                raise ValueError(f"destination escapes research root: {destination}")
            grouped[row[f"{role}_shard"]].append(
                WantedMember(
                    member=row[f"{role}_member"],
                    destination=destination,
                    sample_id=row["sample_id"],
                    role=role,
                )
            )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-id",
        action="append",
        help="download only this manifest sample; may be repeated",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    grouped = load_jobs(set(args.sample_id) if args.sample_id else None)
    totals = {"extracted": 0, "preserved": 0, "bytes": 0}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(grouped))) as executor:
        futures = {
            executor.submit(extract_shard, shard, wanted): shard
            for shard, wanted in grouped.items()
        }
        for future in as_completed(futures):
            result = future.result()
            for key, value in result.items():
                totals[key] += value

    log(
        f"done shards={len(grouped)} extracted={totals['extracted']} "
        f"preserved={totals['preserved']} transferred_gb={totals['bytes'] / 1e9:.2f}"
    )


if __name__ == "__main__":
    main()
