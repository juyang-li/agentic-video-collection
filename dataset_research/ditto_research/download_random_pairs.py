#!/usr/bin/env python3
"""Stream Ditto's split gzip from .01 and save selected local/0130 pairs."""

from __future__ import annotations

import csv
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "random_pairs.csv"
REPO_ID = "QingyanBai/Ditto-1M"
REVISION = "aac1417418d8f7e77d4e257d5620c36b2c3a00e8"
ARCHIVE_PATH = "videos/local/local.tar.gz.01"


@dataclass(frozen=True)
class Destination:
    path: Path
    sample_id: str
    role: str


class CountingReader:
    def __init__(self, raw: object) -> None:
        self.raw = raw
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.bytes_read += len(data)
        return data


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"Accept-Encoding": "identity"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def load_wanted() -> dict[str, list[Destination]]:
    grouped: dict[str, list[Destination]] = defaultdict(list)
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for role, member_field, file_field in (
            ("source", "source_path", "source_file"),
            ("target", "edited_path", "target_file"),
        ):
            destination = (ROOT / row[file_field]).resolve()
            if ROOT.resolve() not in destination.parents:
                raise ValueError(f"destination escapes research root: {destination}")
            grouped[row[member_field]].append(
                Destination(destination, row["sample_id"], role)
            )
    return grouped


def main() -> None:
    wanted = load_wanted()
    pending = {
        member: [destination for destination in destinations if not destination.path.is_file()]
        for member, destinations in wanted.items()
    }
    pending = {member: destinations for member, destinations in pending.items() if destinations}
    preserved = sum(
        destination.path.is_file()
        for destinations in wanted.values()
        for destination in destinations
    )
    if not pending:
        print(f"done extracted=0 preserved={preserved} scan=skipped")
        return

    url = (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
        f"{REVISION}/{ARCHIVE_PATH}?download=true"
    )
    session = make_session()
    response = session.get(url, stream=True, timeout=(30, 180))
    response.raise_for_status()
    response.raw.decode_content = False
    counting = CountingReader(response.raw)
    extracted = 0
    scanned_mp4 = 0
    try:
        with tarfile.open(fileobj=counting, mode="r|gz") as archive:
            for member in archive:
                normalized = member.name.removeprefix("./")
                if not member.isfile() or not normalized.lower().endswith(".mp4"):
                    continue
                scanned_mp4 += 1
                if scanned_mp4 % 100 == 0:
                    print(
                        f"progress scanned_mp4={scanned_mp4} "
                        f"pending_members={len(pending)} "
                        f"compressed_mb={counting.bytes_read / 1e6:.1f}",
                        flush=True,
                    )
                destinations = pending.get(normalized)
                if not destinations:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read member: {member.name}")
                first = destinations[0]
                first.path.parent.mkdir(parents=True, exist_ok=True)
                part_path = first.path.with_name(first.path.name + ".part")
                with part_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                part_path.replace(first.path)
                for duplicate in destinations[1:]:
                    duplicate.path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(first.path, duplicate.path)
                extracted += len(destinations)
                pending.pop(normalized)
                print(
                    f"matched member={normalized} roles="
                    f"{','.join(x.sample_id + ':' + x.role for x in destinations)} "
                    f"pending_members={len(pending)}",
                    flush=True,
                )
                if not pending:
                    break
    finally:
        response.close()
        session.close()

    if pending:
        raise FileNotFoundError(
            "selected members were not found before the end of .01: "
            + ", ".join(sorted(pending))
        )
    print(
        f"done extracted={extracted} preserved={preserved} "
        f"scanned_mp4={scanned_mp4} compressed_prefix_mb={counting.bytes_read / 1e6:.1f}"
    )


if __name__ == "__main__":
    main()
