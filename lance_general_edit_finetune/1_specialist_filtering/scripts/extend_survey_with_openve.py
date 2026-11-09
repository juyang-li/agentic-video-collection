#!/usr/bin/env python3
"""Append duration-qualified OpenVE local-change pairs to the frozen survey."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tarfile
from pathlib import Path
from typing import BinaryIO, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
OPENVE_ROOT = (ROOT / "../../dataset_research/openve_research").resolve()
DEFAULT_BASE_MANIFEST = ROOT / "artifacts/survey_manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/survey_manifest_openve_v2.csv"
DEFAULT_PREFIX = Path("/tmp/openve_local_change_prefix_2g.tar.gz")
SURVEY_VERSION = "specialist_error_survey_411_openve_v2"
OPENVE_REPO = "Lewandofski/OpenVE-3M"
OPENVE_REVISION = "d2c134f01c49d22efc2e0d9aefd798772f039a79"
PREFIX_BYTES = 2 * 1024**3
SOURCE_SETS = (
    (OPENVE_ROOT / "random_pairs.csv", OPENVE_ROOT / "openve_inventory.csv"),
    (
        OPENVE_ROOT / "batch2/random_pairs.csv",
        OPENVE_ROOT / "batch2/openve_inventory.csv",
    ),
)


class ConcatenatedReader:
    """Expose one or more archive byte ranges as a non-seekable stream."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = iter(paths)
        self.current: BinaryIO | None = None

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while size < 0 or remaining > 0:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            chunk = self.current.read(-1 if size < 0 else remaining)
            if chunk:
                chunks.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
                continue
            self.current.close()
            self.current = None
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_openve_rows(
    source_sets: tuple[tuple[Path, Path], ...] = SOURCE_SETS,
    min_duration: float = 3.0,
) -> list[dict[str, str]]:
    """Join the frozen random samples to inventories and apply stage-0 duration."""
    selected: list[dict[str, str]] = []
    for manifest_path, inventory_path in source_sets:
        inventory = {row["sample_id"]: row for row in read_csv(inventory_path)}
        for row in read_csv(manifest_path):
            technical = inventory.get(row["sample_id"])
            if technical is None:
                raise ValueError(f"missing OpenVE inventory row: {row['sample_id']}")
            if min(
                float(technical["source_duration_seconds"]),
                float(technical["target_duration_seconds"]),
            ) < min_duration:
                continue
            merged = dict(row)
            merged.update(
                {
                    "source_sha256": technical["source_sha256"],
                    "target_sha256": technical["target_sha256"],
                    "source_duration_seconds": technical["source_duration_seconds"],
                    "target_duration_seconds": technical["target_duration_seconds"],
                }
            )
            selected.append(merged)
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("OpenVE selection contains duplicate sample_id")
    return selected


class WantedMedia(NamedTuple):
    sha256: str
    destination: Path


def media_destination(sha256: str) -> Path:
    return ROOT / "artifacts/originals/openve" / f"{sha256[:20]}.mp4"


def extract_media(
    rows: list[dict[str, str]], prefix_paths: list[Path], expected_bytes: int
) -> None:
    actual_bytes = sum(path.stat().st_size for path in prefix_paths)
    if actual_bytes != expected_bytes:
        raise ValueError(f"OpenVE prefix size mismatch: {actual_bytes} != {expected_bytes}")

    wanted: dict[str, WantedMedia] = {}
    for row in rows:
        for role, member_key, hash_key in (
            ("source", "source_member", "source_sha256"),
            ("target", "target_member", "target_sha256"),
        ):
            media = WantedMedia(row[hash_key], media_destination(row[hash_key]))
            existing = wanted.setdefault(row[member_key], media)
            if existing != media:
                raise ValueError(f"archive member maps inconsistently: {row[member_key]}")

    pending: dict[str, WantedMedia] = {}
    for member, media in wanted.items():
        if media.destination.is_file() and sha256_file(media.destination) == media.sha256:
            continue
        pending[member] = media
    if not pending:
        print(f"openve_media preserved={len(wanted)} extracted=0")
        return

    extracted = 0
    stream = ConcatenatedReader(prefix_paths)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                normalized = member.name.removeprefix("./")
                media = pending.get(normalized)
                if media is None or not member.isfile():
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot extract OpenVE member: {normalized}")
                media.destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = media.destination.with_suffix(".part.mp4")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if sha256_file(temporary) != media.sha256:
                    temporary.unlink()
                    raise ValueError(f"OpenVE media hash mismatch: {normalized}")
                temporary.replace(media.destination)
                pending.pop(normalized)
                extracted += 1
                print(f"openve_media extracted={extracted}/{len(wanted)} pending={len(pending)}")
                if not pending:
                    break
    except (EOFError, OSError, tarfile.ReadError) as error:
        if pending:
            raise RuntimeError(
                f"OpenVE prefix ended with {len(pending)} selected media still pending"
            ) from error
    finally:
        stream.close()
    if pending:
        raise FileNotFoundError("missing OpenVE members: " + ", ".join(sorted(pending)))


def build_extended_manifest(
    base_rows: list[dict[str, str]], openve_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    if len(base_rows) != 400:
        raise ValueError(f"expected frozen 400-row base manifest, got {len(base_rows)}")
    if [int(row["sample_rank"]) for row in base_rows] != list(range(1, 401)):
        raise ValueError("base manifest ranks are not the frozen 1..400 sequence")

    output: list[dict[str, str]] = []
    for row in base_rows:
        copied = dict(row)
        copied["survey_version"] = SURVEY_VERSION
        output.append(copied)
    for rank, row in enumerate(openve_rows, start=401):
        source_hash = row["source_sha256"]
        target_hash = row["target_sha256"]
        output.append(
            {
                "survey_version": SURVEY_VERSION,
                "sample_rank": str(rank),
                "sample_id": row["sample_id"],
                "dataset": "openve",
                "branch": "openve_local_change",
                "pair_id": row["sample_id"],
                "operation": row["task"],
                "direction": "original_to_edited",
                "instruction": row["instruction"],
                "repo_id": OPENVE_REPO,
                "revision": OPENVE_REVISION,
                "shard": "videos/local_change_new.tar.gz.00[:2GiB]",
                "source_video_id": f"openve:sha256:{source_hash}",
                "target_video_id": f"openve:sha256:{target_hash}",
                "source_locator": row["source_member"],
                "target_locator": row["target_member"],
                "source_local": str(media_destination(source_hash).relative_to(ROOT)),
                "target_local": str(media_destination(target_hash).relative_to(ROOT)),
                "source_is_original": "1",
                "target_is_original": "0",
                "selection_seed": row["seed"],
            }
        )
    sample_ids = [row["sample_id"] for row in output]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("extended manifest contains duplicate sample_id")
    return output


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=Path, action="append")
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    openve_rows = select_openve_rows(min_duration=args.min_duration)
    if len(openve_rows) != 11:
        raise ValueError(f"expected frozen 11-row OpenVE extension, got {len(openve_rows)}")
    if not args.skip_extract:
        extract_media(openve_rows, args.prefix or [DEFAULT_PREFIX], PREFIX_BYTES)
    manifest = build_extended_manifest(read_csv(args.base_manifest), openve_rows)
    write_manifest(args.output, manifest)
    print(
        f"manifest={args.output} pairs={len(manifest)} openve={len(openve_rows)} "
        f"unique_openve_videos={len({row[key] for row in openve_rows for key in ('source_sha256', 'target_sha256')})}"
    )


if __name__ == "__main__":
    main()
