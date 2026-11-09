#!/usr/bin/env python3
"""Download one official tar at a time and extract only Stage 1 pilot members."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/filter_run_v1/stage1_0_survivors.csv"
DEFAULT_STATE = ROOT / "artifacts/filter_run_v1/media_acquisition_state.json"


@dataclass(frozen=True)
class Destination:
    path: Path
    dataset: str
    locator: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_output(value: str) -> Path:
    path = Path(value)
    output = path if path.is_absolute() else ROOT / path
    output = output.resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError(f"media destination escapes specialist root: {output}")
    return output


def parse_mappings(values: list[str]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --archive-map: {value}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mappings[name] = path
    return mappings


def archive_digest(repo_id: str, revision: str, filename: str) -> str:
    return hashlib.sha256(f"{repo_id}:{revision}:{filename}".encode()).hexdigest()[:12]


def download_archive_requests(
    repo_id: str, revision: str, filename: str, work_dir: Path
) -> tuple[Path, Path | None]:
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    digest = archive_digest(repo_id, revision, filename)
    destination = work_dir / f"{digest}_{Path(filename).name}"
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = build_hf_headers(token=True)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    url = hf_hub_url(repo_id, filename, repo_type="dataset", revision=revision)
    response = requests.get(url, headers=headers, stream=True, timeout=180)
    if offset and response.status_code == 200:
        offset = 0
    elif response.status_code not in ({206} if offset else {200}):
        raise RuntimeError(f"download failed status={response.status_code} filename={filename}")
    mode = "ab" if offset else "wb"
    work_dir.mkdir(parents=True, exist_ok=True)
    with partial.open(mode) as output:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                output.write(chunk)
    partial.replace(destination)
    return destination, None


def download_archive_xet(
    repo_id: str, revision: str, filename: str, work_dir: Path
) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    digest = archive_digest(repo_id, revision, filename)
    cache_dir = (work_dir / f"{digest}_xet_cache").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            cache_dir=cache_dir,
        )
    )
    return path, cache_dir


def extract_wanted(
    archive_path: Path, destinations: list[Destination]
) -> tuple[int, list[str]]:
    exact: dict[str, list[Destination]] = defaultdict(list)
    basename: dict[str, list[Destination]] = defaultdict(list)
    for item in destinations:
        if item.path.is_file() and item.path.stat().st_size > 0:
            continue
        normalized = item.locator.removeprefix("./")
        exact[normalized].append(item)
        if item.dataset in {"insave", "reco"}:
            basename[Path(normalized).name].append(item)
    if not exact:
        return 0, []

    extracted = 0
    found: set[Destination] = set()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized = member.name.removeprefix("./")
            targets = exact.get(normalized, [])
            if not targets:
                targets = basename.get(Path(normalized).name, [])
            pending = [item for item in targets if item not in found]
            if not pending:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read member {member.name}")
            first = pending[0].path
            first.parent.mkdir(parents=True, exist_ok=True)
            temporary = first.with_suffix(first.suffix + ".part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            temporary.replace(first)
            found.add(pending[0])
            extracted += 1
            for duplicate in pending[1:]:
                duplicate.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(first, duplicate.path)
                found.add(duplicate)
                extracted += 1
            if len(found) == len(destinations):
                break
    missing = [item.locator for item in destinations if not item.path.is_file()]
    return extracted, missing


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.json")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "lance_stage1_archives")
    parser.add_argument("--dataset", action="append", choices=("insave", "goku", "reco"))
    parser.add_argument("--only-archive", action="append", default=[])
    parser.add_argument("--archive-map", action="append", default=[])
    parser.add_argument("--download-backend", choices=("requests", "hf-xet"), default="requests")
    parser.add_argument("--keep-downloaded-archives", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record members absent from an official archive and continue with later archives.",
    )
    parser.add_argument("--max-archives", type=int)
    args = parser.parse_args()

    datasets = set(args.dataset or ("insave", "goku", "reco"))
    only_archives = set(args.only_archive)
    local_mappings = parse_mappings(args.archive_map)
    grouped: dict[tuple[str, str, str], list[Destination]] = defaultdict(list)
    for row in read_csv(args.manifest):
        if row["dataset"] not in datasets:
            continue
        for role in ("source", "target"):
            archive = row[f"{role}_archive"]
            if only_archives and archive not in only_archives:
                continue
            grouped[(row["repo_id"], row["revision"], archive)].append(
                Destination(
                    safe_output(row[f"{role}_local"]), row["dataset"], row[f"{role}_locator"]
                )
            )
    keys = sorted(grouped)
    if args.max_archives is not None:
        keys = keys[: args.max_archives]
    state: dict[str, object] = {
        "manifest": str(args.manifest),
        "archives_total": len(keys),
        "archives_complete": 0,
        "extracted_files": 0,
        "archives": {},
    }
    for index, (repo_id, revision, archive) in enumerate(keys, 1):
        destinations = grouped[(repo_id, revision, archive)]
        unique_destinations = list({item.path: item for item in destinations}.values())
        pending_destinations = [
            item
            for item in unique_destinations
            if not item.path.is_file() or item.path.stat().st_size == 0
        ]
        if not pending_destinations:
            state["archives_complete"] = index
            state["archives"][archive] = {  # type: ignore[index]
                "wanted": len(unique_destinations),
                "newly_extracted": 0,
                "missing": 0,
                "missing_locators": [],
                "status": "preserved",
            }
            write_state(args.state, state)
            print(
                f"archive={index}/{len(keys)} name={archive} wanted={len(unique_destinations)} "
                "newly_extracted=0 missing=0 preserved=all",
                flush=True,
            )
            continue
        archive_path = local_mappings.get(archive)
        downloaded = archive_path is None
        cleanup_dir: Path | None = None
        if archive_path is None:
            if args.download_backend == "hf-xet":
                archive_path, cleanup_dir = download_archive_xet(
                    repo_id, revision, archive, args.work_dir
                )
            else:
                archive_path, cleanup_dir = download_archive_requests(
                    repo_id, revision, archive, args.work_dir
                )
        extracted, missing = extract_wanted(archive_path, pending_destinations)
        if missing and not args.allow_missing:
            raise FileNotFoundError(
                f"{archive_path.name} did not contain {len(missing)} wanted members: {missing[:5]}"
            )
        state["archives_complete"] = index
        state["extracted_files"] = int(state["extracted_files"]) + extracted
        state["archives"][archive] = {  # type: ignore[index]
            "wanted": len(unique_destinations),
            "newly_extracted": extracted,
            "missing": len(missing),
            "missing_locators": missing,
            "status": "complete" if not missing else "complete_with_missing",
        }
        write_state(args.state, state)
        print(
            f"archive={index}/{len(keys)} name={archive} wanted={len(unique_destinations)} "
            f"newly_extracted={extracted} missing={len(missing)}",
            flush=True,
        )
        if downloaded and not args.keep_downloaded_archives:
            if cleanup_dir is not None:
                if cleanup_dir.parent != args.work_dir.resolve() or not cleanup_dir.name.endswith("_xet_cache"):
                    raise ValueError(f"refusing to remove unexpected cache directory: {cleanup_dir}")
                shutil.rmtree(cleanup_dir)
            else:
                archive_path.unlink(missing_ok=True)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
