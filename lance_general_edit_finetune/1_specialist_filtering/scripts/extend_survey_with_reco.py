#!/usr/bin/env python3
"""Append new, duration-qualified ReCo pairs to the frozen 400-pair survey."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import re
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECO_RESEARCH = (ROOT / "../../dataset_research/reco_research").resolve()
DEFAULT_BASE_MANIFEST = ROOT / "artifacts/survey_manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/survey_manifest_reco_v3.csv"
DEFAULT_SELECTION_OUTPUT = ROOT / "artifacts/reco_extension_selection_v3.csv"
DEFAULT_WORK_DIR = Path("/tmp/reco_survey_63c6481")
RECO_REPO = "HiDream-ai/ReCo-Data"
RECO_REVISION = "63c6481a4aedae39ec602ce8e17a836cc07b4ca9"
SURVEY_VERSION = "specialist_error_survey_460_reco_v3"
SELECTION_SEED = 20260829
MIN_DURATION = 3.0
PREFIX_BYTES = 768 * 1024**2
TASK_SHARDS = {"add": 38, "remove": 26, "replace": 43}
TASK_QUOTAS = {
    "add": {"person": 5, "wearable": 5, "object": 10},
    "remove": {"person": 5, "wearable": 5, "object": 10},
    "replace": {"person": 5, "wearable": 5, "object": 10},
}

FIXED_PROVENANCE_RULES = {
    (
        "suimu/InsAVE-80K",
        "8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289",
        "insave_remove",
    ): ("original", "edited", "official_dataset_rule"),
    (
        "bigfacing/GOKU-2M",
        "f289f4b245648dc57db04afaaeb68659440c0fa6",
        "goku_remove",
    ): ("original", "edited", "official_dataset_rule"),
    (
        "bigfacing/GOKU-2M",
        "f289f4b245648dc57db04afaaeb68659440c0fa6",
        "goku_add",
    ): ("edited", "original", "official_dataset_rule"),
    (
        RECO_REPO,
        RECO_REVISION,
        "reco_add",
    ): ("original", "edited", "official_dataset_rule"),
    (
        RECO_REPO,
        RECO_REVISION,
        "reco_remove",
    ): ("original", "edited", "official_dataset_rule"),
    (
        RECO_REPO,
        RECO_REVISION,
        "reco_replace",
    ): ("original", "edited", "official_dataset_rule"),
}

PERSON_RE = re.compile(
    r"\b(?:person|people|man|men|woman|women|boy|girl|child|children|baby|"
    r"gentleman|lady|bride|groom|worker|chef|player|rider|skier|surfer)\b",
    re.IGNORECASE,
)
ACCESSORY_RE = re.compile(
    r"\b(?:glasses|sunglasses|goggles|hat|cap|helmet|necklace|earring|earrings|"
    r"bracelet|watch|scarf|tie|bowtie|backpack|handbag|purse)\b",
    re.IGNORECASE,
)
CLOTHING_RE = re.compile(
    r"\b(?:clothes|clothing|shirt|t-shirt|tee|blouse|sweater|hoodie|jacket|coat|"
    r"dress|skirt|pants|trousers|shorts|jeans|shoe|shoes|boot|boots|sneaker|"
    r"sneakers|sock|socks|underwear|bra|bikini|swimsuit|uniform|vest|apron)\b",
    re.IGNORECASE,
)


def provenance_fields(
    source: str, target: str, basis: str
) -> dict[str, str]:
    if source == "original" and target == "edited":
        direction = "original_to_edited"
    elif source == "edited" and target == "original":
        direction = "edited_to_original"
    elif source == "edited" and target == "edited":
        direction = "edited_to_edited"
    elif source == "original" and target == "original":
        direction = "original_to_original"
    else:
        direction = "unknown"

    edited_roles = [
        role for role, value in (("source", source), ("target", target))
        if value == "edited"
    ]
    quality_roles = edited_roles[0] if len(edited_roles) == 1 else "both"
    return {
        "direction": direction,
        "source_is_original": "1" if source == "original" else "0" if source == "edited" else "",
        "target_is_original": "1" if target == "original" else "0" if target == "edited" else "",
        "source_provenance": source,
        "target_provenance": target,
        "provenance_basis": basis,
        "quality_checked_roles": quality_roles,
    }


def fixed_provenance(repo_id: str, revision: str, branch: str) -> dict[str, str]:
    """Resolve roles only for audited dataset revisions; otherwise inspect both."""
    roles = FIXED_PROVENANCE_RULES.get((repo_id, revision, branch))
    if roles is None:
        return provenance_fields("unknown", "unknown", "unknown")
    return provenance_fields(*roles)


def manifest_row_provenance(row: dict[str, str]) -> dict[str, str]:
    if (
        row.get("dataset") == "egoedit"
        and row.get("repo_id") == "liguang0115/EgoEdit"
        and row.get("revision") == "a9bd3faafe51a581b359acd7c6ad5c432508ff60"
    ):
        source = "original" if row.get("source_video_id", "").endswith("_no_change") else "edited"
        target = "original" if row.get("target_video_id", "").endswith("_no_change") else "edited"
        return provenance_fields(source, target, "pair_metadata")
    return fixed_provenance(
        row.get("repo_id", ""), row.get("revision", ""), row.get("branch", "")
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def old_example_stems() -> set[str]:
    return {path.stem for path in (RECO_RESEARCH / "examples").glob("*/*.mp4")}


def classify_scope(task: str, instruction: str) -> str | None:
    """Map a prompt to the three deliberately broad audit strata."""
    is_accessory = bool(ACCESSORY_RE.search(instruction))
    is_clothing = bool(CLOTHING_RE.search(instruction))
    if task == "remove" and is_clothing and not is_accessory:
        return None
    if is_accessory or is_clothing:
        return "wearable"
    if PERSON_RE.search(instruction):
        return "person"
    return "object"


def download_inputs(work_dir: Path) -> dict[str, tuple[Path, Path, Path]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to fetch ReCo") from error

    filenames: dict[str, tuple[str, str, str]] = {}
    for task, shard in TASK_SHARDS.items():
        filenames[task] = (
            f"{task}/{task}_data_configs.json",
            f"{task}/src_videos/{task}_src_video_{shard:02d}.tar",
            f"{task}/tar_videos/{task}_tar_video_{shard:02d}.tar",
        )

    def download(name: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=RECO_REPO,
                repo_type="dataset",
                filename=name,
                revision=RECO_REVISION,
                local_dir=work_dir,
            )
        )

    downloaded: dict[str, Path] = {}
    all_names = [name for names in filenames.values() for name in names]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download, name): name for name in all_names}
        for future in as_completed(futures):
            name = futures[future]
            downloaded[name] = future.result()
            print(f"downloaded={name}", flush=True)
    return {
        task: tuple(downloaded[name] for name in names)  # type: ignore[misc]
        for task, names in filenames.items()
    }


def available_download(work_dir: Path, filename: str) -> Path:
    completed = work_dir / filename
    if completed.is_file():
        return completed
    candidates = list(
        (work_dir / ".huggingface/download" / Path(filename).parent).glob(
            Path(filename).name + ".*.incomplete"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"cannot resolve one local download for {filename}")
    return candidates[0]


def copy_prefix(source: Path, destination: Path, byte_count: int) -> None:
    if source.stat().st_size < byte_count:
        raise ValueError(f"download prefix is too short: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.tar")
    remaining = byte_count
    with source.open("rb") as input_file, temporary.open("wb") as output_file:
        while remaining:
            chunk = input_file.read(min(8 * 1024**2, remaining))
            if not chunk:
                raise EOFError(f"download ended early: {source}")
            output_file.write(chunk)
            remaining -= len(chunk)
    temporary.replace(destination)


def prepare_local_prefixes(work_dir: Path) -> dict[str, tuple[Path, Path, Path]]:
    """Freeze equal byte prefixes from resumable local downloads."""
    result: dict[str, tuple[Path, Path, Path]] = {}
    for task, shard in TASK_SHARDS.items():
        config = available_download(work_dir, f"{task}/{task}_data_configs.json")
        source_name = f"{task}/src_videos/{task}_src_video_{shard:02d}.tar"
        target_name = f"{task}/tar_videos/{task}_tar_video_{shard:02d}.tar"
        prefix_dir = work_dir / "fixed_prefixes"
        source_prefix = prefix_dir / f"{task}_src_video_{shard:02d}.prefix_768m.tar"
        target_prefix = prefix_dir / f"{task}_tar_video_{shard:02d}.prefix_768m.tar"
        for name, destination in (
            (source_name, source_prefix),
            (target_name, target_prefix),
        ):
            if destination.is_file() and destination.stat().st_size == PREFIX_BYTES:
                continue
            copy_prefix(available_download(work_dir, name), destination, PREFIX_BYTES)
        result[task] = (config, source_prefix, target_prefix)
    return result


def tar_members_by_stem(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    while True:
        try:
            member = archive.next()
        except (EOFError, OSError, tarfile.ReadError):
            break
        if member is None:
            break
        if not member.isfile() or Path(member.name).suffix.lower() != ".mp4":
            continue
        if member.offset_data + member.size > PREFIX_BYTES:
            continue
        stem = Path(member.name).stem
        if stem in result:
            raise ValueError(f"duplicate video stem in tar: {stem}")
        result[stem] = member
    return result


def probe_duration(ffmpeg: str, path: Path) -> float:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if match is None:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path
) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"cannot extract ReCo member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part.mp4")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    temporary.replace(destination)


def media_path(task: str, stem: str, role: str) -> Path:
    return ROOT / "artifacts/originals/reco" / task / f"{stem}_{role}.mp4"


def select_and_extract_task(
    task: str,
    config_path: Path,
    source_tar_path: Path,
    target_tar_path: Path,
    ffmpeg: str,
    min_duration: float,
) -> tuple[list[dict[str, str]], Counter[str]]:
    configs = json.loads(config_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in configs:
        grouped[Path(row["src_video"]).stem].append(row)
    ambiguous_stems = {stem for stem, rows in grouped.items() if len(rows) != 1}
    by_stem = {stem: rows[0] for stem, rows in grouped.items() if len(rows) == 1}
    excluded = old_example_stems()
    accepted: list[dict[str, str]] = []
    rejected = Counter()

    with tarfile.open(source_tar_path, "r:") as source_tar, tarfile.open(
        target_tar_path, "r:"
    ) as target_tar:
        source_members = tar_members_by_stem(source_tar)
        target_members = tar_members_by_stem(target_tar)
        common = sorted(set(source_members) & set(target_members) & set(grouped))
        if len(common) < sum(TASK_QUOTAS[task].values()):
            raise ValueError(f"too few matched ReCo pairs in {task} shard: {len(common)}")

        pools: dict[str, list[str]] = {scope: [] for scope in TASK_QUOTAS[task]}
        for stem in common:
            if stem in ambiguous_stems:
                rejected["ambiguous_duplicate_metadata"] += 1
                continue
            if stem in excluded:
                rejected["old_example"] += 1
                continue
            scope = classify_scope(task, by_stem[stem]["instruction_final_refine"])
            if scope is None:
                rejected["irrational_clothing_remove"] += 1
                continue
            pools[scope].append(stem)

        for scope, quota in TASK_QUOTAS[task].items():
            rng = random.Random(f"{SELECTION_SEED}:{task}:{scope}")
            rng.shuffle(pools[scope])
            for stem in pools[scope]:
                if sum(row["scope"] == scope for row in accepted) >= quota:
                    break
                source_path = media_path(task, stem, "source")
                target_path = media_path(task, stem, "target")
                extract_member(source_tar, source_members[stem], source_path)
                extract_member(target_tar, target_members[stem], target_path)
                source_duration = probe_duration(ffmpeg, source_path)
                target_duration = probe_duration(ffmpeg, target_path)
                if min(source_duration, target_duration) < min_duration:
                    source_path.unlink(missing_ok=True)
                    target_path.unlink(missing_ok=True)
                    rejected["short_or_invalid"] += 1
                    continue
                accepted.append(
                    {
                        "task": task,
                        "scope": scope,
                        "stem": stem,
                        "instruction": by_stem[stem]["instruction_final_refine"].strip(),
                        "source_member": source_members[stem].name,
                        "target_member": target_members[stem].name,
                        "source_duration_seconds": f"{source_duration:.3f}",
                        "target_duration_seconds": f"{target_duration:.3f}",
                    }
                )
            actual = sum(row["scope"] == scope for row in accepted)
            if actual != quota:
                raise ValueError(f"ReCo {task}/{scope}: expected {quota}, accepted {actual}")
    return accepted, rejected


def build_extended_manifest(
    base_rows: list[dict[str, str]], reco_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    if len(base_rows) != 400:
        raise ValueError(f"expected frozen 400-row base manifest, got {len(base_rows)}")
    if [int(row["sample_rank"]) for row in base_rows] != list(range(1, 401)):
        raise ValueError("base manifest ranks are not the frozen 1..400 sequence")
    if len(reco_rows) != 60:
        raise ValueError(f"expected 60 ReCo pairs, got {len(reco_rows)}")

    output: list[dict[str, str]] = []
    for row in base_rows:
        copied = dict(row)
        copied["survey_version"] = SURVEY_VERSION
        copied.update(manifest_row_provenance(copied))
        output.append(copied)
    for rank, row in enumerate(reco_rows, start=401):
        task = row["task"]
        stem = row["stem"]
        shard = TASK_SHARDS[task]
        provenance = fixed_provenance(RECO_REPO, RECO_REVISION, f"reco_{task}")
        output.append(
            {
                "survey_version": SURVEY_VERSION,
                "sample_rank": str(rank),
                "sample_id": f"reco-{task}-{stem[:20]}",
                "dataset": "reco",
                "branch": f"reco_{task}",
                "pair_id": stem,
                "operation": task,
                "direction": provenance["direction"],
                "instruction": row["instruction"],
                "repo_id": RECO_REPO,
                "revision": RECO_REVISION,
                "shard": (
                    f"{task}_src_video_{shard:02d}.tar + "
                    f"{task}_tar_video_{shard:02d}.tar; scope={row['scope']}"
                ),
                "source_video_id": f"reco:{task}:{stem}:source",
                "target_video_id": f"reco:{task}:{stem}:target",
                "source_locator": row["source_member"],
                "target_locator": row["target_member"],
                "source_local": str(media_path(task, stem, "source").relative_to(ROOT)),
                "target_local": str(media_path(task, stem, "target").relative_to(ROOT)),
                **provenance,
                "selection_seed": str(SELECTION_SEED),
            }
        )
    sample_ids = [row["sample_id"] for row in output]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("extended manifest contains duplicate sample_id")
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION)
    parser.add_argument(
        "--use-local-prefixes",
        action="store_true",
        help="freeze 768 MiB prefixes from existing complete/resumable downloads",
    )
    args = parser.parse_args()

    inputs = (
        prepare_local_prefixes(args.work_dir)
        if args.use_local_prefixes
        else download_inputs(args.work_dir)
    )
    reco_rows: list[dict[str, str]] = []
    for task in ("add", "remove", "replace"):
        selected, rejected = select_and_extract_task(
            task, *inputs[task], args.ffmpeg, args.min_duration
        )
        reco_rows.extend(selected)
        print(
            f"reco_task={task} selected={len(selected)} "
            f"scopes={dict(Counter(row['scope'] for row in selected))} "
            f"rejected={dict(rejected)}",
            flush=True,
        )
    manifest = build_extended_manifest(read_csv(args.base_manifest), reco_rows)
    write_csv(args.selection_output, reco_rows)
    write_csv(args.output, manifest)
    print(f"manifest={args.output} pairs={len(manifest)} reco={len(reco_rows)}")


if __name__ == "__main__":
    main()
