#!/usr/bin/env python3
"""Strict, reproducible validation for the frozen benchmark deliverables."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from decord import VideoReader, cpu
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
IVE = PROJECT / "model_evaluation/ivebench_lance"
AXES = ("appearance", "outfit", "action", "camera")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluated_hashes() -> set[str]:
    result = set()
    for relative in ("artifacts/short_subject/short_subject_manifest.csv", "artifacts/long_subject/manifest.csv",
                     "artifacts/attribute_all/manifest.csv"):
        for row in read_csv(IVE / relative):
            for key in ("official_source_sha256", "source_sha256"):
                if row.get(key):
                    result.add(row[key])
    return result


def jaccard(left: str, right: str) -> float:
    a, b = set(TOKEN_RE.findall(left.lower())), set(TOKEN_RE.findall(right.lower()))
    return len(a & b) / len(a | b) if a or b else 0.0


def validate_media(path: Path) -> dict:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                    "-map", "0:v:0", "-f", "null", "-"], check=True)
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
                           capture_output=True, text=True)
    match = re.search(r"Video:\s*([^,]+),\s*([^,\s]+)", probe.stderr)
    if not match:
        raise ValueError(f"cannot parse codec for {path}")
    codec, pixel_format = match.group(1).strip().split()[0], match.group(2).split("(")[0]
    header = path.read_bytes()[:2_000_000]
    faststart = header.find(b"moov") >= 0 and (header.find(b"mdat") < 0 or header.find(b"moov") < header.find(b"mdat"))
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    if codec != "h264" or pixel_format != "yuv420p" or not faststart or len(reader) < 2:
        raise ValueError(f"nonportable media {path}: codec={codec}, pixel_format={pixel_format}, faststart={faststart}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size,
            "frames": len(reader), "fps": round(float(reader.get_avg_fps()), 6), "codec": codec,
            "pixel_format": pixel_format, "faststart": faststart, "full_decode_ok": True}


def main() -> None:
    chains = [json.loads(line) for line in (ROOT / "benchmark.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    turns = [turn for chain in chains for turn in chain["turns"]]
    if len(chains) != 30 or len({chain["chain_id"] for chain in chains}) != 30:
        raise ValueError("expected 30 unique chains")
    if len(turns) != 112:
        raise ValueError(f"expected 112 turns, got {len(turns)}")
    lengths = Counter(len(chain["turns"]) for chain in chains)
    types = Counter(chain["chain_type"] for chain in chains)
    axes = Counter(turn["axis"] for turn in turns)
    if lengths != Counter({3: 13, 4: 12, 5: 5}): raise ValueError(f"length quotas: {lengths}")
    if types != Counter({"progressive": 18, "repeated_axis": 6, "rollback": 6}): raise ValueError(f"type quotas: {types}")
    if axes != Counter({axis: 28 for axis in AXES}): raise ValueError(f"axis quotas: {axes}")
    source_hashes = [chain["source"]["official_sha256"] for chain in chains]
    if len(set(source_hashes)) != 30: raise ValueError("duplicate source hashes")
    overlap = set(source_hashes) & evaluated_hashes()
    if overlap: raise ValueError(f"source overlap with prior IVEBench evaluation: {sorted(overlap)}")
    reserves = json.loads((ROOT / "metadata/reserve_candidates.json").read_text())
    reserve_hashes = [row["official_sha256"] for row in reserves]
    if len(reserves) != 5 or len(set(reserve_hashes)) != 5:
        raise ValueError("expected five unique reserve candidates")
    if set(reserve_hashes) & (set(source_hashes) | evaluated_hashes()):
        raise ValueError("reserve candidate overlaps benchmark or prior evaluation")
    instructions = []
    for chain in chains:
        expected_rounds = list(range(1, len(chain["turns"]) + 1))
        if [turn["round"] for turn in chain["turns"]] != expected_rounds:
            raise ValueError(f"invalid rounds in {chain['chain_id']}")
        for turn in chain["turns"]:
            text = turn["instruction"]
            if re.search(r"\b(rpmte|short_\d+|long_\d+|IVEBench)\b", text, re.I):
                raise ValueError(f"internal identifier leaked into instruction: {text}")
            reference = turn["history_reference_round"]
            if reference is not None and (not isinstance(reference, int) or reference < 0 or reference >= turn["round"]):
                raise ValueError(f"invalid history reference in {chain['chain_id']} round {turn['round']}")
            if chain["chain_type"] == "rollback" and turn["round"] == len(chain["turns"]):
                if reference is None: raise ValueError(f"rollback missing reference in {chain['chain_id']}")
            elif reference is not None:
                raise ValueError(f"unexpected history reference in {chain['chain_id']} round {turn['round']}")
            for prior in instructions:
                if jaccard(text, prior) > 0.75:
                    raise ValueError(f"near-duplicate benchmark instructions: {text!r} and {prior!r}")
            instructions.append(text)
    api = json.loads((ROOT / "metadata/api_summary.json").read_text())
    if api["required_provider"] != "Alibaba" or api["personal_credential_used"]:
        raise ValueError("invalid API provider/credential receipt")
    if float(api["reported_cost_usd"]) > float(api["hard_budget_usd"]) or float(api["hard_budget_usd"]) > 2.0:
        raise ValueError("API cost exceeded frozen cap")
    inventory = read_csv(ROOT / "source_inventory.csv")
    if len(inventory) != 30: raise ValueError("source inventory count mismatch")
    media = [validate_media(ROOT / chain["source"]["media"]) for chain in chains]
    report = {"status": "passed", "chains": len(chains), "turns": len(turns), "lengths": dict(lengths),
              "chain_types": dict(types), "axes": dict(axes), "unique_sources": len(set(source_hashes)),
              "prior_evaluation_overlap": 0, "reserve_candidates": len(reserves),
              "reported_api_cost_usd": api["reported_cost_usd"],
              "media_files": len(media), "media": media}
    (ROOT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "media"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
