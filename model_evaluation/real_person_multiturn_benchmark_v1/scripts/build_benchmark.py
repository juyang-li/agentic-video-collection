#!/usr/bin/env python3
"""Build a small real-video, person-centric, multi-turn editing benchmark.

The builder is deliberately restartable.  Every model response is written before
the next candidate is attempted, downloaded IVEBench files are hash checked, and
the final benchmark is only materialized after all frozen quotas are satisfied.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import requests
from decord import VideoReader, cpu


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
IVE = PROJECT / "model_evaluation" / "ivebench_lance"
OFFICIAL = IVE / "data" / "official"
SHORT_PROMPTS = OFFICIAL / "ivebenchdb_prompt_short.json"
LONG_PROMPTS = OFFICIAL / "ivebenchdb_prompt_long.json"
TREE = OFFICIAL / "dataset_tree.json"
SENIOR_CHAINS = Path(
    "/scratch/prj0000000275/jcy/xks/DataMake/00_engine/outputs/delivery/"
    "first_batch_candidate_20260830_v002/dataset_v1/chains.json"
)

DATASET_ID = "Coraxor/IVEBench"
DATASET_REVISION = "1e6cb655c0a537f6093edcf18e2607fb03e76734"
MODEL = "qwen/qwen3.8-omni-flash"
EXPECTED_PROVIDER = "Alibaba"
SEED = 20260929
API_BUDGET_USD = 2.0
TARGET_CHAINS = 30
RESERVE_ACCEPTS = 5
AXES = ("appearance", "outfit", "action", "camera")
TARGET_AXIS_COUNT = {axis: 28 for axis in AXES}
TARGET_LENGTHS = {3: 13, 4: 12, 5: 5}
TARGET_TYPES = {"progressive": 18, "repeated_axis": 6, "rollback": 6}
PERSON_RE = re.compile(
    r"\b(man|men|woman|women|person|people|individual|adult|athlete|dancer|performer|"
    r"singer|musician|chef|worker|doctor|nurse|artist|player|surfer|skier|cyclist|rider|"
    r"runner|speaker|presenter|bride|groom|farmer|soldier|officer|customer|employee)\b",
    re.I,
)
MINOR_RE = re.compile(r"\b(child|children|kid|kids|baby|babies|boy|boys|girl|girls|teen|teenager)\b", re.I)
NONREAL_RE = re.compile(r"\b(cartoon|anime|animated|animation|statue|sculpture|robot|mannequin|toy|painting)\b", re.I)
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Blueprint:
    slot: int
    chain_type: str
    length: int
    axes: tuple[str, ...]
    rollback_turn: int | None

    @property
    def chain_id(self) -> str:
        return f"rpmte_{self.slot:03d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def candidate_sequences(kind: str, length: int) -> list[tuple[str, ...]]:
    sequences = []
    for seq in itertools.product(AXES, repeat=length):
        counts = Counter(seq)
        if kind == "progressive" and len(counts) == length:
            sequences.append(seq)
        elif kind in {"repeated_axis", "rollback"}:
            if sorted(counts.values()) == [1] * (length - 2) + [2]:
                if kind == "repeated_axis" or seq[-1] in seq[:-1]:
                    sequences.append(seq)
    return sequences


def make_blueprints() -> list[Blueprint]:
    # Slot distribution is frozen before axis assignment.
    slots: list[tuple[int, str]] = (
        [(3, "progressive")] * 10 + [(4, "progressive")] * 8
        + [(3, "repeated_axis")] * 2 + [(4, "repeated_axis")] * 2 + [(5, "repeated_axis")] * 2
        + [(3, "rollback")] + [(4, "rollback")] * 2 + [(5, "rollback")] * 3
    )
    # Harder five-turn slots first makes the exact-quota search nearly instant.
    ordered = sorted(enumerate(slots), key=lambda item: (-item[1][0], item[0]))
    memo: set[tuple[int, tuple[int, ...]]] = set()

    def solve(index: int, totals: tuple[int, ...]) -> list[tuple[str, ...]] | None:
        state = (index, totals)
        if state in memo:
            return None
        if index == len(ordered):
            return [] if totals == (28, 28, 28, 28) else None
        length, kind = ordered[index][1]
        remaining_turns = sum(item[1][0] for item in ordered[index + 1 :])
        candidates = candidate_sequences(kind, length)
        candidates.sort(key=lambda seq: (
            sum((totals[i] + Counter(seq)[axis] - 28) ** 2 for i, axis in enumerate(AXES)), seq
        ))
        for seq in candidates:
            counts = Counter(seq)
            updated = tuple(totals[i] + counts[axis] for i, axis in enumerate(AXES))
            if any(value > 28 or value + remaining_turns < 28 for value in updated):
                continue
            tail = solve(index + 1, updated)
            if tail is not None:
                return [seq] + tail
        memo.add(state)
        return None

    solution = solve(0, (0, 0, 0, 0))
    if solution is None:
        raise RuntimeError("could not construct exact axis-balanced blueprints")
    restored: dict[int, tuple[str, ...]] = {ordered[i][0]: seq for i, seq in enumerate(solution)}
    result = []
    for index, (length, kind) in enumerate(slots, 1):
        seq = restored[index - 1]
        rollback_turn = length if kind == "rollback" else None
        result.append(Blueprint(index, kind, length, seq, rollback_turn))
    validate_blueprints(result)
    return result


def validate_blueprints(blueprints: list[Blueprint]) -> None:
    if len(blueprints) != TARGET_CHAINS:
        raise ValueError("blueprint count mismatch")
    if Counter(bp.length for bp in blueprints) != Counter(TARGET_LENGTHS):
        raise ValueError(f"length quota mismatch: {Counter(bp.length for bp in blueprints)}")
    if Counter(bp.chain_type for bp in blueprints) != Counter(TARGET_TYPES):
        raise ValueError(f"chain-type quota mismatch: {Counter(bp.chain_type for bp in blueprints)}")
    axes = Counter(axis for bp in blueprints for axis in bp.axes)
    if axes != Counter(TARGET_AXIS_COUNT):
        raise ValueError(f"axis quota mismatch: {axes}")
    for bp in blueprints:
        counts = Counter(bp.axes)
        if bp.chain_type == "progressive" and len(counts) != bp.length:
            raise ValueError(f"progressive axes repeat in {bp.chain_id}")
        if bp.chain_type in {"repeated_axis", "rollback"} and sorted(counts.values()) != [1] * (bp.length - 2) + [2]:
            raise ValueError(f"invalid repeated-axis structure in {bp.chain_id}")
        if bp.chain_type == "rollback" and bp.axes[-1] not in bp.axes[:-1]:
            raise ValueError(f"rollback final axis has no earlier state in {bp.chain_id}")


def load_tree() -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for entry in json.loads(TREE.read_text(encoding="utf-8")):
        match = re.fullmatch(r"ivebenchdb_mp4_(short|long)/(.+\.mp4)", entry.get("path", ""))
        if match and entry.get("type") == "file":
            result[(match.group(1), match.group(2))] = entry
    return result


def evaluated_sources() -> tuple[set[str], set[tuple[str, str]], set[str]]:
    hashes: set[str] = set()
    ids: set[tuple[str, str]] = set()
    names: set[str] = set()
    manifests = [
        IVE / "artifacts/short_subject/short_subject_manifest.csv",
        IVE / "artifacts/long_subject/manifest.csv",
        IVE / "artifacts/attribute_all/manifest.csv",
    ]
    for path in manifests:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for field in ("official_source_sha256", "source_sha256"):
                    if row.get(field):
                        hashes.add(row[field])
                subset = row.get("subset", "")
                if subset and row.get("official_id"):
                    ids.add((subset, row["official_id"]))
                for field in ("official_source_path", "source_path"):
                    if row.get(field):
                        names.add(Path(row[field]).name)
    return hashes, ids, names


def candidate_rows() -> list[dict[str, Any]]:
    tree = load_tree()
    used_hashes, used_ids, used_names = evaluated_sources()
    result = []
    for subset, prompt_path in (("short", SHORT_PROMPTS), ("long", LONG_PROMPTS)):
        for row in json.loads(prompt_path.read_text(encoding="utf-8")):
            caption = row["source_prompt"]
            entry = tree[(subset, row["src_video_name"])]
            digest = entry["lfs"]["oid"]
            if digest in used_hashes or (subset, str(row["id"])) in used_ids or row["src_video_name"] in used_names:
                continue
            if not PERSON_RE.search(caption) or MINOR_RE.search(caption) or NONREAL_RE.search(caption):
                continue
            enriched = dict(row)
            enriched.update({"subset": subset, "official_sha256": digest, "official_bytes": int(entry["size"]), "tree_path": entry["path"]})
            result.append(enriched)
    # Short is always exhausted before Long.  Within a subset the seeded digest is stable.
    result.sort(key=lambda row: (row["subset"] != "short", hashlib.sha256(f"{SEED}:{row['subset']}:{row['id']}".encode()).hexdigest()))
    return result


def source_path(row: dict[str, Any]) -> Path:
    return OFFICIAL / row["subset"] / row["src_video_name"]


def validate_official(path: Path, row: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != row["official_bytes"]:
        raise ValueError(f"official source size mismatch: {path}")
    if sha256_file(path) != row["official_sha256"]:
        raise ValueError(f"official source hash mismatch: {path}")


def download_source(row: dict[str, Any]) -> Path:
    destination = source_path(row)
    if destination.exists():
        validate_official(destination, row)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".mp4.part")
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{row['tree_path']}"
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with partial.open(mode) as stream:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if block:
                                stream.write(block)
            if partial.stat().st_size != row["official_bytes"] or sha256_file(partial) != row["official_sha256"]:
                raise ValueError("incomplete or corrupt IVEBench download")
            partial.replace(destination)
            return destination
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def contact_sheet(video: Path, output: Path) -> tuple[str, dict[str, Any]]:
    reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
    if len(reader) < 3:
        raise ValueError(f"too few frames: {video}")
    indices = [round((len(reader) - 1) * fraction) for fraction in (0.1, 0.5, 0.9)]
    frames = []
    for frame, label in zip(reader.get_batch(indices).asnumpy(), ("10%", "50%", "90%")):
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image.width, image.height + 30), "black")
        canvas.paste(image, (0, 30))
        ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
        frames.append(canvas)
    height = max(frame.height for frame in frames)
    sheet = Image.new("RGB", (sum(frame.width for frame in frames), height), "black")
    x = 0
    for frame in frames:
        sheet.paste(frame, (x, 0)); x += frame.width
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.read_bytes()).decode("ascii")
    metadata = {"source_frames": len(reader), "source_fps": round(float(reader.get_avg_fps()), 6), "contact_indices": indices,
                "contact_sheet_sha256": sha256_file(output), "contact_sheet_bytes": output.stat().st_size}
    return f"data:image/jpeg;base64,{encoded}", metadata


def prompt_for(row: dict[str, Any], blueprint: Blueprint, feedback: str = "") -> str:
    repeated_note = {
        "progressive": "Each turn edits a different axis; later turns preserve all prior edits.",
        "repeated_axis": "One axis occurs twice. Its second edit must be a new compatible change, not a rollback.",
        "rollback": "The final turn must explicitly restore the earlier state on its repeated axis while preserving other edits.",
    }[blueprint.chain_type]
    return f"""You are curating a real-video benchmark for autoregressive multi-turn video editing.
Inspect the three chronological frames. Decide whether this is a REAL, clearly ADULT human video with exactly ONE unambiguous primary person suitable for all requested edit axes.

Source caption (may be imperfect): {row['source_prompt']}
Frozen chain type: {blueprint.chain_type}
Frozen axis sequence: {json.dumps(blueprint.axes)}
Required number of turns: {blueprint.length}
Rule: {repeated_note}

Suitability requirements:
- Reject animation, CGI characters, statues, robots, minors, crowds, two co-equal people, tiny/distant people, severe occlusion, or uncertain adulthood.
- appearance requires a visible face/head; outfit requires visible upper body; action requires enough visible body; camera requires a coherent scene.
- A single background bystander is allowed only when unmistakably secondary.

If suitable, produce concise English instructions. Every turn must make exactly ONE feasible edit on its frozen axis:
- appearance: hair, facial hair, makeup, age-neutral facial detail, or similarly local identity-preserving appearance.
- outfit: garment type, color, material, or accessory worn by the primary person.
- action: change the primary person's pose or simple action without changing identity or scene.
- camera: a modest shot size, pan/tilt, orbit, or tracking change that remains feasible from this video.
Do not change global visual style. Do not replace the person. Avoid text/logos and unsafe content.
Instructions must be grounded in what is visible, diverse, natural, and not copy stock benchmark wording.
For every turn, target_state is a short factual description of the resulting state on that axis. preserve_requirements lists earlier edited states that should persist.
history_reference_round is null except the final rollback turn; for that turn use 0 to restore the original source state, or an earlier round number whose state is being restored.
If unsuitable, return turns=[] and concise rejection_codes.
{('Previous response violations to correct: ' + feedback) if feedback else ''}"""


def response_schema(blueprint: Blueprint) -> dict[str, Any]:
    turn = {
        "type": "object", "additionalProperties": False,
        "required": ["round", "axis", "instruction", "target_state", "preserve_requirements", "history_reference_round"],
        "properties": {
            "round": {"type": "integer"},
            "axis": {"type": "string", "enum": list(AXES)},
            "instruction": {"type": "string"},
            "target_state": {"type": "string"},
            "preserve_requirements": {"type": "array", "items": {"type": "string"}},
            "history_reference_round": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["suitable", "rejection_codes", "real_human", "adult", "primary_person_count", "person_scale",
                     "face_visible", "upper_body_visible", "action_editable", "scene_type", "scene_tag", "scene_summary", "turns"],
        "properties": {
            "suitable": {"type": "boolean"},
            "rejection_codes": {"type": "array", "items": {"type": "string"}},
            "real_human": {"type": "boolean"}, "adult": {"type": "boolean"},
            "primary_person_count": {"type": "integer"},
            "person_scale": {"type": "string", "enum": ["tiny", "small", "medium", "large"]},
            "face_visible": {"type": "boolean"}, "upper_body_visible": {"type": "boolean"},
            "action_editable": {"type": "boolean"},
            "scene_type": {"type": "string", "enum": ["indoor", "outdoor", "mixed"]},
            "scene_tag": {"type": "string"}, "scene_summary": {"type": "string"},
            "turns": {"type": "array", "items": turn, "maxItems": blueprint.length},
        },
    }


def call_openrouter(key: str, row: dict[str, Any], blueprint: Blueprint, image_uri: str, feedback: str = "") -> tuple[dict, dict]:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt_for(row, blueprint, feedback)},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "benchmark_candidate", "strict": True,
                                                                      "schema": response_schema(blueprint)}},
        "provider": {"data_collection": "deny", "allow_fallbacks": False, "order": [EXPECTED_PROVIDER]},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://localhost/real-person-multiturn-benchmark", "X-Title": "RPMTE benchmark builder"}
    for attempt in range(1, 4):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                        json=payload, timeout=(30, 300))
            response.raise_for_status()
            raw = response.json()
            if raw.get("provider") != EXPECTED_PROVIDER:
                raise RuntimeError(f"unexpected provider: {raw.get('provider')!r}")
            parsed = json.loads(raw["choices"][0]["message"]["content"])
            # Qwen occasionally wraps an otherwise valid strict-schema object in
            # a singleton array. Treat that as harmless transport syntax.
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                raise ValueError(f"model JSON root must be an object, got {type(parsed).__name__}")
            return parsed, raw
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def reference_prompts() -> list[str]:
    prompts = []
    for path in (SHORT_PROMPTS, LONG_PROMPTS):
        prompts.extend(row["edit_prompt"] for row in json.loads(path.read_text(encoding="utf-8")))
    senior = json.loads(SENIOR_CHAINS.read_text(encoding="utf-8"))
    prompts.extend(turn["prompt"]["en"] for chain in senior for turn in chain["turns"])
    return prompts


def content_violations(result: dict[str, Any], blueprint: Blueprint, references: list[str], accepted: list[dict]) -> list[str]:
    problems = []
    if not result.get("suitable"):
        return problems
    if not (result.get("real_human") and result.get("adult")) or result.get("primary_person_count") != 1:
        problems.append("suitability flags contradict suitable=true")
    if result.get("person_scale") not in {"medium", "large"}:
        problems.append("primary person is too small")
    required = set(blueprint.axes)
    if "appearance" in required and not result.get("face_visible"):
        problems.append("appearance axis lacks a visible face")
    if "outfit" in required and not result.get("upper_body_visible"):
        problems.append("outfit axis lacks visible upper body")
    if "action" in required and not result.get("action_editable"):
        problems.append("action axis is not editable")
    turns = result.get("turns", [])
    if len(turns) != blueprint.length:
        problems.append(f"expected {blueprint.length} turns")
        return problems
    if [turn.get("round") for turn in turns] != list(range(1, blueprint.length + 1)):
        problems.append("round indices do not match 1..N")
    if tuple(turn.get("axis") for turn in turns) != blueprint.axes:
        problems.append("axis sequence differs from frozen blueprint")
    for index, turn in enumerate(turns, 1):
        instruction = str(turn.get("instruction", "")).strip()
        if not instruction or len(instruction.split()) < 4:
            problems.append(f"round {index} instruction is empty or too short")
        if any(jaccard(instruction, ref) > 0.75 for ref in references):
            problems.append(f"round {index} is too similar to a training/official prompt")
        for prior in accepted:
            if any(jaccard(instruction, old["instruction"]) > 0.75 for old in prior["result"]["turns"]):
                problems.append(f"round {index} duplicates an accepted benchmark instruction")
        reference = turn.get("history_reference_round")
        if blueprint.chain_type == "rollback" and index == blueprint.length:
            if reference is None or not isinstance(reference, int) or reference < 0 or reference >= index:
                problems.append("final rollback has an invalid history reference")
        elif reference is not None:
            problems.append(f"round {index} has an unexpected history reference")
    return problems


def response_cost(raw: dict[str, Any]) -> float:
    usage = raw.get("usage") or {}
    for value in (usage.get("cost"), raw.get("cost")):
        if value is not None:
            return float(value)
    return 0.0


def normalize_result(result: dict[str, Any], blueprint: Blueprint) -> dict[str, Any]:
    """Canonicalize a harmless Qwen null representation without changing semantics."""
    normalized = json.loads(json.dumps(result))
    changes = []
    for turn in normalized.get("turns", []):
        is_final_rollback = blueprint.chain_type == "rollback" and turn.get("round") == blueprint.length
        if not is_final_rollback and turn.get("history_reference_round") == 0:
            turn["history_reference_round"] = None
            changes.append(turn.get("round"))
    if changes:
        normalized["normalizations"] = {"zero_history_reference_to_null_rounds": changes}
    return normalized


def normalize_video(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(source), "-map", "0:v:0",
               "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(temporary)]
    subprocess.run(command, check=True)
    temporary.replace(destination)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(destination),
                    "-map", "0:v:0", "-f", "null", "-"], check=True)
    reader = VideoReader(str(destination), ctx=cpu(0), num_threads=2)
    first = reader[0]
    header = destination.read_bytes()[:2_000_000]
    if not (header.find(b"moov") >= 0 and (header.find(b"mdat") < 0 or header.find(b"moov") < header.find(b"mdat"))):
        raise ValueError(f"faststart verification failed: {destination}")
    return {"benchmark_media": destination.relative_to(ROOT).as_posix(), "media_sha256": sha256_file(destination),
            "media_bytes": destination.stat().st_size, "media_frames": len(reader),
            "media_fps": round(float(reader.get_avg_fps()), 6), "media_height": int(first.shape[0]),
            "media_width": int(first.shape[1]), "codec": "h264", "pixel_format": "yuv420p",
            "faststart": True, "full_decode_ok": True}


def materialize(accepted: list[dict[str, Any]], blueprints: list[Blueprint], total_cost: float,
                reserve_accepts: int = 0) -> None:
    chains = []
    turns_csv = []
    inventory = []
    for bp, accepted_row in zip(blueprints, accepted, strict=True):
        row, result = accepted_row["source"], accepted_row["result"]
        media_path = ROOT / "media" / f"{bp.chain_id}_source.mp4"
        media = normalize_video(source_path(row), media_path)
        turns = []
        for turn in result["turns"]:
            normalized = dict(turn)
            normalized["instruction"] = normalized["instruction"].strip()
            turns.append(normalized)
            turns_csv.append({"chain_id": bp.chain_id, "round": turn["round"], "chain_type": bp.chain_type,
                              "axis": turn["axis"], "instruction": normalized["instruction"],
                              "target_state": turn["target_state"],
                              "preserve_requirements": " | ".join(turn["preserve_requirements"]),
                              "history_reference_round": "" if turn["history_reference_round"] is None else turn["history_reference_round"]})
        chain = {"benchmark_version": "real_person_multiturn_benchmark_v1", "chain_id": bp.chain_id,
                 "chain_type": bp.chain_type, "source": {"dataset": "IVEBench", "dataset_revision": DATASET_REVISION,
                 "subset": row["subset"], "official_id": int(row["id"]), "official_video_name": row["src_video_name"],
                 "official_sha256": row["official_sha256"], "official_source_caption": row["source_prompt"],
                 "media": media["benchmark_media"]}, "scene": {key: result[key] for key in ("scene_type", "scene_tag", "scene_summary")},
                 "turns": turns}
        chains.append(chain)
        inventory.append({"chain_id": bp.chain_id, "dataset": "IVEBench", "dataset_revision": DATASET_REVISION,
                          "subset": row["subset"], "official_id": row["id"], "official_video_name": row["src_video_name"],
                          "official_source_caption": row["source_prompt"], "official_sha256": row["official_sha256"],
                          "official_bytes": row["official_bytes"], **media})
    with (ROOT / "benchmark.jsonl").open("w", encoding="utf-8") as stream:
        for chain in chains:
            stream.write(json.dumps(chain, ensure_ascii=False) + "\n")
    write_csv(ROOT / "turns.csv", turns_csv)
    write_csv(ROOT / "source_inventory.csv", inventory)
    atomic_json(ROOT / "metadata/api_summary.json", {"model": MODEL, "required_provider": EXPECTED_PROVIDER,
                "credential_env": "LIJUYANG_ICLR", "personal_credential_used": False, "data_collection": "deny",
                "temperature": 0, "hard_budget_usd": API_BUDGET_USD, "reported_cost_usd": round(total_cost, 8),
                "accepted_chains": len(chains), "reserve_accepts": reserve_accepts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate sources/quotas without downloads or API calls")
    args = parser.parse_args()
    blueprints = make_blueprints()
    candidates = candidate_rows()
    atomic_json(ROOT / "metadata/blueprints.json", [bp.__dict__ | {"axes": list(bp.axes), "chain_id": bp.chain_id} for bp in blueprints])
    preflight = {"blueprints": len(blueprints), "turns": sum(bp.length for bp in blueprints),
                 "lengths": dict(Counter(bp.length for bp in blueprints)),
                 "chain_types": dict(Counter(bp.chain_type for bp in blueprints)),
                 "axes": dict(Counter(axis for bp in blueprints for axis in bp.axes)),
                 "candidate_count": len(candidates), "short_candidates": sum(row["subset"] == "short" for row in candidates),
                 "long_candidates": sum(row["subset"] == "long" for row in candidates)}
    atomic_json(ROOT / "metadata/preflight.json", preflight)
    print(json.dumps(preflight, indent=2))
    if args.dry_run:
        return
    key = os.environ.get("LIJUYANG_ICLR")
    if not key:
        raise RuntimeError("LIJUYANG_ICLR is not set; personal credentials are intentionally unsupported")
    references = reference_prompts()
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    total_cost = 0.0
    # Raw records are immutable/uniquely named, so their sum is the most
    # accurate spend ledger even if a process stopped before writing state.
    for raw_path in (ROOT / "metadata/api_raw").glob("*.json"):
        total_cost += response_cost(json.loads(raw_path.read_text()))
    state_path = ROOT / "metadata/build_state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("status") == "benchmark_materialized":
            print("benchmark is already materialized")
            return
        attempts = previous.get("attempts", [])
        total_cost = max(total_cost, float(previous.get("reported_cost_usd", 0.0)))
        # Replay saved records under the current validator. A recovered record
        # may fill only the exact blueprint slot for which it was generated.
        for record in attempts:
            expected = blueprints[len(accepted) % len(blueprints)]
            record["accepted"] = False
            if int(record["blueprint"]["slot"]) != expected.slot:
                continue
            for item in reversed(record.get("api_attempts", [])):
                result = item.get("result")
                if not isinstance(result, dict) or not result.get("suitable"):
                    continue
                result = normalize_result(result, expected)
                violations = content_violations(result, expected, references, accepted)
                item["result"], item["violations"] = result, violations
                if not violations:
                    record["accepted"] = True
                    accepted.append({"source": record["source"], "result": result,
                                     "media_probe": record["media_probe"], "api_record": item})
                    break
    processed = {(record["source"]["subset"], str(record["source"]["id"])) for record in attempts}
    target_accepts = TARGET_CHAINS + RESERVE_ACCEPTS
    for candidate_index, row in enumerate(candidates, 1):
        if len(accepted) >= target_accepts:
            break
        if (row["subset"], str(row["id"])) in processed:
            continue
        blueprint = blueprints[len(accepted) % len(blueprints)]
        video = download_source(row)
        sheet_path = ROOT / "metadata/contact_sheets" / f"{row['subset']}_{int(row['id']):04d}.jpg"
        image_uri, media_meta = contact_sheet(video, sheet_path)
        record: dict[str, Any] = {"candidate_index": candidate_index, "source": row, "blueprint": blueprint.__dict__ | {"axes": list(blueprint.axes)},
                                  "media_probe": media_meta, "api_attempts": []}
        final_result = None
        for content_try in range(2):
            feedback = "" if content_try == 0 else "; ".join(record["api_attempts"][-1]["violations"])
            try:
                result, raw = call_openrouter(key, row, blueprint, image_uri, feedback)
            except Exception as error:
                record["api_error"] = f"{type(error).__name__}: {error}"
                break
            cost = response_cost(raw); total_cost += cost
            raw_path = ROOT / "metadata/api_raw" / f"{row['subset']}_{int(row['id']):04d}_try{content_try + 1}.json"
            suffix = 1
            while raw_path.exists():
                raw_path = ROOT / "metadata/api_raw" / f"{row['subset']}_{int(row['id']):04d}_try{content_try + 1}_rerun{suffix}.json"
                suffix += 1
            atomic_json(raw_path, raw)
            result = normalize_result(result, blueprint)
            violations = content_violations(result, blueprint, references, accepted)
            record["api_attempts"].append({"try": content_try + 1, "provider": raw.get("provider"), "model": raw.get("model"),
                                           "generation_id": raw.get("id"), "cost_usd": cost, "raw_record": raw_path.relative_to(ROOT).as_posix(),
                                           "result": result, "violations": violations})
            if total_cost > API_BUDGET_USD:
                atomic_json(ROOT / "metadata/build_state.json", {"status": "budget_exceeded", "reported_cost_usd": total_cost,
                            "accepted": len(accepted), "attempts": attempts + [record]})
                raise RuntimeError(f"OpenRouter hard budget exceeded: ${total_cost:.4f} > ${API_BUDGET_USD:.2f}")
            if not result.get("suitable"):
                break
            if not violations:
                final_result = result
                break
        record["accepted"] = final_result is not None
        attempts.append(record)
        if final_result is not None:
            accepted.append({"source": row, "result": final_result, "media_probe": media_meta,
                             "api_record": record["api_attempts"][-1]})
        atomic_json(ROOT / "metadata/build_state.json", {"status": "screening", "reported_cost_usd": total_cost,
                    "accepted": len(accepted), "target_accepts": target_accepts, "attempts": attempts})
        print(f"[{candidate_index}] {row['subset']}:{row['id']} accepted={record['accepted']} total={len(accepted)}/{target_accepts} cost=${total_cost:.5f}", flush=True)
    if len(accepted) < TARGET_CHAINS:
        atomic_json(ROOT / "metadata/shortage_report.json", {"status": "insufficient_high_quality_candidates",
                    "accepted": len(accepted), "required": TARGET_CHAINS, "screened": len(attempts), "quality_threshold_lowered": False})
        raise RuntimeError(f"only {len(accepted)} acceptable candidates; quality threshold was not lowered")
    reserves = accepted[TARGET_CHAINS:TARGET_CHAINS + RESERVE_ACCEPTS]
    atomic_json(ROOT / "metadata/reserve_candidates.json", [{
        "subset": item["source"]["subset"], "official_id": int(item["source"]["id"]),
        "official_video_name": item["source"]["src_video_name"],
        "official_sha256": item["source"]["official_sha256"],
        "source_caption": item["source"]["source_prompt"],
        "screening_result": item["result"],
    } for item in reserves])
    materialize(accepted[:TARGET_CHAINS], blueprints, total_cost, reserve_accepts=len(reserves))
    atomic_json(ROOT / "metadata/build_state.json", {"status": "benchmark_materialized", "reported_cost_usd": total_cost,
                "accepted": len(accepted), "benchmark_chains": TARGET_CHAINS, "attempts": attempts})
    print(f"materialized {TARGET_CHAINS} chains; accepted reserve={max(0, len(accepted)-TARGET_CHAINS)}; cost=${total_cost:.5f}")


if __name__ == "__main__":
    main()
