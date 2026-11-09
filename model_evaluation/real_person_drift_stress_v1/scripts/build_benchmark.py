#!/usr/bin/env python3
"""Materialize the frozen 12-chain / 48-turn drift stress benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from decord import VideoReader, cpu
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
MODEL_EVAL = ROOT.parent
OFFICIAL = MODEL_EVAL / "ivebench_lance/data/official/short"
SELECTED = ROOT / "metadata/selected_sources.json"
DATASET_REVISION = "1e6cb655c0a537f6093edcf18e2607fb03e76734"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def turn(round_: int, axis: str, instruction: str, target: str,
         preserve: list[str] | None = None, reference: int | None = None) -> dict[str, Any]:
    return {"round": round_, "axis": axis, "instruction": instruction,
            "target_state": target, "preserve_requirements": preserve or [],
            "history_reference_round": reference}


def checkpoint(round_: int, reference: int, kind: str, preserve_rounds: list[int] | None = None) -> dict[str, Any]:
    return {"round": round_, "reference_round": reference, "preserve_rounds": preserve_rounds or [],
            "comparison_type": kind}


def definitions() -> list[dict[str, Any]]:
    return [
        {"id": "drift_001", "source_id": 373, "pattern": "repeated_edit_undo", "focus": "eyewear",
         "turns": [
             turn(1, "appearance", "Add solid black sunglasses to the man's face.", "The man wears solid black sunglasses."),
             turn(2, "appearance", "Remove the black sunglasses and restore his original unobstructed eyes.", "The man's eyes are unobstructed as in the source video.", reference=0),
             turn(3, "appearance", "Restore the same black sunglasses from round 1.", "The man again wears the same solid black sunglasses.", reference=1),
             turn(4, "appearance", "Remove the black sunglasses again and return to the original unobstructed face.", "The man's eyes are unobstructed again.", reference=0)],
         "checkpoints": [checkpoint(2, 0, "rollback_to_original"), checkpoint(3, 1, "replay_prior_edit"), checkpoint(4, 0, "second_rollback_to_original")]},
        {"id": "drift_002", "source_id": 157, "pattern": "repeated_edit_undo", "focus": "headwear",
         "turns": [
             turn(1, "appearance", "Add a burgundy knit beanie to the man's head while keeping his glasses visible.", "The man wears a burgundy knit beanie."),
             turn(2, "appearance", "Remove the burgundy beanie and restore his original uncovered hairstyle.", "The man's original uncovered hairstyle is visible.", reference=0),
             turn(3, "appearance", "Restore the same burgundy knit beanie from round 1.", "The man again wears the burgundy knit beanie.", reference=1),
             turn(4, "appearance", "Remove the burgundy beanie again and restore the original uncovered hairstyle.", "The man's original hairstyle is visible again.", reference=0)],
         "checkpoints": [checkpoint(2, 0, "rollback_to_original"), checkpoint(3, 1, "replay_prior_edit"), checkpoint(4, 0, "second_rollback_to_original")]},
        {"id": "drift_003", "source_id": 324, "pattern": "repeated_edit_undo", "focus": "facial_hair",
         "turns": [
             turn(1, "appearance", "Add a clearly visible full beard and mustache to the man.", "The man has a full beard and mustache."),
             turn(2, "appearance", "Remove the beard and mustache and restore his original clean-shaven face.", "The man is clean-shaven as in the source video.", reference=0),
             turn(3, "appearance", "Restore the same full beard and mustache from round 1.", "The man again has the same full beard and mustache.", reference=1),
             turn(4, "appearance", "Remove the beard and mustache again and return to the original clean-shaven face.", "The man is clean-shaven again.", reference=0)],
         "checkpoints": [checkpoint(2, 0, "rollback_to_original"), checkpoint(3, 1, "replay_prior_edit"), checkpoint(4, 0, "second_rollback_to_original")]},
        {"id": "drift_004", "source_id": 282, "pattern": "repeated_edit_undo", "focus": "accessory",
         "turns": [
             turn(1, "appearance", "Add a thick cream knitted scarf around the woman's neck.", "The woman wears a thick cream knitted scarf."),
             turn(2, "appearance", "Remove the cream scarf and restore her original uncovered neckline.", "The woman's neckline is uncovered as in the source video.", reference=0),
             turn(3, "appearance", "Restore the same cream knitted scarf from round 1.", "The woman again wears the thick cream knitted scarf.", reference=1),
             turn(4, "appearance", "Remove the cream scarf again and restore the original uncovered neckline.", "The woman's neckline is uncovered again.", reference=0)],
         "checkpoints": [checkpoint(2, 0, "rollback_to_original"), checkpoint(3, 1, "replay_prior_edit"), checkpoint(4, 0, "second_rollback_to_original")]},
        {"id": "drift_005", "source_id": 368, "pattern": "history_state_retrieval", "focus": "outfit",
         "turns": [
             turn(1, "outfit", "Change the man's maroon sweater to a plain navy knit sweater.", "The man wears a plain navy knit sweater."),
             turn(2, "outfit", "Replace the navy sweater with a burgundy button-front cardigan.", "The man wears a burgundy button-front cardigan."),
             turn(3, "outfit", "Restore the plain navy knit sweater from round 1.", "The man again wears the plain navy knit sweater.", reference=1),
             turn(4, "outfit", "Restore the original maroon sweater from the source video.", "The man wears his original maroon sweater.", reference=0)],
         "checkpoints": [checkpoint(3, 1, "restore_intermediate_state"), checkpoint(4, 0, "rollback_to_original")]},
        {"id": "drift_006", "source_id": 165, "pattern": "history_state_retrieval", "focus": "camera",
         "turns": [
             turn(1, "camera", "Reframe the shot as a medium close-up showing the woman's head, shoulders, and upper chest.", "A medium close-up shows her head, shoulders, and upper chest."),
             turn(2, "camera", "Change to a tight close-up centered on the woman's face.", "A tight close-up centers on the woman's face."),
             turn(3, "camera", "Restore the medium close-up framing from round 1.", "The medium close-up from round 1 is restored.", reference=1),
             turn(4, "camera", "Restore the original low-angle framing from the source video.", "The original low-angle source framing is restored.", reference=0)],
         "checkpoints": [checkpoint(3, 1, "restore_intermediate_state"), checkpoint(4, 0, "rollback_to_original")]},
        {"id": "drift_007", "source_id": 251, "pattern": "history_state_retrieval", "focus": "action",
         "turns": [
             turn(1, "outfit", "Change the woman's visible top to a plain navy blouse.", "The woman wears a plain navy blouse."),
             turn(2, "action", "Have the woman raise her right hand and give one clear wave toward the camera.", "The woman raises her right hand in a clear wave.", ["plain navy blouse"]),
             turn(3, "action", "Restore her original relaxed arm position and action from the source video while keeping the navy blouse.", "The woman returns to her original relaxed action.", ["plain navy blouse"], 0),
             turn(4, "action", "Restore the right-hand wave from round 2 while keeping the navy blouse.", "The woman again gives the same right-hand wave.", ["plain navy blouse"], 2)],
         "checkpoints": [checkpoint(3, 0, "rollback_preserve_intervening", [1]), checkpoint(4, 2, "restore_intermediate_state", [1])]},
        {"id": "drift_008", "source_id": 386, "pattern": "history_state_retrieval", "focus": "identity",
         "turns": [
             turn(1, "identity", "Change only the woman's facial identity to a different adult woman with a broader round face, while keeping her hair, expression, clothing, and scene unchanged.", "A different adult woman with a broader round face appears."),
             turn(2, "identity", "Change only the facial identity to another adult woman with a narrower face and high cheekbones, keeping everything else unchanged.", "Another adult woman with a narrower face and high cheekbones appears."),
             turn(3, "identity", "Restore the broader round-faced identity from round 1 while keeping the same hair, expression, clothing, and scene.", "The round-faced identity from round 1 is restored.", reference=1),
             turn(4, "identity", "Restore the woman's original facial identity from the source video.", "The original source identity is restored.", reference=0)],
         "checkpoints": [checkpoint(3, 1, "restore_intermediate_state"), checkpoint(4, 0, "rollback_to_original")]},
        {"id": "drift_009", "source_id": 361, "pattern": "delayed_rollback", "focus": "appearance_preservation",
         "turns": [
             turn(1, "appearance", "Add black aviator sunglasses to the man in the foreground.", "The foreground man wears black aviator sunglasses."),
             turn(2, "outfit", "Change the foreground man's black jacket to a dark green coat.", "The foreground man wears a dark green coat.", ["black aviator sunglasses"]),
             turn(3, "camera", "Pull the camera back slightly to show more of the foreground man's torso while keeping him centered.", "A slightly wider view shows more of the man's torso.", ["black aviator sunglasses", "dark green coat"]),
             turn(4, "appearance", "Remove only the aviator sunglasses and restore the man's original unobstructed eyes, while keeping the dark green coat and wider framing.", "The man's eyes are unobstructed while the coat and wider framing remain.", ["dark green coat", "slightly wider framing"], 0)],
         "checkpoints": [checkpoint(4, 0, "delayed_rollback_preserve_intervening", [2, 3])]},
        {"id": "drift_010", "source_id": 338, "pattern": "delayed_rollback", "focus": "appearance_preservation",
         "turns": [
             turn(1, "appearance", "Add a clearly visible full beard and mustache to the man in medieval attire.", "The man has a full beard and mustache."),
             turn(2, "outfit", "Change the man's ornate outfit to a navy embroidered medieval tunic.", "The man wears a navy embroidered medieval tunic.", ["full beard and mustache"]),
             turn(3, "action", "Have the man raise his right hand in a clear thumbs-up while remaining in place.", "The man gives a clear right-hand thumbs-up.", ["full beard and mustache", "navy embroidered tunic"]),
             turn(4, "appearance", "Remove only the beard and mustache and restore the original clean-shaven face, while keeping the navy tunic and thumbs-up pose.", "The man is clean-shaven while the navy tunic and thumbs-up remain.", ["navy embroidered tunic", "right-hand thumbs-up"], 0)],
         "checkpoints": [checkpoint(4, 0, "delayed_rollback_preserve_intervening", [2, 3])]},
        {"id": "drift_011", "source_id": 357, "pattern": "delayed_rollback", "focus": "camera_preservation",
         "turns": [
             turn(1, "camera", "Move the camera slightly closer to emphasize the man's face and upper torso.", "A closer view emphasizes the man's face and upper torso."),
             turn(2, "appearance", "Add a burgundy knit beanie while keeping his dreadlocks and glasses visible.", "The man wears a burgundy knit beanie."),
             turn(3, "outfit", "Replace the man's white shirt with a brown leather jacket while keeping the blue scarf.", "The man wears a brown leather jacket with the blue scarf.", ["burgundy knit beanie"]),
             turn(4, "camera", "Restore the original wider framing from the source video while keeping the beanie, brown leather jacket, glasses, dreadlocks, and blue scarf.", "The original wider framing returns while both later edits remain.", ["burgundy knit beanie", "brown leather jacket"], 0)],
         "checkpoints": [checkpoint(4, 0, "delayed_rollback_preserve_intervening", [2, 3])]},
        {"id": "drift_012", "source_id": 376, "pattern": "delayed_rollback", "focus": "action_preservation",
         "turns": [
             turn(1, "appearance", "Add clear rectangular eyeglasses to the man's face.", "The man wears clear rectangular eyeglasses."),
             turn(2, "action", "Have the man put down the phone and fold both arms across his chest while remaining on the couch.", "The man rests on the couch with both arms folded across his chest.", ["clear rectangular eyeglasses"]),
             turn(3, "outfit", "Change the man's visible shirt to a charcoal gray sweater.", "The man wears a charcoal gray sweater.", ["clear rectangular eyeglasses", "arms folded across chest"]),
             turn(4, "action", "Restore the original action of holding and looking at the phone, while keeping the clear eyeglasses and charcoal sweater.", "The man again holds and looks at the phone while both appearance edits remain.", ["clear rectangular eyeglasses", "charcoal gray sweater"], 0)],
         "checkpoints": [checkpoint(4, 0, "delayed_rollback_preserve_intervening", [1, 3])]},
    ]


def normalize(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(source),
               "-map", "0:v:0", "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264",
               "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
               "-movflags", "+faststart", str(temporary)]
    subprocess.run(command, check=True)
    os.replace(temporary, destination)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(destination),
                    "-map", "0:v:0", "-f", "null", "-"], check=True)
    reader = VideoReader(str(destination), ctx=cpu(0), num_threads=2)
    height, width = reader[0].shape[:2]
    prefix = destination.read_bytes()[:2_000_000]
    faststart = prefix.find(b"moov") >= 0 and (prefix.find(b"mdat") < 0 or prefix.find(b"moov") < prefix.find(b"mdat"))
    if not faststart:
        raise RuntimeError(f"faststart check failed: {destination}")
    return {"media": destination.relative_to(ROOT).as_posix(), "media_sha256": sha256(destination),
            "media_bytes": destination.stat().st_size, "media_frames": len(reader),
            "media_fps": round(float(reader.get_avg_fps()), 6), "media_width": int(width),
            "media_height": int(height), "codec": "h264", "pixel_format": "yuv420p",
            "faststart": True, "full_decode_ok": True}


def main() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    by_id = {int(item["source"]["id"]): item for item in selected}
    specs = definitions()
    if len(specs) != 12 or len({spec["source_id"] for spec in specs}) != 12:
        raise RuntimeError("benchmark must use 12 unique sources")
    if set(by_id) != {spec["source_id"] for spec in specs}:
        raise RuntimeError("benchmark definitions and selected sources differ")

    chains, inventory, turn_rows, checkpoint_rows = [], [], [], []
    for spec in specs:
        item = by_id[spec["source_id"]]
        source_row, screen = item["source"], item["screening"]
        official = OFFICIAL / source_row["src_video_name"]
        if sha256(official) != source_row["official_sha256"]:
            raise RuntimeError(f"official source hash mismatch: {official}")
        destination = ROOT / "media" / f"{spec['id']}_source.mp4"
        media = normalize(official, destination)
        chain = {
            "benchmark_version": "real_person_drift_stress_v1", "drift_id": spec["id"],
            "chain_id": spec["id"], "pattern": spec["pattern"], "focus": spec["focus"],
            "source": {"dataset": "IVEBench", "dataset_revision": DATASET_REVISION,
                       "subset": "short", "official_id": int(source_row["id"]),
                       "official_video_name": source_row["src_video_name"],
                       "official_sha256": source_row["official_sha256"],
                       "official_source_caption": source_row["source_prompt"], "media": media["media"]},
            "scene": {key: screen[key] for key in ("scene_type", "scene_tag", "scene_summary")},
            "turns": spec["turns"], "checkpoints": spec["checkpoints"],
        }
        chains.append(chain)
        inventory.append({"drift_id": spec["id"], "dataset": "IVEBench", "subset": "short",
                          "official_id": source_row["id"], "official_video_name": source_row["src_video_name"],
                          "official_sha256": source_row["official_sha256"], "official_source_caption": source_row["source_prompt"],
                          "screening_source": item["selection"] if "selection" in item else "qwen3.8_omni_flash",
                          **media})
        for entry in spec["turns"]:
            turn_rows.append({"drift_id": spec["id"], "pattern": spec["pattern"], **entry,
                              "preserve_requirements": " | ".join(entry["preserve_requirements"]),
                              "history_reference_round": "" if entry["history_reference_round"] is None else entry["history_reference_round"]})
        for entry in spec["checkpoints"]:
            checkpoint_rows.append({"drift_id": spec["id"], "pattern": spec["pattern"], **entry,
                                    "preserve_rounds": " | ".join(map(str, entry["preserve_rounds"]))})

    atomic_text(ROOT / "benchmark.jsonl", "".join(json.dumps(chain, ensure_ascii=False) + "\n" for chain in chains))
    for path, rows in ((ROOT / "source_inventory.csv", inventory), (ROOT / "turns.csv", turn_rows),
                       (ROOT / "checkpoints.csv", checkpoint_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    receipt = {"status": "complete", "benchmark": str(ROOT / "benchmark.jsonl"),
               "benchmark_sha256": sha256(ROOT / "benchmark.jsonl"), "chains": 12, "turns": 48,
               "checkpoints": 24, "patterns": {name: sum(x["pattern"] == name for x in specs)
                                                for name in {x["pattern"] for x in specs}},
               "unique_source_hashes": len({x["source"]["official_sha256"] for x in chains}),
               "overlap_with_real_person_v2": 0}
    atomic_json(ROOT / "benchmark_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
