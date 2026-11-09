#!/usr/bin/env python3
"""Freeze the non-rollback, no-camera IVEBench History-ViT benchmark.

Every surviving turn after a removed camera turn is listed explicitly in
REWRITES.  This intentionally avoids a keyword-based rewrite that could alter
the semantic edit target.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
MODEL_EVALUATION = ROOT.parent
SOURCE_ROOT = MODEL_EVALUATION / "real_person_multiturn_benchmark_v2"
SOURCE_BENCHMARK = SOURCE_ROOT / "benchmark.jsonl"
SOURCE_BENCHMARK_SHA256 = "1f9edd319806e546eef367d82facc14cdb2240b872ad41135e980cd2fe62998a"
OUTPUT = ROOT / "benchmark.json"
REWRITE_AUDIT = ROOT / "camera_dependency_rewrites.json"
EXPECTED_CHAIN_IDS = tuple(f"rpmte_{index:03d}" for index in range(1, 25))


# The 33 entries are all non-camera turns occurring after a removed camera
# turn.  Full replacement text is frozen even where only preserve_requirements
# changes, so each semantic decision remains human-readable and reviewable.
REWRITES: dict[tuple[str, int], dict] = {
    ("rpmte_002", 3): {
        "instruction": "Change the woman's jacket to a dark green coat.",
        "preserve_requirements": ["The woman is wearing black sunglasses."],
        "target_state": "The woman is wearing a dark green coat over her shirt.",
        "reason": "Removed the requirement to preserve the deleted zoom-out edit.",
    },
    ("rpmte_003", 3): {
        "instruction": "Have the man raise his left hand in a clear thumbs-up gesture.",
        "preserve_requirements": ["The man still wears dark rectangular eyeglasses."],
        "target_state": "The man gives a distinct left-hand thumbs-up while staying seated.",
        "reason": "Removed the requirement to preserve the deleted wider framing.",
    },
    ("rpmte_004", 3): {
        "instruction": "Change the person's jacket to a bright red color.",
        "preserve_requirements": ["The person is wearing a dark knit beanie."],
        "target_state": "The person is wearing a bright red jacket.",
        "reason": "Removed the requirement to preserve the deleted wider beach view.",
    },
    ("rpmte_005", 3): {
        "instruction": "Change the man’s suit jacket to a dark navy blazer while keeping the white shirt.",
        "preserve_requirements": ["The man wears thick tortoiseshell glasses."],
        "target_state": "The man wears a dark navy blazer over a white shirt and still holds the smartphone.",
        "reason": "Removed the requirement to preserve the deleted tighter medium shot.",
    },
    ("rpmte_006", 3): {
        "instruction": "Change the performer’s costume to a deep red velvet robe with gold embroidery along the sleeves and hem.",
        "preserve_requirements": ["large dark wide-brimmed stage hat"],
        "target_state": "The performer wears a deep red velvet robe with gold embroidery, still wearing the large dark wide-brimmed stage hat and playing the wind instrument.",
        "reason": "Removed the requirement to preserve the deleted closer waist-up framing.",
    },
    ("rpmte_007", 3): {
        "instruction": "Change the musician’s plain black shirt to a dark charcoal button-up while keeping the goatee, mustache, and white violin intact.",
        "preserve_requirements": ["goatee and mustache"],
        "target_state": "The musician wears a dark charcoal button-up shirt, retains the goatee and mustache, and continues playing the white violin.",
        "reason": "Removed all references to the deleted closer shot while preserving the outfit target.",
    },
    ("rpmte_009", 3): {
        "instruction": "Change her black textured outfit to a dark green leather jacket.",
        "preserve_requirements": ["She wears a black fedora hat."],
        "target_state": "She wears a dark green leather jacket.",
        "reason": "Removed the requirement to preserve the deleted wider upper-body framing.",
    },
    ("rpmte_010", 2): {
        "instruction": "Change the gray turtleneck sweater to a dark navy crewneck sweater while keeping the same fit and texture.",
        "preserve_requirements": [],
        "target_state": "The man wears a dark navy crewneck sweater instead of the gray turtleneck.",
        "reason": "Removed the requirement to preserve the deleted first-round wider framing.",
    },
    ("rpmte_010", 3): {
        "instruction": "Have the seated man give a right-hand thumbs-up while remaining seated.",
        "preserve_requirements": ["The man wears a dark navy crewneck sweater instead of the gray turtleneck."],
        "target_state": "The man remains seated and gives a right-hand thumbs-up.",
        "reason": "Removed the requirement to preserve the deleted first-round wider framing.",
    },
    ("rpmte_011", 3): {
        "instruction": "Make the man cross both arms over his chest during the conversation, preserving the fuller beard and mustache.",
        "preserve_requirements": ["fuller beard and mustache"],
        "target_state": "The man is standing with both arms crossed over his chest while talking.",
        "reason": "Removed the deleted wider-view state from the action instruction and preservation list.",
    },
    ("rpmte_011", 4): {
        "instruction": "Change the man’s orange shirt to a dark green button-up shirt while keeping the fuller beard and mustache and the crossed-arm pose.",
        "preserve_requirements": ["fuller beard and mustache", "both arms crossed over his chest"],
        "target_state": "The man wears a dark green button-up shirt.",
        "reason": "Removed the deleted wider-view state from the outfit instruction and preservation list.",
    },
    ("rpmte_012", 3): {
        "instruction": "Have the woman fold both arms firmly across her torso while maintaining her concerned expression and smoky eye makeup.",
        "preserve_requirements": ["smoky eye makeup"],
        "target_state": "The woman is standing with both arms crossed over her torso.",
        "reason": "Removed the deleted wider-framing state from the action instruction and preservation list.",
    },
    ("rpmte_012", 4): {
        "instruction": "Change the woman’s black top to a dark navy hoodie while keeping her crossed arms and smoky eye makeup intact.",
        "preserve_requirements": ["smoky eye makeup", "crossed arms"],
        "target_state": "The woman is wearing a dark navy hoodie.",
        "reason": "Removed the deleted wider-framing state from the outfit instruction and preservation list.",
    },
    ("rpmte_013", 3): {
        "instruction": "Point toward a clearly visible ornate detail in the scene.",
        "preserve_requirements": ["The person wears a dark hood framing the silver mask."],
        "target_state": "The person points toward an ornate detail in the background.",
        "reason": "Removed the deleted wider upper-body view; 'framing the silver mask' describes the hood, not a camera state.",
    },
    ("rpmte_013", 4): {
        "instruction": "Change the outfit to a dark medieval tunic with subtle embroidery.",
        "preserve_requirements": ["The person wears a dark hood framing the silver mask.", "The person points toward an ornate detail in the background."],
        "target_state": "The person wears a dark medieval tunic with subtle embroidery.",
        "reason": "Removed the deleted wider upper-body view while retaining the hood and action states.",
    },
    ("rpmte_014", 3): {
        "instruction": "Change the man’s brown sweater to a dark green crewneck sweater.",
        "preserve_requirements": ["The man has a sharply outlined chin goatee."],
        "target_state": "The man is now wearing a dark green crewneck sweater.",
        "reason": "Removed the requirement to preserve the deleted wider desk view.",
    },
    ("rpmte_014", 4): {
        "instruction": "Raise his right hand and wave once.",
        "preserve_requirements": ["The man has a sharply outlined chin goatee.", "The man is now wearing a dark green crewneck sweater."],
        "target_state": "The man raises his right hand and waves once.",
        "reason": "Removed the requirement to preserve the deleted wider desk view.",
    },
    ("rpmte_015", 4): {
        "instruction": "Change the groom’s tie to a deep burgundy silk tie.",
        "preserve_requirements": ["full dark beard", "waving to the seated guests"],
        "target_state": "The groom wears a deep burgundy silk tie with his blue suit.",
        "reason": "Removed the requirement to preserve the deleted closer frontal angle.",
    },
    ("rpmte_016", 3): {
        "instruction": "Have the soldier extend one arm and point toward the ocean horizon.",
        "preserve_requirements": ["The soldier has a short beard with visible gray streaks."],
        "target_state": "The soldier is pointing toward the ocean horizon with one extended arm.",
        "reason": "Removed the requirement to preserve the deleted wider ship-deck view.",
    },
    ("rpmte_016", 4): {
        "instruction": "Replace the soldier’s uniform with a dark navy jacket featuring visible shoulder insignia.",
        "preserve_requirements": ["The soldier has a short beard with visible gray streaks.", "The soldier is pointing toward the ocean horizon with one extended arm."],
        "target_state": "The soldier is wearing a dark navy jacket with visible shoulder insignia.",
        "reason": "Removed the requirement to preserve the deleted wider ship-deck view.",
    },
    ("rpmte_017", 3): {
        "instruction": "Replace the man's dark shirt with a charcoal-gray button-up shirt.",
        "preserve_requirements": ["The man has shoulder-length wavy hair."],
        "target_state": "The man is wearing a charcoal-gray button-up shirt.",
        "reason": "Removed the requirement to preserve the deleted wider upper-torso view.",
    },
    ("rpmte_017", 4): {
        "instruction": "Have the man bring both forearms across his chest during the tense conversation.",
        "preserve_requirements": ["The man has shoulder-length wavy hair.", "The man is wearing a charcoal-gray button-up shirt."],
        "target_state": "The man crosses both arms over his chest while continuing the conversation.",
        "reason": "Removed the requirement to preserve the deleted wider upper-torso view.",
    },
    ("rpmte_018", 3): {
        "instruction": "Replace the black suit jacket with a dark navy blazer while keeping the white shirt and tie.",
        "preserve_requirements": ["The man has a short beard with a strongly defined mustache."],
        "target_state": "The man is wearing a dark navy blazer over a white shirt and tie.",
        "reason": "Removed the requirement to preserve the deleted wider outdoor view.",
    },
    ("rpmte_018", 4): {
        "instruction": "Lower the visible cigarette and hold it at his side.",
        "preserve_requirements": ["The man has a short beard with a strongly defined mustache.", "The man is wearing a dark navy blazer over a white shirt and tie."],
        "target_state": "The man holds the cigarette lowered at his side.",
        "reason": "Removed the requirement to preserve the deleted wider outdoor view.",
    },
    ("rpmte_020", 2): {
        "instruction": "Change his plain light shirt to a dark zip-up hoodie while keeping the same fit and seated pose.",
        "preserve_requirements": ["same room and desk setup", "same seated posture"],
        "target_state": "The man is wearing a dark zip-up hoodie at the desk.",
        "reason": "Removed the requirement to preserve the deleted first-round tighter side view.",
    },
    ("rpmte_020", 3): {
        "instruction": "Add a black baseball cap worn backward over his hair while keeping the dark zip-up hoodie unchanged.",
        "preserve_requirements": ["dark zip-up hoodie from round 2", "same room and desk setup", "same seated posture"],
        "target_state": "The man wears a backward black baseball cap with the dark zip-up hoodie.",
        "reason": "Removed the requirement to preserve the deleted first-round tighter side view.",
    },
    ("rpmte_021", 4): {
        "instruction": "Make her clap both hands twice in time with the music while keeping the high ponytail and headphones.",
        "preserve_requirements": ["sleek high ponytail", "over-ear headphones", "plain purple studio background"],
        "target_state": "The woman claps both hands twice in rhythm with a high ponytail and over-ear headphones.",
        "reason": "Removed the deleted wider waist-up state from the action instruction, preservation list, and target state.",
    },
    ("rpmte_023", 3): {
        "instruction": "Have the man point toward the wine glass on the table.",
        "preserve_requirements": ["short boxed beard"],
        "target_state": "The man is pointing toward the wine glass on the table.",
        "reason": "Removed the requirement to preserve the deleted tighter framing.",
    },
    ("rpmte_023", 4): {
        "instruction": "Place both of the man’s hands flat on the table.",
        "preserve_requirements": ["short boxed beard"],
        "target_state": "Both of the man’s hands are resting flat on the table.",
        "reason": "Removed the requirement to preserve the deleted tighter framing.",
    },
    ("rpmte_023", 5): {
        "instruction": "Change the man’s tie to a solid dark blue necktie.",
        "preserve_requirements": ["short boxed beard", "both hands flat on the table"],
        "target_state": "The man is wearing a solid dark blue necktie.",
        "reason": "Removed the requirement to preserve the deleted tighter framing.",
    },
    ("rpmte_024", 3): {
        "instruction": "Have the man pet the visible dog on the head with one hand.",
        "preserve_requirements": ["full beard and mustache"],
        "target_state": "The man pets the dog’s head with one hand while remaining seated on the sand.",
        "reason": "Removed the requirement to preserve the deleted tighter man-and-dog framing.",
    },
    ("rpmte_024", 4): {
        "instruction": "Have the man wave toward the camera with his free hand.",
        "preserve_requirements": ["full beard and mustache", "dog remains visible beside him"],
        "target_state": "The man waves at the camera with his free hand while staying seated beside the dog.",
        "reason": "Removed only the deleted tighter-framing state; 'toward the camera' specifies the wave direction.",
    },
    ("rpmte_024", 5): {
        "instruction": "Change the man’s shirt to a dark blue long-sleeve button-up.",
        "preserve_requirements": ["full beard and mustache", "waving pose with free hand"],
        "target_state": "The man wears a dark blue long-sleeve button-up shirt while continuing the beach interaction with the dog.",
        "reason": "Removed the requirement to preserve the deleted tighter man-and-dog framing.",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_media_atomic(source: Path, destination: Path) -> str:
    source_sha = sha256(source)
    if destination.is_file() and sha256(destination) == source_sha:
        return source_sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"media copy hash mismatch: {source}")
    os.replace(temporary, destination)
    return source_sha


def changed_fields(original: dict, revised: dict) -> list[str]:
    return [
        field
        for field in ("instruction", "preserve_requirements", "target_state")
        if original[field] != revised[field]
    ]


def main() -> None:
    if sha256(SOURCE_BENCHMARK) != SOURCE_BENCHMARK_SHA256:
        raise RuntimeError(f"source benchmark hash mismatch: {SOURCE_BENCHMARK}")
    rows = [json.loads(line) for line in SOURCE_BENCHMARK.read_text(encoding="utf-8").splitlines() if line]
    selected = [
        row
        for row in rows
        if row["chain_type"] != "rollback"
        and all(turn.get("history_reference_round") is None for turn in row["turns"])
    ]
    if tuple(row["chain_id"] for row in selected) != EXPECTED_CHAIN_IDS:
        raise RuntimeError("source selection no longer yields rpmte_001 through rpmte_024")

    cases: list[dict] = []
    rewrite_audit: list[dict] = []
    removed_camera_turns: list[dict] = []
    seen_rewrites: set[tuple[str, int]] = set()
    axis_counts = {"appearance": 0, "action": 0, "outfit": 0}

    for row in selected:
        chain_id = row["chain_id"]
        camera_turns = [turn for turn in row["turns"] if turn["axis"] == "camera"]
        if len(camera_turns) > 1:
            raise RuntimeError(f"unexpected multiple camera turns: {chain_id}")
        camera_round = camera_turns[0]["round"] if camera_turns else None
        if camera_turns:
            removed_camera_turns.append({
                "chain_id": chain_id,
                "source_round": camera_round,
                "removed_turn": copy.deepcopy(camera_turns[0]),
                "reason": "camera axis is unsupported by the current model training",
            })

        source_relative = Path(row["source"]["media"])
        source_media = SOURCE_ROOT / source_relative
        destination_media = ROOT / source_relative
        local_sha = copy_media_atomic(source_media, destination_media)

        outputs: list[dict] = []
        conversation_turns: list[dict] = []
        round_mapping: list[dict] = []
        new_round = 0
        for source_turn in row["turns"]:
            source_round = source_turn["round"]
            if source_turn["axis"] == "camera":
                continue
            new_round += 1
            revised = copy.deepcopy(source_turn)
            rewrite_key = (chain_id, source_round)
            if camera_round is not None and source_round > camera_round:
                if rewrite_key not in REWRITES:
                    raise RuntimeError(f"missing explicit post-camera review: {chain_id} R{source_round}")
                replacement = REWRITES[rewrite_key]
                for field in ("instruction", "preserve_requirements", "target_state"):
                    revised[field] = copy.deepcopy(replacement[field])
                fields = changed_fields(source_turn, revised)
                if not fields:
                    raise RuntimeError(f"explicit rewrite made no change: {chain_id} R{source_round}")
                rewrite_audit.append({
                    "chain_id": chain_id,
                    "source_round": source_round,
                    "derived_round": new_round,
                    "camera_source_round_removed": camera_round,
                    "changed_fields": fields,
                    "reason": replacement["reason"],
                    "original": {
                        field: copy.deepcopy(source_turn[field])
                        for field in ("instruction", "preserve_requirements", "target_state")
                    },
                    "revised": {
                        field: copy.deepcopy(revised[field])
                        for field in ("instruction", "preserve_requirements", "target_state")
                    },
                })
                seen_rewrites.add(rewrite_key)
            elif rewrite_key in REWRITES:
                raise RuntimeError(f"rewrite does not follow a camera turn: {chain_id} R{source_round}")

            revised["round"] = new_round
            revised["source_round"] = source_round
            axis_counts[revised["axis"]] += 1
            output = {
                "round": new_round,
                "source_round": source_round,
                "axis": revised["axis"],
                "instruction": revised["instruction"],
                "preserve_requirements": revised["preserve_requirements"],
                "target_state": revised["target_state"],
                "history_reference_round": None,
            }
            outputs.append(output)
            content = []
            if new_round == 1:
                content.append({"type": "video", "value": str(destination_media.resolve())})
            content.append({"type": "text", "value": revised["instruction"]})
            conversation_turns.append({
                "role": "user",
                "action": "edit_video",
                "content": content,
                "meta": {
                    "chain_id": chain_id,
                    "round": new_round,
                    "source_round": source_round,
                    "axis": revised["axis"],
                    "preserve_requirements": revised["preserve_requirements"],
                    "target_state": revised["target_state"],
                },
            })
            round_mapping.append({"source_round": source_round, "derived_round": new_round})

        source_record = copy.deepcopy(row["source"])
        source_record.update({
            "parent_media": str(source_media.resolve()),
            "derived_media": str(destination_media.resolve()),
            "local_sha256": local_sha,
        })
        cases.append({
            "chain_id": chain_id,
            "chain_type": row["chain_type"],
            "source": str(destination_media.resolve()),
            "source_sha256": local_sha,
            "source_metadata": source_record,
            "scene": copy.deepcopy(row["scene"]),
            "face_detail_level": row.get("face_detail_level"),
            "kept_source_rounds": [item["source_round"] for item in round_mapping],
            "removed_source_rounds": [camera_round] if camera_round is not None else [],
            "round_mapping": round_mapping,
            "conversation": {"system_prompt": None, "turns": conversation_turns},
            "outputs": outputs,
        })

    if seen_rewrites != set(REWRITES):
        missing = sorted(set(REWRITES) - seen_rewrites)
        raise RuntimeError(f"unused explicit rewrites: {missing}")
    total_turns = sum(len(case["outputs"]) for case in cases)
    formal_checkpoints = sum(max(0, len(case["outputs"]) - 1) for case in cases)
    if (len(cases), len(removed_camera_turns), len(rewrite_audit), total_turns, formal_checkpoints) != (24, 21, 33, 65, 41):
        raise RuntimeError("derived benchmark counts differ from the frozen contract")
    if axis_counts != {"appearance": 23, "action": 20, "outfit": 22}:
        raise RuntimeError(f"unexpected derived axis counts: {axis_counts}")

    benchmark = {
        "schema_version": "ivebench_nonrollback_nocamera_history_vit_v1",
        "status": "frozen",
        "source_benchmark": str(SOURCE_BENCHMARK.resolve()),
        "source_benchmark_sha256": SOURCE_BENCHMARK_SHA256,
        "derivation": {
            "selection": "chain_type != rollback and every history_reference_round is null",
            "camera_policy": "remove every camera-axis turn before inference",
            "text_policy": "explicitly revise every surviving turn after a removed camera turn; no keyword replacement",
            "ground_truth_available": False,
        },
        "chain_count": len(cases),
        "total_turns": total_turns,
        "formal_checkpoints": formal_checkpoints,
        "axis_counts": axis_counts,
        "removed_camera_turn_count": len(removed_camera_turns),
        "post_camera_rewrite_count": len(rewrite_audit),
        "removed_camera_turns": removed_camera_turns,
        "camera_dependency_rewrites": rewrite_audit,
        "cases": cases,
    }
    write_json_atomic(OUTPUT, benchmark)
    write_json_atomic(REWRITE_AUDIT, {
        "schema_version": "ivebench_camera_dependency_rewrite_audit_v1",
        "source_benchmark_sha256": SOURCE_BENCHMARK_SHA256,
        "removed_camera_turn_count": len(removed_camera_turns),
        "rewrite_count": len(rewrite_audit),
        "removed_camera_turns": removed_camera_turns,
        "rewrites": rewrite_audit,
    })
    print(json.dumps({
        "benchmark": str(OUTPUT),
        "benchmark_sha256": sha256(OUTPUT),
        "chains": len(cases),
        "turns": total_turns,
        "formal_checkpoints": formal_checkpoints,
        "axis_counts": axis_counts,
        "removed_camera_turns": len(removed_camera_turns),
        "post_camera_rewrites": len(rewrite_audit),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
