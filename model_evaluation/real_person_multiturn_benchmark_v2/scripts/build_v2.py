#!/usr/bin/env python3
"""Regenerate v2 instructions for the frozen v1 real-video sources."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
MODEL_EVAL = ROOT.parent
PROJECT = MODEL_EVAL.parent.parent
V1 = MODEL_EVAL / "real_person_multiturn_benchmark_v1"
IVE = MODEL_EVAL / "ivebench_lance"
SENIOR = Path("/scratch/prj0000000275/jcy/xks/DataMake/00_engine/outputs/delivery/"
              "first_batch_candidate_20260830_v002/dataset_v1/chains.json")
MODEL = "qwen/qwen3.8-omni-flash"
PROVIDER = "Alibaba"
DATASET_REVISION = "1e6cb655c0a537f6093edcf18e2607fb03e76734"
AXES = ("appearance", "outfit", "action", "camera")
ACTION_POSITION_TARGET = {1: 0, 2: 4, 3: 14, 4: 9, 5: 1}
ACTION_FAMILIES = ("wave", "clap", "thumbs_up", "crossed_arms", "catch_breath", "tie_shoe",
                   "point", "object_interaction", "dance", "restore_original", "not_applicable")
APPEARANCE_FAMILIES = ("facial_hair", "headwear", "eyewear", "hairstyle", "makeup",
                       "visible_face_detail", "restore_original", "not_applicable")
APPEARANCE_LIMITS = {"facial_hair": 10, "headwear": 5, "eyewear": 5, "hairstyle": 5,
                     "makeup": 2, "visible_face_detail": 3}
ACTION_REQUIREMENTS = {
    "rpmte_001": {2: "thumbs_up"}, "rpmte_003": {3: "thumbs_up"},
    "rpmte_008": {2: "thumbs_up"}, "rpmte_010": {3: "thumbs_up"},
    "rpmte_011": {3: "crossed_arms"}, "rpmte_012": {3: "crossed_arms"},
    "rpmte_013": {3: "point"}, "rpmte_014": {4: "wave"}, "rpmte_015": {2: "wave"},
    "rpmte_016": {3: "point"}, "rpmte_017": {4: "crossed_arms"},
    "rpmte_018": {4: "object_interaction"},
    "rpmte_021": {2: "dance", 4: "clap"},
    "rpmte_022": {3: "catch_breath", 4: "crossed_arms"},
    "rpmte_023": {3: "point", 4: "object_interaction"},
    "rpmte_024": {3: "object_interaction", 4: "wave"},
    "rpmte_026": {3: "clap", 4: "restore_original"},
    "rpmte_027": {3: "dance", 4: "restore_original"},
    "rpmte_028": {3: "clap", 5: "restore_original"},
    "rpmte_029": {3: "catch_breath"}, "rpmte_030": {3: "point"},
}
APPEARANCE_REQUIREMENTS = {
    "rpmte_001": {1: "eyewear"}, "rpmte_002": {1: "eyewear"},
    "rpmte_003": {1: "eyewear"}, "rpmte_004": {1: "headwear"},
    "rpmte_005": {1: "eyewear"}, "rpmte_006": {1: "headwear"},
    "rpmte_007": {1: "facial_hair"}, "rpmte_008": {1: "facial_hair"},
    "rpmte_009": {1: "headwear"}, "rpmte_011": {1: "facial_hair"},
    "rpmte_012": {1: "makeup"}, "rpmte_013": {1: "headwear"},
    "rpmte_014": {1: "facial_hair"}, "rpmte_015": {1: "facial_hair"},
    "rpmte_016": {1: "facial_hair"}, "rpmte_017": {1: "hairstyle"},
    "rpmte_018": {1: "facial_hair"},
    "rpmte_019": {1: "facial_hair", 2: "hairstyle"},
    "rpmte_021": {1: "hairstyle"}, "rpmte_022": {1: "hairstyle"},
    "rpmte_023": {1: "facial_hair"}, "rpmte_024": {1: "facial_hair"},
    "rpmte_026": {1: "makeup"}, "rpmte_028": {1: "headwear"},
    "rpmte_029": {1: "hairstyle", 5: "restore_original"},
    "rpmte_030": {1: "eyewear"},
}
ACTION_CONCEPTS = {
    "rpmte_001": {2: ("act_right_thumbsup", "raise her right hand in a clear thumbs-up")},
    "rpmte_003": {3: ("act_left_thumbsup", "raise his left hand in a clear thumbs-up")},
    "rpmte_008": {2: ("act_double_thumbsup", "give a thumbs-up with both hands")},
    "rpmte_010": {3: ("act_seated_thumbsup", "give a right-hand thumbs-up while remaining seated")},
    "rpmte_011": {3: ("act_cross_arms_conversation", "cross both arms over his chest during the conversation")},
    "rpmte_012": {3: ("act_fold_arms_concerned", "fold both arms firmly across her torso")},
    "rpmte_013": {3: ("act_point_ornate_detail", "point toward a clearly visible ornate detail in the scene")},
    "rpmte_014": {4: ("act_office_wave", "raise his right hand and wave once")},
    "rpmte_015": {2: ("act_wedding_wave", "wave to the seated guests while continuing down the aisle")},
    "rpmte_016": {3: ("act_point_horizon", "extend one arm and point toward the ocean horizon")},
    "rpmte_017": {4: ("act_cross_arms_intense", "bring both forearms across his chest during the tense conversation")},
    "rpmte_018": {4: ("act_lower_cigarette", "lower the visible cigarette and hold it at his side")},
    "rpmte_021": {2: ("act_side_step_dance", "perform a clear side-to-side dance step"),
                    4: ("act_clap_to_music", "clap both hands twice in time with the music")},
    "rpmte_022": {3: ("act_hands_knees_breath", "place both hands on her knees and catch her breath"),
                    4: ("act_cross_arms_tightly", "stand upright with both arms crossed tightly over her chest")},
    "rpmte_023": {3: ("act_point_table_item", "point toward a visible item on the table"),
                    4: ("act_palms_on_table", "place both hands flat on the table")},
    "rpmte_024": {3: ("act_pet_dog", "pet the visible dog on the head with one hand"),
                    4: ("act_wave_free_hand", "wave toward the camera with his free hand")},
    "rpmte_026": {3: ("act_single_clap", "bring both hands together in one clear clap"),
                    4: ("act_restore_original", "restore the original hand position and action")},
    "rpmte_027": {3: ("act_rooftop_spin", "perform one full dance spin with both arms raised"),
                    4: ("act_restore_original", "restore the original rooftop dance action")},
    "rpmte_028": {3: ("act_cinema_clap", "clap along while watching the movie"),
                    5: ("act_restore_original", "restore the original cinema audience action")},
    "rpmte_029": {3: ("act_hand_chest_breath", "place one hand on her chest and take a visible recovering breath")},
    "rpmte_030": {3: ("act_point_across_desk", "point toward the other person across the desk")},
}
APPEARANCE_CONCEPTS = {
    "rpmte_001": {1: ("app_gold_rim_glasses", "thin gold-rimmed eyeglasses")},
    "rpmte_002": {1: ("app_black_sunglasses", "black sunglasses")},
    "rpmte_003": {1: ("app_rectangular_glasses", "dark rectangular eyeglasses")},
    "rpmte_004": {1: ("app_knit_beanie", "a dark knit beanie")},
    "rpmte_005": {1: ("app_tortoiseshell_glasses", "thick tortoiseshell glasses")},
    "rpmte_006": {1: ("app_stage_hat", "a large dark wide-brimmed stage hat")},
    "rpmte_007": {1: ("app_goatee_mustache", "a defined goatee and mustache")},
    "rpmte_008": {1: ("app_short_full_beard", "a clearly visible short full beard")},
    "rpmte_009": {1: ("app_black_fedora", "a black fedora")},
    "rpmte_011": {1: ("app_fuller_beard_mustache", "a fuller beard and mustache")},
    "rpmte_012": {1: ("app_smoky_eye_makeup", "a clearly visible smoky eye makeup look")},
    "rpmte_013": {1: ("app_dark_hood", "a dark medieval-style hood around the silver mask")},
    "rpmte_014": {1: ("app_neat_goatee", "a sharply outlined chin goatee with both cheeks left clean-shaven")},
    "rpmte_015": {1: ("app_full_dark_beard", "a full dark beard")},
    "rpmte_016": {1: ("app_gray_streak_beard", "a short beard with visible gray streaks")},
    "rpmte_017": {1: ("app_shoulder_wavy_hair", "shoulder-length wavy hair")},
    "rpmte_018": {1: ("app_beard_defined_mustache", "a short beard with a strongly defined mustache")},
    "rpmte_019": {1: ("app_long_gray_beard", "a longer gray beard"),
                    2: ("app_slick_gray_hair", "slicked-back gray hair")},
    "rpmte_021": {1: ("app_high_ponytail", "a sleek high ponytail")},
    "rpmte_022": {1: ("app_chin_bob", "a chin-length blunt bob hairstyle")},
    "rpmte_023": {1: ("app_boxed_beard", "a short boxed beard")},
    "rpmte_024": {1: ("app_beard_and_mustache", "a full beard and mustache")},
    "rpmte_026": {1: ("app_red_lip_makeup", "clearly defined deep-red lipstick")},
    "rpmte_028": {1: ("app_burgundy_beanie", "a burgundy knit beanie")},
    "rpmte_029": {1: ("app_loose_wavy_hair", "loose shoulder-length waves"),
                    5: ("app_restore_original", "restore the original hairstyle from the source video")},
    "rpmte_030": {1: ("app_dark_rectangular_glasses", "dark rectangular glasses")},
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_v1() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (V1 / "benchmark.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_sequences(kind: str, length: int) -> list[tuple[str, ...]]:
    result = []
    for sequence in itertools.product(AXES, repeat=length):
        if sequence[0] == "action":
            continue
        counts = Counter(sequence)
        progressive = kind == "progressive" and len(counts) == length
        repeated = kind in {"repeated_axis", "rollback"} and sorted(counts.values()) == [1] * (length - 2) + [2]
        if progressive or (repeated and (kind != "rollback" or sequence[-1] in sequence[:-1])):
            result.append(sequence)
    return result


def solve_blueprints(chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_axes = (28, 28, 28, 28)
    target_positions = tuple(ACTION_POSITION_TARGET[index] for index in range(1, 6))
    ordered = sorted(enumerate(chains), key=lambda item: (-len(item[1]["turns"]), item[1]["chain_type"] != "rollback", item[0]))
    candidates = [candidate_sequences(chain["chain_type"], len(chain["turns"])) for _, chain in ordered]
    memo = set()

    def solve(index: int, axis_totals: tuple[int, ...], position_totals: tuple[int, ...]):
        state = (index, axis_totals, position_totals)
        if state in memo: return None
        if index == len(ordered):
            return [] if axis_totals == target_axes and position_totals == target_positions else None
        options = []
        for sequence in candidates[index]:
            count = Counter(sequence)
            new_axes = tuple(axis_totals[i] + count[axis] for i, axis in enumerate(AXES))
            new_positions = list(position_totals)
            for turn, axis in enumerate(sequence):
                if axis == "action": new_positions[turn] += 1
            new_positions = tuple(new_positions)
            if any(new_axes[i] > target_axes[i] for i in range(4)) or any(new_positions[i] > target_positions[i] for i in range(5)):
                continue
            score = sum((target_axes[i] - new_axes[i]) ** 2 for i in range(4))
            score += 4 * sum((target_positions[i] - new_positions[i]) ** 2 for i in range(5))
            options.append((score, sequence, new_axes, new_positions))
        for _, sequence, new_axes, new_positions in sorted(options):
            tail = solve(index + 1, new_axes, new_positions)
            if tail is not None: return [sequence] + tail
        memo.add(state)
        return None

    solution = solve(0, (0, 0, 0, 0), (0, 0, 0, 0, 0))
    if solution is None: raise RuntimeError("cannot solve v2 blueprint quotas")
    restored = {original: sequence for (original, _), sequence in zip(ordered, solution)}
    result = []
    for index, chain in enumerate(chains):
        result.append({"chain_id": chain["chain_id"], "chain_type": chain["chain_type"],
                       "length": len(chain["turns"]), "axes": list(restored[index])})
    validate_blueprints(result)
    return result


def validate_blueprints(blueprints: list[dict[str, Any]]) -> None:
    if Counter(item["length"] for item in blueprints) != Counter({3: 13, 4: 12, 5: 5}): raise ValueError("length quota")
    if Counter(item["chain_type"] for item in blueprints) != Counter({"progressive": 18, "repeated_axis": 6, "rollback": 6}): raise ValueError("type quota")
    axes = Counter(axis for item in blueprints for axis in item["axes"])
    if axes != Counter({axis: 28 for axis in AXES}): raise ValueError(f"axis quota {axes}")
    positions = Counter(turn for item in blueprints for turn, axis in enumerate(item["axes"], 1) if axis == "action")
    if positions != Counter({key: value for key, value in ACTION_POSITION_TARGET.items() if value}):
        raise ValueError(f"action position quota {positions}")
    expected_action, expected_appearance = Counter(), Counter()
    for item in blueprints:
        for turn, axis in enumerate(item["axes"], 1):
            if axis == "action":
                family = ACTION_REQUIREMENTS.get(item["chain_id"], {}).get(turn)
                if not family: raise ValueError(f"missing action family: {item['chain_id']} round {turn}")
                if turn not in ACTION_CONCEPTS.get(item["chain_id"], {}): raise ValueError(f"missing action concept: {item['chain_id']} round {turn}")
                expected_action[family] += 1
            if axis == "appearance":
                family = APPEARANCE_REQUIREMENTS.get(item["chain_id"], {}).get(turn)
                if not family: raise ValueError(f"missing appearance family: {item['chain_id']} round {turn}")
                if turn not in APPEARANCE_CONCEPTS.get(item["chain_id"], {}): raise ValueError(f"missing appearance concept: {item['chain_id']} round {turn}")
                expected_appearance[family] += 1
    if expected_appearance != Counter({"facial_hair": 10, "headwear": 5, "eyewear": 5,
                                       "hairstyle": 5, "makeup": 2, "restore_original": 1}):
        raise ValueError(f"appearance family quota {expected_appearance}")
    if any(count > 5 for family, count in expected_action.items() if family != "restore_original"):
        raise ValueError(f"action family overuse {expected_action}")


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def references() -> list[set[str]]:
    texts = []
    for name in ("ivebenchdb_prompt_short.json", "ivebenchdb_prompt_long.json"):
        texts.extend(row["edit_prompt"] for row in json.loads((IVE / "data/official" / name).read_text()))
    senior = json.loads(SENIOR.read_text())
    texts.extend(turn["prompt"]["en"] for chain in senior for turn in chain["turns"])
    for chain in load_v1(): texts.extend(turn["instruction"] for turn in chain["turns"])
    return [token_set(text) for text in texts]


def max_jaccard(text: str, refs: list[set[str]]) -> float:
    tokens = token_set(text)
    return max((len(tokens & other) / len(tokens | other) for other in refs if tokens or other), default=0.0)


def prompt(chain: dict[str, Any], blueprint: dict[str, Any], action_counts: Counter,
           appearance_counts: Counter, feedback: str = "") -> str:
    kind_rule = {
        "progressive": "Every turn edits a different axis and preserves all earlier edits.",
        "repeated_axis": "The repeated axis gets two distinct compatible edits; the second is not a rollback.",
        "rollback": "The final turn restores the original source state of its repeated axis and preserves all other edits.",
    }[blueprint["chain_type"]]
    return f"""Rewrite one real-video multi-turn editing chain. The source has already passed human-centered suitability screening.

Source caption: {chain['source']['official_source_caption']}
Scene summary: {chain['scene']['scene_summary']}
Frozen chain type: {blueprint['chain_type']}
Frozen axis sequence: {json.dumps(blueprint['axes'])}
Required action families by round: {json.dumps(ACTION_REQUIREMENTS.get(chain['chain_id'], {}))}
Required appearance families by round: {json.dumps(APPEARANCE_REQUIREMENTS.get(chain['chain_id'], {}))}
Required action concept IDs and visible changes by round: {json.dumps(ACTION_CONCEPTS.get(chain['chain_id'], {}))}
Required appearance concept IDs and visible changes by round: {json.dumps(APPEARANCE_CONCEPTS.get(chain['chain_id'], {}))}
Rule: {kind_rule}

Return exactly {blueprint['length']} concise English instructions, one edit per turn.
For each action or appearance turn, copy the required concept ID exactly into concept_id and realize its specified visible change. For outfit/camera turns use concept_id="not_applicable".

ACTION POLICY (important):
- Action is deliberately never round 1. Use only a clearly visible, objectively judgeable body action compatible with the frames and scene.
- Allowed families: wave; clap; thumbs-up; cross both arms; place both hands on knees to catch breath; tie one shoe (only if full body and shoes are visible); point toward a clearly visible scene object; interact with an already visible object/instrument; dance (only for a dancer/performer with enough body visible); restore the original action for rollback.
- Never use leaning forward/back, slight/subtle motion, gaze-only/head-only changes, vague audience reaction, or "as if" descriptions.
- Do not invent a new handheld object. Prefer a source-specific action when an instrument, pet, tray, notebook, or similar object is already visible.
- Use exactly the required action family for every action round. Current accepted action-family counts: {dict(action_counts)}.

APPEARANCE POLICY:
- First assess face_detail_level from the contact sheet: small, medium, or close.
- When the face is small, do not request eye makeup, blush, lip tint, a nose stud, eyebrow detail, a tiny scar, or light stubble. Use a visually large accessory (hat/cap/beanie, sunglasses, clear-frame glasses, scarf) or a clearly defined beard/goatee where appropriate.
- Medium faces may use strong hairstyle silhouette, eyewear, headwear, or clearly visible facial hair, but not fine eye makeup or tiny details.
- Only a clear close face may use eye makeup, and use makeup sparingly (at most two benchmark turns).
- Facial hair is useful when obvious: clearly visible short beard, defined goatee, beard and mustache, fuller beard, or gray-streaked beard. Never ask for "light stubble" or an almost imperceptible trim.
- Use exactly the required appearance family for every appearance round. Current accepted appearance-family counts: {dict(appearance_counts)}. Final family totals are deliberately balanced.

OTHER RULES:
- Outfit edits must remain visible and plausible. Camera edits should be modest and coherent.
- Do not replace the person, change global style, add text/logos, or change the scene.
- target_state is a short factual resulting state. preserve_requirements names prior edited states that remain.
- history_reference_round is null except the final rollback, where it must be 0.
- Do not copy stock IVEBench or synthetic-training wording.
{('Correct these previous-response problems: ' + feedback) if feedback else ''}"""


def schema(blueprint: dict[str, Any]) -> dict[str, Any]:
    turn = {"type": "object", "additionalProperties": False,
            "required": ["round", "axis", "instruction", "target_state", "preserve_requirements",
                         "history_reference_round", "action_family", "appearance_family", "concept_id"],
            "properties": {
                "round": {"type": "integer"}, "axis": {"type": "string", "enum": list(AXES)},
                "instruction": {"type": "string"}, "target_state": {"type": "string"},
                "preserve_requirements": {"type": "array", "items": {"type": "string"}},
                "history_reference_round": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "action_family": {"type": "string", "enum": list(ACTION_FAMILIES)},
                "appearance_family": {"type": "string", "enum": list(APPEARANCE_FAMILIES)},
                "concept_id": {"type": "string"},
            }}
    return {"type": "object", "additionalProperties": False,
            "required": ["face_detail_level", "turns"],
            "properties": {
                "face_detail_level": {"type": "string", "enum": ["small", "medium", "close"]},
                "turns": {"type": "array", "minItems": blueprint["length"], "maxItems": blueprint["length"], "items": turn},
            }}


def call_api(key: str, chain: dict[str, Any], blueprint: dict[str, Any], image_uri: str,
             action_counts: Counter, appearance_counts: Counter, feedback: str) -> tuple[dict, dict]:
    payload = {"model": MODEL, "temperature": 0,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": prompt(chain, blueprint, action_counts, appearance_counts, feedback)},
                   {"type": "image_url", "image_url": {"url": image_uri}},
               ]}],
               "response_format": {"type": "json_schema", "json_schema": {"name": "v2_chain", "strict": True,
                                                                              "schema": schema(blueprint)}},
               "provider": {"data_collection": "deny", "allow_fallbacks": False, "order": [PROVIDER]}}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://localhost/real-person-multiturn-benchmark-v2", "X-Title": "RPMTE benchmark v2"}
    for attempt in range(1, 4):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                        json=payload, timeout=(30, 300))
            response.raise_for_status(); raw = response.json()
            if raw.get("provider") != PROVIDER: raise RuntimeError(f"unexpected provider {raw.get('provider')}")
            value = json.loads(raw["choices"][0]["message"]["content"])
            if isinstance(value, list) and len(value) == 1: value = value[0]
            if not isinstance(value, dict): raise ValueError("JSON root is not an object")
            return value, raw
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)
    raise AssertionError


def validate_result(value: dict[str, Any], blueprint: dict[str, Any], refs: list[set[str]],
                    accepted_instructions: list[str], action_counts: Counter, appearance_counts: Counter) -> list[str]:
    problems = []
    turns = value.get("turns", [])
    if len(turns) != blueprint["length"]: return [f"expected {blueprint['length']} turns"]
    if [turn.get("round") for turn in turns] != list(range(1, blueprint["length"] + 1)): problems.append("round numbers mismatch")
    if [turn.get("axis") for turn in turns] != blueprint["axes"]: problems.append("axis sequence mismatch")
    local_action = Counter(); local_appearance = Counter()
    for index, turn in enumerate(turns, 1):
        instruction = str(turn.get("instruction", "")).strip()
        if len(instruction.split()) < 4: problems.append(f"round {index} instruction too short")
        if max_jaccard(instruction, refs) > 0.75: problems.append(f"round {index} too similar to an existing prompt")
        if any(len(token_set(instruction) & token_set(old)) / max(1, len(token_set(instruction) | token_set(old))) > 0.75
               for old in accepted_instructions): problems.append(f"round {index} duplicates v2 instruction")
        history = turn.get("history_reference_round")
        final_rollback = blueprint["chain_type"] == "rollback" and index == blueprint["length"]
        if final_rollback and history != 0: problems.append("final rollback must reference source round 0")
        if not final_rollback and history is not None: problems.append(f"round {index} unexpected history reference")
        action_family = turn.get("action_family")
        appearance_family = turn.get("appearance_family")
        concept_id = turn.get("concept_id")
        if turn.get("axis") == "action":
            if index == 1: problems.append("action occurs in round 1")
            if action_family == "not_applicable" or action_family not in ACTION_FAMILIES: problems.append(f"round {index} missing action family")
            required_action = ACTION_REQUIREMENTS.get(blueprint["chain_id"], {}).get(index)
            if action_family != required_action: problems.append(f"round {index} requires action family {required_action}")
            required_concept = ACTION_CONCEPTS.get(blueprint["chain_id"], {}).get(index, (None, ""))[0]
            if concept_id != required_concept: problems.append(f"round {index} requires concept_id {required_concept}")
            if not final_rollback and action_family != "restore_original": local_action[action_family] += 1
            if action_family != "restore_original" and action_counts[action_family] + local_action[action_family] > 5:
                problems.append(f"action family {action_family} would exceed five uses")
        elif action_family != "not_applicable": problems.append(f"round {index} non-action turn has action family")
        if turn.get("axis") == "appearance":
            if appearance_family == "not_applicable" or appearance_family not in APPEARANCE_FAMILIES:
                problems.append(f"round {index} missing appearance family")
            elif not final_rollback and appearance_family != "restore_original": local_appearance[appearance_family] += 1
            required_appearance = APPEARANCE_REQUIREMENTS.get(blueprint["chain_id"], {}).get(index)
            if appearance_family != required_appearance: problems.append(f"round {index} requires appearance family {required_appearance}")
            required_concept = APPEARANCE_CONCEPTS.get(blueprint["chain_id"], {}).get(index, (None, ""))[0]
            if concept_id != required_concept: problems.append(f"round {index} requires concept_id {required_concept}")
            if appearance_family == "makeup" and appearance_counts["makeup"] + local_appearance["makeup"] > 2:
                problems.append("makeup would exceed two turns")
            if appearance_family in APPEARANCE_LIMITS and appearance_counts[appearance_family] + local_appearance[appearance_family] > APPEARANCE_LIMITS[appearance_family]:
                problems.append(f"appearance family {appearance_family} would exceed {APPEARANCE_LIMITS[appearance_family]} uses")
        elif appearance_family != "not_applicable": problems.append(f"round {index} non-appearance turn has appearance family")
        if turn.get("axis") not in {"action", "appearance"} and concept_id != "not_applicable":
            problems.append(f"round {index} non-action/appearance turn must use concept_id not_applicable")
    return sorted(set(problems))


def raw_cost(raw: dict[str, Any]) -> float:
    return float((raw.get("usage") or {}).get("cost") or raw.get("cost") or 0.0)


def normalize_result(value: dict[str, Any], blueprint: dict[str, Any]) -> dict[str, Any]:
    """Normalize Qwen's tendency to use history refs as preservation refs."""
    normalized = json.loads(json.dumps(value))
    for turn in normalized.get("turns", []):
        final_rollback = blueprint["chain_type"] == "rollback" and turn.get("round") == blueprint["length"]
        if not final_rollback:
            turn["history_reference_round"] = None
    return normalized


def materialize(chains: list[dict[str, Any]], blueprints: list[dict[str, Any]], records: dict[str, dict], cost: float) -> None:
    output, flat, inventory = [], [], []
    media_dir = ROOT / "media"; media_dir.mkdir(parents=True, exist_ok=True)
    for chain, blueprint in zip(chains, blueprints, strict=True):
        result = records[chain["chain_id"]]["result"]
        source_media = V1 / chain["source"]["media"]
        destination = media_dir / f"{chain['chain_id']}_source.mp4"
        shutil.copy2(source_media, destination)
        source = dict(chain["source"]); source["media"] = destination.relative_to(ROOT).as_posix()
        turns = result["turns"]
        output.append({"benchmark_version": "real_person_multiturn_benchmark_v2", "chain_id": chain["chain_id"],
                       "chain_type": chain["chain_type"], "source": source,
                       "scene": chain["scene"], "face_detail_level": result["face_detail_level"], "turns": turns})
        for turn in turns:
            flat.append({"chain_id": chain["chain_id"], "round": turn["round"], "chain_type": chain["chain_type"],
                         "axis": turn["axis"], "action_family": turn["action_family"],
                         "appearance_family": turn["appearance_family"], "instruction": turn["instruction"],
                         "target_state": turn["target_state"], "preserve_requirements": " | ".join(turn["preserve_requirements"]),
                         "history_reference_round": "" if turn["history_reference_round"] is None else turn["history_reference_round"]})
        inventory.append({"chain_id": chain["chain_id"], "subset": source["subset"], "official_id": source["official_id"],
                          "official_video_name": source["official_video_name"], "official_sha256": source["official_sha256"],
                          "official_source_caption": source["official_source_caption"], "benchmark_media": source["media"],
                          "media_sha256": sha256(destination), "media_bytes": destination.stat().st_size})
    with (ROOT / "benchmark.jsonl").open("w", encoding="utf-8") as stream:
        for chain in output: stream.write(json.dumps(chain, ensure_ascii=False) + "\n")
    write_csv(ROOT / "turns.csv", flat); write_csv(ROOT / "source_inventory.csv", inventory)
    atomic_json(ROOT / "metadata/api_summary.json", {"model": MODEL, "required_provider": PROVIDER,
                "credential_env": "LIJUYANG_ICLR", "personal_credential_used": False, "data_collection": "deny",
                "temperature": 0, "hard_budget_usd": 2.0, "reported_cost_usd": round(cost, 8), "chains": 30})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    chains = load_v1(); blueprints = solve_blueprints(chains)
    atomic_json(ROOT / "metadata/blueprints.json", blueprints)
    preflight = {"chains": len(chains), "turns": sum(item["length"] for item in blueprints),
                 "axes": dict(Counter(axis for item in blueprints for axis in item["axes"])),
                 "action_positions": dict(Counter(turn for item in blueprints for turn, axis in enumerate(item["axes"], 1) if axis == "action")),
                 "source_hashes_match_v1": True}
    atomic_json(ROOT / "metadata/preflight.json", preflight); print(json.dumps(preflight, indent=2))
    if args.dry_run: return
    key = os.environ.get("LIJUYANG_ICLR")
    if not key: raise RuntimeError("LIJUYANG_ICLR missing; personal credentials are unsupported")
    refs = references(); records: dict[str, dict] = {}
    state_path = ROOT / "metadata/build_state.json"
    if state_path.exists(): records = json.loads(state_path.read_text()).get("records", {})
    action_counts, appearance_counts, accepted_instructions = Counter(), Counter(), []
    for chain, blueprint in zip(chains, blueprints, strict=True):
        saved = records.get(chain["chain_id"])
        if saved:
            saved["result"] = normalize_result(saved["result"], blueprint)
        if saved and not validate_result(saved["result"], blueprint, refs, accepted_instructions, action_counts, appearance_counts):
            result = saved["result"]
        else:
            source = chain["source"]
            sheet = V1 / "metadata/contact_sheets" / f"{source['subset']}_{int(source['official_id']):04d}.jpg"
            image_uri = "data:image/jpeg;base64," + base64.b64encode(sheet.read_bytes()).decode("ascii")
            feedback = ""; saved = None
            for attempt in range(1, 4):
                value, raw = call_api(key, chain, blueprint, image_uri, action_counts, appearance_counts, feedback)
                raw_dir = ROOT / "metadata/api_raw"; raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{chain['chain_id']}_try{attempt}.json"
                suffix = 1
                while raw_path.exists():
                    raw_path = raw_dir / f"{chain['chain_id']}_try{attempt}_rerun{suffix}.json"; suffix += 1
                atomic_json(raw_path, raw)
                value = normalize_result(value, blueprint)
                problems = validate_result(value, blueprint, refs, accepted_instructions, action_counts, appearance_counts)
                saved = {"result": value, "attempt": attempt, "problems": problems,
                         "raw_record": raw_path.relative_to(ROOT).as_posix(), "provider": raw.get("provider"),
                         "generation_id": raw.get("id"), "cost_usd": raw_cost(raw)}
                if not problems: break
                feedback = "; ".join(problems)
            if saved is None or saved["problems"]: raise RuntimeError(f"{chain['chain_id']} failed after retries: {saved}")
            result = saved["result"]; records[chain["chain_id"]] = saved
            cost = sum(raw_cost(json.loads(path.read_text())) for path in (ROOT / "metadata/api_raw").glob("*.json"))
            if cost > 2.0: raise RuntimeError(f"API budget exceeded: {cost}")
            atomic_json(state_path, {"status": "generating", "completed": len(records), "reported_cost_usd": cost, "records": records})
        for turn in result["turns"]:
            accepted_instructions.append(turn["instruction"])
            if turn["axis"] == "action" and turn["action_family"] not in {"restore_original", "not_applicable"}:
                action_counts[turn["action_family"]] += 1
            if turn["axis"] == "appearance" and turn["appearance_family"] not in {"restore_original", "not_applicable"}:
                appearance_counts[turn["appearance_family"]] += 1
        print(f"{chain['chain_id']} complete action={dict(action_counts)} appearance={dict(appearance_counts)}", flush=True)
    total_cost = sum(raw_cost(json.loads(path.read_text())) for path in (ROOT / "metadata/api_raw").glob("*.json"))
    materialize(chains, blueprints, records, total_cost)
    atomic_json(state_path, {"status": "materialized", "completed": 30, "reported_cost_usd": total_cost,
                             "action_families": dict(action_counts), "appearance_families": dict(appearance_counts), "records": records})
    print(f"materialized v2 cost=${total_cost:.6f}")


if __name__ == "__main__": main()
