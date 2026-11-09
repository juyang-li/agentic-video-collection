#!/usr/bin/env python3
"""Select 12 new real-person IVEBench sources for the drift stress test.

Five candidates reuse the frozen v1 reserve decisions.  Only the additional
seven candidates are sent to Qwen3.8 Omni Flash, through the non-personal
OpenRouter credential.  The script is restartable and stores every raw reply.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
MODEL_EVAL = ROOT.parent
V1 = MODEL_EVAL / "real_person_multiturn_benchmark_v1"
V2 = MODEL_EVAL / "real_person_multiturn_benchmark_v2"
sys.path.insert(0, str(V1 / "scripts"))
import build_benchmark as old  # noqa: E402


MODEL = "qwen/qwen3.8-omni-flash"
PROVIDER = "Alibaba"
RESERVE_IDS = (376, 357, 338)
CANDIDATE_IDS = (324, 373, 165, 368, 386, 157, 361, 282, 251, 327, 326, 374, 378, 379, 380)
TARGET_NEW = 9
HARD_BUDGET_USD = 1.0


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def response_cost(raw: dict[str, Any]) -> float:
    usage = raw.get("usage") or {}
    value = usage.get("cost", raw.get("cost", 0))
    return float(value or 0)


def schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["suitable", "rejection_codes", "real_human", "adult",
                     "primary_person_count", "person_scale", "face_visible",
                     "upper_body_visible", "action_editable", "stable_shot",
                     "scene_type", "scene_tag", "scene_summary"],
        "properties": {
            "suitable": {"type": "boolean"},
            "rejection_codes": {"type": "array", "items": {"type": "string"}},
            "real_human": {"type": "boolean"}, "adult": {"type": "boolean"},
            "primary_person_count": {"type": "integer"},
            "person_scale": {"type": "string", "enum": ["tiny", "small", "medium", "large"]},
            "face_visible": {"type": "boolean"}, "upper_body_visible": {"type": "boolean"},
            "action_editable": {"type": "boolean"}, "stable_shot": {"type": "boolean"},
            "scene_type": {"type": "string", "enum": ["indoor", "outdoor", "mixed"]},
            "scene_tag": {"type": "string"}, "scene_summary": {"type": "string"},
        },
    }


def call_openrouter(key: str, row: dict[str, Any], image_uri: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = f"""Inspect these three chronological frames from one real video.
We are selecting sources for a multi-turn person-editing drift benchmark.
Caption (may be imperfect): {row['source_prompt']}

Accept only when all are true: a clearly adult real human; exactly one unambiguous
primary person (a clearly secondary partial/background person is allowed); the
primary face/head and upper body are visible at medium or large scale across the
clip; no shot cut, severe occlusion, or unstable composition.  The source must
support obvious edits such as eyewear/headwear/facial hair/outfit/camera framing,
and preferably a simple visible arm gesture.  Reject animation, uncertain age,
tiny/distant people, crowds, co-equal people, and face-obscuring shots.
Return only the requested structured assessment. Do not write edit instructions."""
    payload = {
        "model": MODEL, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "drift_source_screen", "strict": True, "schema": schema()}},
        "provider": {"data_collection": "deny", "allow_fallbacks": False, "order": [PROVIDER]},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://localhost/real-person-drift-stress",
               "X-Title": "Real-person drift benchmark source screen"}
    for attempt in range(3):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post("https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers, json=payload, timeout=(30, 300))
            response.raise_for_status()
            raw = response.json()
            if raw.get("provider") != PROVIDER:
                raise RuntimeError(f"unexpected provider: {raw.get('provider')!r}")
            result = json.loads(raw["choices"][0]["message"]["content"])
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            if not isinstance(result, dict):
                raise TypeError("structured response is not an object")
            return result, raw
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def acceptable(result: dict[str, Any]) -> bool:
    return bool(
        result.get("suitable") and result.get("real_human") and result.get("adult")
        and result.get("primary_person_count") == 1
        and result.get("person_scale") in {"medium", "large"}
        and result.get("face_visible") and result.get("upper_body_visible")
        and result.get("stable_shot")
    )


def prior_reserves(rows_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    state = json.loads((V1 / "metadata/build_state.json").read_text(encoding="utf-8"))
    attempts = {int(item["source"]["id"]): item for item in state["attempts"] if item.get("accepted")}
    selected = []
    for source_id in RESERVE_IDS:
        record = attempts[source_id]
        result = next(item["result"] for item in reversed(record["api_attempts"])
                      if item.get("result", {}).get("suitable") and not item.get("violations"))
        selected.append({"selection": "v1_frozen_reserve", "source": rows_by_id[source_id],
                         "screening": result, "reported_cost_usd": 0.0})
    return selected


def main() -> None:
    key = os.environ.get("LIJUYANG_ICLR")
    if not key:
        raise RuntimeError("LIJUYANG_ICLR is required; the personal credential is intentionally unsupported")
    rows_by_id = {int(row["id"]): row for row in old.candidate_rows() if row["subset"] == "short"}
    missing = (set(RESERVE_IDS) | set(CANDIDATE_IDS)) - set(rows_by_id)
    if missing:
        raise RuntimeError(f"candidate IDs missing from frozen source pool: {sorted(missing)}")

    used_hashes = set()
    with (V2 / "benchmark.jsonl").open(encoding="utf-8") as stream:
        used_hashes.update(json.loads(line)["source"]["official_sha256"] for line in stream if line.strip())
    if any(rows_by_id[source_id]["official_sha256"] in used_hashes
           for source_id in RESERVE_IDS + CANDIDATE_IDS):
        raise RuntimeError("a proposed source overlaps the existing 30-video benchmark")

    state_path = ROOT / "metadata/source_screening.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "screening", "model": MODEL, "provider": PROVIDER,
        "credential_env": "LIJUYANG_ICLR", "personal_credential_used": False,
        "data_collection": "deny", "reported_cost_usd": 0.0, "screened": [],
    }
    processed = {int(item["source"]["id"]) for item in state["screened"]}
    accepted = [item for item in state["screened"] if item["accepted"]]
    for source_id in CANDIDATE_IDS:
        if len(accepted) >= TARGET_NEW:
            break
        if source_id in processed:
            continue
        row = rows_by_id[source_id]
        source = old.download_source(row)
        sheet = ROOT / "metadata/contact_sheets" / f"short_{source_id:04d}.jpg"
        image_uri, media_probe = old.contact_sheet(source, sheet)
        result, raw = call_openrouter(key, row, image_uri)
        cost = response_cost(raw)
        raw_path = ROOT / "metadata/api_raw" / f"short_{source_id:04d}.json"
        atomic_json(raw_path, raw)
        item = {"source": row, "screening": result, "media_probe": media_probe,
                "accepted": acceptable(result), "reported_cost_usd": cost,
                "raw_record": raw_path.relative_to(ROOT).as_posix()}
        state["screened"].append(item)
        state["reported_cost_usd"] = round(float(state["reported_cost_usd"]) + cost, 8)
        if state["reported_cost_usd"] > HARD_BUDGET_USD:
            state["status"] = "budget_exceeded"
            atomic_json(state_path, state)
            raise RuntimeError("source-screening budget exceeded")
        if item["accepted"]:
            accepted.append(item)
        atomic_json(state_path, state)
        print(f"short:{source_id} accepted={item['accepted']} new={len(accepted)}/{TARGET_NEW} "
              f"cost=${state['reported_cost_usd']:.5f}", flush=True)

    if len(accepted) < TARGET_NEW:
        state["status"] = "insufficient_sources"
        atomic_json(state_path, state)
        raise RuntimeError(f"only {len(accepted)} of {TARGET_NEW} new sources passed")
    selected = prior_reserves(rows_by_id) + accepted[:TARGET_NEW]
    if len({item["source"]["official_sha256"] for item in selected}) != 12:
        raise RuntimeError("selected source hashes are not unique")
    state.update(status="complete", accepted_new=TARGET_NEW, selected_total=12)
    atomic_json(state_path, state)
    atomic_json(ROOT / "metadata/selected_sources.json", selected)
    print(json.dumps({"status": "complete", "selected_ids": [item["source"]["id"] for item in selected],
                      "reported_cost_usd": state["reported_cost_usd"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
