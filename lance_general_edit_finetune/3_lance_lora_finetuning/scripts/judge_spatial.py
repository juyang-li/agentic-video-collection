#!/usr/bin/env python3
"""Resumable Gemini Spatial judge using the frozen official prompt and score formula."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parents[1] / "model_evaluation/lance_base/runtime/omni_edit_evaluator"
MODEL = "gemini-3.8-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
DIMENSIONS = ("accuracy", "preservation", "realism", "consistency")
GENERATION_CONFIG = {
    "thinkingConfig": {"thinkingLevel": "low"},
    "mediaResolution": "MEDIA_RESOLUTION_LOW",
    "responseMimeType": "application/json",
    "maxOutputTokens": 2048,
    "responseSchema": {
        "type": "OBJECT",
        "properties": {
            **{key: {"type": "INTEGER", "minimum": 0, "maximum": 5} for key in DIMENSIONS},
            "reason": {"type": "STRING"},
        },
        "required": [*DIMENSIONS, "reason"],
    },
}
PRICING = {
    "source": "https://ai.google.dev/gemini-api/docs/pricing",
    "verified_on": "2026-09-05",
    "effective_until": "2026-12-31",
    "input_usd_per_million": 0.75,
    "cached_input_usd_per_million": 0.075,
    "output_including_thoughts_usd_per_million": 3.75,
    "basis": "paid-tier list price estimate from returned usage, not a billing invoice",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def protocol() -> tuple[dict, str]:
    prompt_path = OFFICIAL / "eval_configs/prompts.json"
    prompt = json.loads(prompt_path.read_text())
    rules = f"{prompt['base_instruction']}\n{prompt['tracks']['spatial']}\n{prompt['output_format']}"
    metadata = {
        "protocol": "omni_edit_spatial_gemini38_low_v1",
        "model": MODEL, "endpoint": ENDPOINT,
        "generation_config": GENERATION_CONFIG,
        "official_prompt_sha256": sha256_file(prompt_path),
        "official_evaluator_sha256": sha256_file(OFFICIAL / "gemini_eval.py"),
        "system_rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "pricing": PRICING,
    }
    return metadata, rules


def parse_score(response: dict) -> dict:
    candidates = response.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("missing, blocked, truncated, or non-STOP candidate")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    score = json.loads(text)
    if not isinstance(score, dict) or not isinstance(score.get("reason"), str) or not score["reason"].strip():
        raise ValueError("missing score reason")
    for key in DIMENSIONS:
        if type(score.get(key)) is not int or not 0 <= score[key] <= 5:
            raise ValueError(f"invalid {key}: expected integer 0..5")
    score = {key: score[key] for key in (*DIMENSIONS, "reason")}
    accuracy = score["accuracy"]
    # Exactly match the official evaluator, including its accuracy penalty.
    penalty = accuracy * 0.2 if accuracy <= 5 else 1.0
    score["overall_score"] = round(
        0.5 * accuracy + 0.2 * score["preservation"] * penalty
        + 0.15 * score["realism"] * penalty + 0.15 * score["consistency"] * penalty, 4
    )
    return score


def cost_estimate(usage: dict, date: str) -> float | None:
    if not usage or date > PRICING["effective_until"]:
        return None
    prompt = usage.get("promptTokenCount", 0)
    cached = usage.get("cachedContentTokenCount", 0)
    output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return ((prompt - cached) * 0.75 + cached * 0.075 + output * 3.75) / 1_000_000


def evaluate(task: dict, output: Path, metadata: dict, rules: str, api_key: str, attempts: int) -> dict:
    source, generated = Path(task["source_path"]), Path(task["generated_video"])
    identity = {
        "task": task, "protocol_sha256": canonical_hash(metadata),
        "source_sha256": sha256_file(source), "generated_sha256": sha256_file(generated),
    }
    signature = canonical_hash(identity)
    directory = output / "records" / signature
    result_path = directory / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("identity") != identity or previous.get("status") != "success":
            raise ValueError(f"invalid cached result: {signature}")
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    parts = [{"text": f"Evaluation System Rules:\n{rules}\n\nUser Edit Instruction: {task['instruction']}\n"}]
    for label, path in (("Original Video (Reference):", source), ("Edited Video (Target):", generated)):
        parts.extend([{"text": label}, {"inline_data": {
            "mime_type": "video/mp4", "data": base64.b64encode(path.read_bytes()).decode("ascii")
        }}])
    payload = {"contents": [{"parts": parts}], "generationConfig": GENERATION_CONFIG}
    prior_attempts = len(list(directory.glob("attempt_*.json")))
    session = requests.Session()
    session.trust_env = False
    for retry in range(attempts):
        number = prior_attempts + retry + 1
        timestamp = datetime.now(timezone.utc).isoformat()
        receipt = {"attempt": number, "created_at": timestamp, "identity": identity}
        score = None
        try:
            response = session.post(ENDPOINT, headers={"x-goog-api-key": api_key}, json=payload, timeout=(20, 300))
            receipt["http_status"] = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {"unparseable_response": True}
            # Credentials travel only in the request header; redact any accidental echo.
            body = json.loads(json.dumps(body).replace(api_key, "[REDACTED]"))
            receipt["response"] = body
            usage = body.get("usageMetadata", {})
            receipt["usageMetadata"] = usage
            receipt["estimated_cost_usd"] = cost_estimate(usage, timestamp[:10])
            if response.ok:
                try:
                    score = parse_score(body)
                except (ValueError, KeyError, TypeError) as error:
                    receipt["parse_error"] = str(error)
            else:
                receipt["request_error"] = f"HTTP {response.status_code}"
        except requests.RequestException as error:
            receipt["request_error"] = type(error).__name__
        write_json(directory / f"attempt_{number:03d}.json", receipt)
        if score is not None:
            result = {"status": "success", "identity": identity, "score": score, "successful_attempt": number}
            write_json(result_path, result)
            print(f"scored step={task['checkpoint_step']} sample={task['sample_id']} score={score['overall_score']}", flush=True)
            return result
        if receipt.get("http_status") in (400, 401, 403, 404):
            break
        if retry + 1 < attempts:
            time.sleep(min(40, 5 * 2**retry))
    return {"status": "failed", "identity": identity}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inspect-config", action="store_true")
    args = parser.parse_args()
    metadata, rules = protocol()
    if args.inspect_config:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    if not 1 <= args.workers <= 2 or not 1 <= args.attempts <= 8:
        parser.error("workers must be 1..2 and attempts 1..8")
    if args.tasks is None:
        parser.error("--tasks is required")
    tasks = json.loads(args.tasks.read_text())
    ids = [(t["stage"], t["checkpoint_step"], t["sample_id"]) for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation task")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError("no evaluation tasks")
    for task in tasks:
        for field in ("source_path", "generated_video"):
            if not Path(task[field]).is_file():
                raise FileNotFoundError(task[field])
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing environment variable {args.api_key_env}")
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != metadata:
        raise ValueError("protocol changed: use a new output directory")
    write_json(protocol_path, metadata)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda task: evaluate(task, args.output, metadata, rules, api_key, args.attempts), tasks))
    successful = [r for r in results if r["status"] == "success"]
    score_rows = [{**r["identity"]["task"], **r["score"]} for r in successful]
    if score_rows:
        temporary = args.output / "scores.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
            writer.writeheader()
            writer.writerows(score_rows)
        temporary.replace(args.output / "scores.csv")
    receipts = [json.loads(p.read_text()) for p in (args.output / "records").glob("*/attempt_*.json")]
    costs = [r.get("estimated_cost_usd") for r in receipts]
    summary = {
        "tasks_requested": len(tasks), "tasks_succeeded": len(successful),
        "tasks_failed": len(tasks) - len(successful),
        "failed_task_ids": [r["identity"]["task"]["sample_id"] for r in results if r["status"] != "success"],
        "all_recorded_requests_including_retries": len(receipts),
        "usage_totals": {k: sum(r.get("usageMetadata", {}).get(k, 0) for r in receipts)
                         for k in ("promptTokenCount", "cachedContentTokenCount", "candidatesTokenCount", "thoughtsTokenCount")},
        "estimated_paid_tier_cost_usd_known_usage": sum(c for c in costs if c is not None),
        "requests_without_cost_estimate": sum(c is None for c in costs),
        "pricing": PRICING,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return int(len(successful) != len(tasks))


if __name__ == "__main__":
    raise SystemExit(main())
