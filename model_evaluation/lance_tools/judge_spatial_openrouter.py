#!/usr/bin/env python3
"""Resumable official OmniEdit Spatial scoring via one fixed OpenRouter route."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import requests

OFFICIAL = Path(__file__).resolve().parents[1] / "lance_base/runtime/omni_edit_evaluator"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.8-flash"
PROVIDER = "Google AI Studio"
PROTOCOL_ID = "omni_edit_spatial_gemini38_low_openrouter_v1"
DIMENSIONS = ("accuracy", "preservation", "realism", "consistency")
PROMPT_SHA = "8168fe4b4edf0f608e79476fd6b9edf28cb1ede6e6734184f1b12d65c2dedbaf"
EVALUATOR_SHA = "e3fd80f726c33fe4737267161f3440e1eed9d5c79f435e15ec924b4f3a6cc1dd"
SETTINGS = {
    "model": MODEL,
    "provider": {"only": ["google-ai-studio"], "order": ["google-ai-studio"],
                 "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"},
    "reasoning": {"effort": "low"},
    "max_tokens": 2048,
    "stream": False,
    "response_format": {"type": "json_schema", "json_schema": {
        "name": "omni_edit_spatial", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {
                       **{k: {"type": "integer", "minimum": 0, "maximum": 5} for k in DIMENSIONS},
                       "reason": {"type": "string"}},
                   "required": [*DIMENSIONS, "reason"]}}},
}


def configure_route(route: str, temperature_zero: bool = False) -> None:
    """Choose one explicit route and sampling protocol; never fall back providers."""
    global PROVIDER, PROTOCOL_ID
    if route == "studio":
        slug = "google-ai-studio"
        PROVIDER = "Google AI Studio"
        base_protocol_id = "omni_edit_spatial_gemini38_low_openrouter_v1"
    elif route == "vertex":
        slug = "google-vertex/global"
        PROVIDER = "Google"
        base_protocol_id = "omni_edit_spatial_gemini38_low_vertex_v2"
    else:
        raise ValueError("unknown provider route")
    SETTINGS["provider"] = {**SETTINGS["provider"], "only": [slug], "order": [slug]}
    if temperature_zero:
        SETTINGS["temperature"] = 0.0
        PROTOCOL_ID = f"{base_protocol_id}_temp0_v1"
    else:
        SETTINGS.pop("temperature", None)
        PROTOCOL_ID = base_protocol_id


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def protocol() -> tuple[dict, str]:
    prompt_path = OFFICIAL / "eval_configs/prompts.json"
    if sha256_file(prompt_path) != PROMPT_SHA or sha256_file(OFFICIAL / "gemini_eval.py") != EVALUATOR_SHA:
        raise ValueError("frozen official evaluator or prompt changed")
    prompt = json.loads(prompt_path.read_text())
    rules = f"{prompt['base_instruction']}\n{prompt['tracks']['spatial']}\n{prompt['output_format']}"
    return {
        "protocol": PROTOCOL_ID,
        "endpoint": ENDPOINT, "settings": SETTINGS,
        "expected_response_provider": PROVIDER,
        "official_prompt_sha256": PROMPT_SHA, "official_evaluator_sha256": EVALUATOR_SHA,
        "system_rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "media_resolution": "provider_default_not_explicitly_controllable_via_this_route",
        "video_input": "unaltered_complete_source_mp4_then_generated_mp4_base64",
        "cost_basis": "OpenRouter returned usage.cost in USD, including retries; unknown charges reported separately",
    }, rules


def payload_for(task: dict, rules: str) -> dict:
    content = [{"type": "text", "text":
                f"Evaluation System Rules:\n{rules}\n\nUser Edit Instruction: {task['instruction']}\n"}]
    for label, field in (("Original Video (Reference):", "source_path"),
                         ("Edited Video (Target):", "generated_video")):
        content += [{"type": "text", "text": label}, {"type": "video_url", "video_url": {
            "url": "data:video/mp4;base64," + base64.b64encode(Path(task[field]).read_bytes()).decode("ascii")}}]
    return {**SETTINGS, "messages": [{"role": "user", "content": content}]}


def parse_score(body: dict) -> dict:
    if body.get("model") not in (MODEL, "google/gemini-3.8-flash-20260902"):
        raise ValueError("unexpected returned model")
    if body.get("provider") != PROVIDER:
        raise ValueError("unexpected returned provider")
    choices = body.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("missing, blocked, truncated or non-stop response")
    score = json.loads(choices[0]["message"]["content"])
    if not isinstance(score, dict) or not isinstance(score.get("reason"), str) or not score["reason"].strip():
        raise ValueError("missing reason")
    for key in DIMENSIONS:
        if type(score.get(key)) is not int or not 0 <= score[key] <= 5:
            raise ValueError(f"{key} must be an integer 0..5")
    score = {key: score[key] for key in (*DIMENSIONS, "reason")}
    acc = score["accuracy"]
    penalty = acc * 0.2 if acc <= 5 else 1.0
    score["overall_score"] = round(0.5 * acc + 0.2 * score["preservation"] * penalty
                                   + 0.15 * score["realism"] * penalty
                                   + 0.15 * score["consistency"] * penalty, 4)
    return score


def evaluate(task: dict, output: Path, metadata: dict, rules: str, key: str, max_attempts: int) -> dict:
    identity = {"task": task, "protocol_sha256": digest(metadata),
                "source_sha256": sha256_file(Path(task["source_path"])),
                "generated_sha256": sha256_file(Path(task["generated_video"]))}
    for field in ("source_sha256", "generated_sha256"):
        if task.get(field) and task[field] != identity[field]:
            raise ValueError(f"{task['sample_id']}: {field} mismatch")
    directory = output / "records" / digest(identity)
    result_path = directory / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("identity") != identity or cached.get("status") != "success":
            raise ValueError("invalid result cache")
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    payload = payload_for(task, rules)
    # Persist an attempt marker BEFORE a paid request. Interrupted/unknown requests remain visible.
    prior = max((int(p.stem.split("_")[1]) for p in directory.glob("attempt_*.json")), default=0)
    with requests.Session() as session:
        session.trust_env = False
        for number in range(prior + 1, max_attempts + 1):
            receipt = {"attempt": number, "created_at": datetime.now(timezone.utc).isoformat(),
                       "identity": identity, "status": "request_started_outcome_unknown"}
            receipt_path = directory / f"attempt_{number:03d}.json"
            write_json(receipt_path, receipt)
            score = None
            try:
                response = session.post(ENDPOINT, headers={"Authorization": "Bearer " + key},
                                        json=payload, timeout=(20, 300))
                receipt["http_status"] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {"unparseable_response": True}
                body = json.loads(json.dumps(body).replace(key, "[REDACTED]"))
                receipt.update(status="response_received", response=body, usage=body.get("usage", {}))
                if response.ok:
                    try:
                        score = parse_score(body)
                    except (ValueError, KeyError, TypeError) as error:
                        receipt["parse_error"] = str(error)
                else:
                    receipt["request_error"] = f"HTTP {response.status_code}"
            except requests.RequestException as error:
                receipt["request_error"] = type(error).__name__
            write_json(receipt_path, receipt)
            if score is not None:
                result = {"status": "success", "identity": identity, "score": score,
                          "successful_attempt": number,
                          "returned_model": body["model"], "returned_provider": body["provider"]}
                write_json(result_path, result)
                print(f"scored {task['sample_id']} overall={score['overall_score']}", flush=True)
                return result
            if receipt.get("http_status") in (400, 401, 402, 403, 404, 413):
                break
            if number < max_attempts:
                time.sleep(min(40, 5 * 2 ** (number - 1)))
    return {"status": "failed", "identity": identity}


def summarize(output: Path, tasks: list[dict], results: list[dict]) -> dict:
    successful = [r for r in results if r["status"] == "success"]
    rows = [{**r["identity"]["task"], **r["score"]} for r in successful]
    if rows:
        temporary = output / "scores.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "scores.csv")
    receipts = [json.loads(p.read_text()) for p in (output / "records").glob("*/attempt_*.json")]
    costs = [r.get("usage", {}).get("cost") for r in receipts]
    request_counts = Counter(digest(r["identity"]) for r in receipts)
    categories = defaultdict(list)
    for row in rows:
        categories[row.get("category", "unspecified")].append(row)
    def means(group):
        return {k: sum(r[k] for r in group) / len(group) for k in (*DIMENSIONS, "overall_score")} if group else {}
    summary = {
        "status": "complete" if len(successful) == len(tasks) else "incomplete",
        "tasks_requested": len(tasks), "tasks_succeeded": len(successful),
        "tasks_failed": len(tasks) - len(successful),
        "failed_task_ids": [r["identity"]["task"]["sample_id"] for r in results if r["status"] != "success"],
        "mean_scores_successes_only": means(rows),
        "categories": {k: {"count": len(v), **means(v)} for k, v in sorted(categories.items())},
        "all_recorded_requests_including_retries": len(receipts),
        "retry_requests": sum(max(0, n - 1) for n in request_counts.values()),
        "usage_totals": {k: sum(r.get("usage", {}).get(k, 0) for r in receipts)
                         for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "reasoning_tokens": sum(r.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                                for r in receipts),
        "reported_cost_usd_known_requests": sum(c for c in costs if isinstance(c, (float, int))),
        "requests_without_reported_cost": sum(not isinstance(c, (float, int)) for c in costs),
        "returned_models": sorted({r["returned_model"] for r in successful}),
        "returned_providers": sorted({r["returned_provider"] for r in successful}),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OPEN_ROUTER_QWEN")
    parser.add_argument("--route", choices=("studio", "vertex"), default="studio",
                        help="explicit fixed route; formal full Spatial uses vertex and a separate v2 output directory")
    parser.add_argument("--temperature-zero", action="store_true",
                        help="send temperature=0.0 and create a distinct, non-combinable protocol")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-attempts", type=int, default=4, help="total attempts per identity, including prior runs")
    parser.add_argument("--limit", type=int, help="preflight only; successful results are reused by the full run")
    args = parser.parse_args()
    # Default stays four. Explicit, cooled-down recovery may add one attempt at
    # a time after the initial eight; keep a finite lifetime ceiling of twelve.
    if not 1 <= args.max_attempts <= 12 or (args.limit is not None and args.limit < 1):
        parser.error("invalid attempt/limit count")
    configure_route(args.route, temperature_zero=args.temperature_zero)
    metadata, rules = protocol()
    tasks = json.loads(args.tasks.read_text())
    manifest_task_count = len(tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    identities = [(t.get("stage"), t.get("checkpoint_step"), t["sample_id"]) for t in tasks]
    if not tasks or len(set(identities)) != len(tasks):
        raise ValueError("empty or duplicate stage/checkpoint/sample tasks")
    for task in tasks:
        for field in ("source_path", "generated_video"):
            if not Path(task[field]).is_file():
                raise FileNotFoundError(task[field])
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise RuntimeError(f"missing {args.api_key_env}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = args.output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != metadata:
            raise ValueError("protocol changed: use a new output directory")
        write_json(protocol_path, metadata)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda t: evaluate(t, args.output, metadata, rules, key, args.max_attempts), tasks))
        summary = summarize(args.output, tasks, results)
        summary["manifest_task_count"] = manifest_task_count
        summary["scope"] = "preflight" if args.limit else "full_manifest"
        if args.limit and summary["status"] == "complete":
            summary["status"] = "preflight_complete"
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return int(summary["tasks_failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
