#!/usr/bin/env python3
"""Resumable Gemini API substitute for IVEBench IS/CF and blinded pairwise comparison."""

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


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.8-flash"
PROVIDER_SLUG = "google-ai-studio"
PROVIDER_RESPONSE = "Google AI Studio"
SETTINGS = {
    "model": MODEL,
    "provider": {
        "only": [PROVIDER_SLUG], "order": [PROVIDER_SLUG], "allow_fallbacks": False,
        "require_parameters": True, "data_collection": "deny",
    },
    "reasoning": {"effort": "low"},
    "temperature": 0.0,
    "max_tokens": 1024,
    "stream": False,
}

ABSOLUTE_PROMPT = """You are evaluating instruction-guided video editing. You receive the source video first and the edited video second.

Score two independent IVEBench-style criteria using integers from 1 to 5.

Instruction satisfaction: Does the edited video match the expected result after applying the edit instruction?
1 = completely unrelated or the requested edit is absent; 2 = possibly matches; 3 = somewhat matches; 4 = mostly matches; 5 = perfectly matches.

Content fidelity: Apart from changes required by the instruction, how strictly does the edited video preserve the source video's subjects, background, appearance, composition, motion, and temporal content?
1 = only a small portion is preserved; 2 = about half; 3 = most; 4 = almost all with minor differences; 5 = even small unrelated details are preserved.

Judge the complete visible temporal result. Do not reward realism as a substitute for following the instruction. Do not penalize changes explicitly required by the instruction under content fidelity. Provide a concise reason for each score."""

PAIRWISE_PROMPT = """You are comparing two anonymous outputs of the same instruction-guided video editing task. You receive the source video, Candidate A, and Candidate B in that order.

For instruction preference, choose the candidate that more completely and consistently performs the requested edit.
For fidelity preference, choose the candidate that better preserves every source detail, composition, motion, and temporal behavior not required to change.
For overall preference, balance both criteria and choose the better editing result. Use tie only when there is no meaningful visible difference. Do not infer model identity or favor A/B position. Provide one concise reason."""

SCHEMAS = {
    "absolute": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "instruction_satisfaction": {"type": "integer", "minimum": 1, "maximum": 5},
            "content_fidelity": {"type": "integer", "minimum": 1, "maximum": 5},
            "instruction_reason": {"type": "string"}, "fidelity_reason": {"type": "string"},
        },
        "required": ["instruction_satisfaction", "content_fidelity", "instruction_reason", "fidelity_reason"],
    },
    "pairwise": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "instruction_preference": {"type": "string", "enum": ["A", "B", "tie"]},
            "fidelity_preference": {"type": "string", "enum": ["A", "B", "tie"]},
            "overall_preference": {"type": "string", "enum": ["A", "B", "tie"]},
            "reason": {"type": "string"},
        },
        "required": ["instruction_preference", "fidelity_preference", "overall_preference", "reason"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def protocol(kind: str) -> dict:
    effective_kind = "pairwise" if kind == "pairwise" else "absolute"
    prompt = PAIRWISE_PROMPT if effective_kind == "pairwise" else ABSOLUTE_PROMPT
    return {
        "protocol": f"ivebench_gemini38_flash_low_temp0_{effective_kind}_v1",
        "endpoint": ENDPOINT, "settings": SETTINGS,
        "expected_response_provider": PROVIDER_RESPONSE,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "official_metric_substitution": {
            "instruction_satisfaction": "Gemini 3.8 Flash API replaces official local Qwen2.5-VL-72B",
            "content_fidelity": "Gemini 3.8 Flash API replaces official local Qwen2.5-VL-72B",
        } if effective_kind == "absolute" else None,
        "judge_media": "up to 20 paired frames, max side 426, 5 fps; analogous to official IS/CF preprocessing",
        "cost_basis": "OpenRouter response usage.cost in USD; retries included; missing cost listed separately",
    }


def video_block(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}}


def payload(task: dict) -> dict:
    effective_kind = "pairwise" if task["kind"] == "pairwise" else "absolute"
    schema = SCHEMAS[effective_kind]
    settings = {
        **SETTINGS,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": f"ivebench_{effective_kind}", "strict": True, "schema": schema,
        }},
    }
    if effective_kind == "absolute":
        content = [
            {"type": "text", "text": ABSOLUTE_PROMPT + f"\n\nEdit instruction: {task['instruction']}"},
            {"type": "text", "text": "Source video:"}, video_block(task["source_video"]),
            {"type": "text", "text": "Edited video:"}, video_block(task["edited_video"]),
        ]
    else:
        content = [
            {"type": "text", "text": PAIRWISE_PROMPT + f"\n\nEdit instruction: {task['instruction']}"},
            {"type": "text", "text": "Source video:"}, video_block(task["source_video"]),
            {"type": "text", "text": "Candidate A:"}, video_block(task["candidate_a_video"]),
            {"type": "text", "text": "Candidate B:"}, video_block(task["candidate_b_video"]),
        ]
    return {**settings, "messages": [{"role": "user", "content": content}]}


def validate_score(kind: str, score: object) -> dict:
    effective_kind = "pairwise" if kind == "pairwise" else "absolute"
    if not isinstance(score, dict) or set(score) != set(SCHEMAS[effective_kind]["required"]):
        raise ValueError("response fields do not match strict schema")
    if effective_kind == "absolute":
        for name in ("instruction_satisfaction", "content_fidelity"):
            if type(score[name]) is not int or not 1 <= score[name] <= 5:
                raise ValueError(f"invalid {name}")
        for name in ("instruction_reason", "fidelity_reason"):
            if not isinstance(score[name], str) or not score[name].strip():
                raise ValueError(f"invalid {name}")
    else:
        for name in ("instruction_preference", "fidelity_preference", "overall_preference"):
            if score[name] not in ("A", "B", "tie"):
                raise ValueError(f"invalid {name}")
        if not isinstance(score["reason"], str) or not score["reason"].strip():
            raise ValueError("invalid reason")
    return score


def retryable_provider_error(status: int | None, body: object) -> bool:
    """Recognize provider-routing failures that OpenRouter sometimes reports as HTTP 400.

    Most 4xx responses are permanent request errors and must fail fast.  Google AI
    Studio's transient region-routing failure is the exception observed in this
    evaluation: an identical retry can immediately be routed successfully.
    """
    if status != 400:
        return False
    try:
        message = json.dumps(body, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        return False
    return "user location is not supported for the api use" in message


def task_media_hashes(task: dict) -> dict:
    fields = ["source_video", "edited_video"] if task["kind"] != "pairwise" else [
        "source_video", "candidate_a_video", "candidate_b_video"]
    hashes = {}
    for field in fields:
        path = Path(task[field])
        expected = task[field.replace("video", "sha256")]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"task media changed: {field} {actual} != {expected}")
        hashes[field.replace("video", "sha256")] = actual
    return hashes


def evaluate(task: dict, output: Path, metadata: dict, key: str, max_attempts: int) -> dict:
    identity = {"task": task, "protocol_sha256": digest(metadata), "verified_media": task_media_hashes(task)}
    directory = output / "records" / digest(identity)
    result_path = directory / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("status") != "success" or cached.get("identity") != identity:
            raise ValueError(f"invalid cached result: {result_path}")
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    request_payload = payload(task)
    prior = max((int(path.stem.split("_")[1]) for path in directory.glob("attempt_*.json")), default=0)
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(prior + 1, max_attempts + 1):
            receipt = {
                "attempt": attempt, "created_at": datetime.now(timezone.utc).isoformat(),
                "identity": identity, "status": "request_started_outcome_unknown",
            }
            receipt_path = directory / f"attempt_{attempt:03d}.json"
            write_json(receipt_path, receipt)
            score = None
            body = {}
            try:
                response = session.post(
                    ENDPOINT, headers={"Authorization": "Bearer " + key}, json=request_payload,
                    timeout=(20, 300),
                )
                receipt["http_status"] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {"unparseable_response": True}
                body = json.loads(json.dumps(body).replace(key, "[REDACTED]"))
                receipt.update(status="response_received", response=body, usage=body.get("usage", {}))
                if response.ok:
                    try:
                        if body.get("model") not in (MODEL, "google/gemini-3.8-flash-20260902"):
                            raise ValueError("unexpected returned model")
                        if body.get("provider") != PROVIDER_RESPONSE:
                            raise ValueError("unexpected returned provider")
                        choices = body.get("choices", [])
                        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                            raise ValueError("missing, blocked, or truncated response")
                        score = validate_score(task["kind"], json.loads(choices[0]["message"]["content"]))
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                        receipt["parse_error"] = str(error)
                else:
                    receipt["request_error"] = f"HTTP {response.status_code}"
            except requests.RequestException as error:
                receipt["request_error"] = type(error).__name__
            write_json(receipt_path, receipt)
            if score is not None:
                result = {
                    "status": "success", "identity": identity, "score": score,
                    "successful_attempt": attempt, "returned_model": body["model"],
                    "returned_provider": body["provider"],
                }
                write_json(result_path, result)
                print(f"scored {task['kind']} {task['sample_id']}", flush=True)
                return result
            if (receipt.get("http_status") in (400, 401, 402, 403, 404, 413)
                    and not retryable_provider_error(receipt.get("http_status"), body)):
                break
            if attempt < max_attempts:
                time.sleep(min(40, 5 * 2 ** (attempt - 1)))
    return {"status": "failed", "identity": identity}


def bidirectional_consensus(rows: list[dict]) -> dict:
    """Count a winner only when the A/B and B/A judgments agree after normalization."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sample_id"], row["model_pair"])].append(row)
    models = sorted({
        row[name]
        for row in rows
        for name in ("candidate_a_model", "candidate_b_model")
    })
    outcomes = (*models, "tie", "order_inconsistent")
    counts = Counter()
    by_pair = Counter()
    by_pair_stage = Counter()
    pair_groups = Counter(pair for _sample, pair in grouped)
    pair_stage_groups = Counter()
    for (_sample, pair), values in grouped.items():
        stages = {row["stage"] for row in values}
        stage = next(iter(stages)) if len(stages) == 1 else "malformed_stage"
        pair_stage_groups[(pair, stage)] += 1
        for dimension in ("instruction", "fidelity", "overall"):
            winners = [row[f"normalized_{dimension}_winner"] for row in values]
            outcome = winners[0] if len(winners) == 2 and winners[0] == winners[1] else "order_inconsistent"
            counts[(dimension, outcome)] += 1
            by_pair[(pair, dimension, outcome)] += 1
            by_pair_stage[(pair, stage, dimension, outcome)] += 1
    pair_names = sorted(pair_groups)
    return {
        "groups": len(grouped),
        "counts": {
            dimension: {outcome: counts[(dimension, outcome)] for outcome in outcomes}
            for dimension in ("instruction", "fidelity", "overall")
        },
        "rates": {
            dimension: {outcome: counts[(dimension, outcome)] / len(grouped) for outcome in outcomes}
            for dimension in ("instruction", "fidelity", "overall")
        },
        "by_pair": {
            pair: {
                "groups": pair_groups[pair],
                "counts": {
                    dimension: {
                        outcome: by_pair[(pair, dimension, outcome)]
                        for outcome in (*pair.split("__vs__"), "tie", "order_inconsistent")
                    }
                    for dimension in ("instruction", "fidelity", "overall")
                },
                "rates": {
                    dimension: {
                        outcome: by_pair[(pair, dimension, outcome)] / pair_groups[pair]
                        for outcome in (*pair.split("__vs__"), "tie", "order_inconsistent")
                    }
                    for dimension in ("instruction", "fidelity", "overall")
                },
            }
            for pair in pair_names
        },
        "by_pair_stage": {
            pair: {
                stage: {
                    "groups": pair_stage_groups[(pair, stage)],
                    "counts": {
                        dimension: {
                            outcome: by_pair_stage[(pair, stage, dimension, outcome)]
                            for outcome in (*pair.split("__vs__"), "tie", "order_inconsistent")
                        }
                        for dimension in ("instruction", "fidelity", "overall")
                    },
                    "rates": {
                        dimension: {
                            outcome: by_pair_stage[(pair, stage, dimension, outcome)] / pair_stage_groups[(pair, stage)]
                            for outcome in (*pair.split("__vs__"), "tie", "order_inconsistent")
                        }
                        for dimension in ("instruction", "fidelity", "overall")
                    },
                }
                for stage in sorted(stage for candidate_pair, stage in pair_stage_groups if candidate_pair == pair)
            }
            for pair in pair_names
        },
    }


def summarize(output: Path, tasks: list[dict], results: list[dict]) -> dict:
    successful = [result for result in results if result["status"] == "success"]
    receipts = [json.loads(path.read_text()) for path in (output / "records").glob("*/attempt_*.json")]
    costs = [receipt.get("usage", {}).get("cost") for receipt in receipts]
    summary = {
        "status": "complete" if len(successful) == len(tasks) else "incomplete",
        "tasks_requested": len(tasks), "tasks_succeeded": len(successful),
        "tasks_failed": len(tasks) - len(successful),
        "failed_task_ids": [result["identity"]["task"]["sample_id"] for result in results if result["status"] != "success"],
        "requests_including_retries": len(receipts),
        "reported_cost_usd": sum(value for value in costs if isinstance(value, (int, float))),
        "requests_without_reported_cost": sum(not isinstance(value, (int, float)) for value in costs),
        "usage_totals": {name: sum(receipt.get("usage", {}).get(name, 0) for receipt in receipts)
                         for name in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }
    rows = []
    if successful and tasks[0]["kind"] != "pairwise":
        groups = defaultdict(list)
        for result in successful:
            task = result["identity"]["task"]
            row = {**task, **result["score"]}
            rows.append(row)
            groups[task["model_kind"]].append(result["score"])
        summary["model_means"] = {
            model: {metric: sum(row[metric] for row in values) / len(values)
                    for metric in ("instruction_satisfaction", "content_fidelity")}
            for model, values in sorted(groups.items())
        }
    elif successful:
        normalized = Counter()
        normalized_by_pair = Counter()
        candidate_models = sorted({
            task[name]
            for task in tasks
            for name in ("candidate_a_model", "candidate_b_model")
        })
        for result in successful:
            task = result["identity"]["task"]
            row = {**task, **result["score"]}
            for dimension in ("instruction", "fidelity", "overall"):
                preference = result["score"][f"{dimension}_preference"]
                winner = "tie" if preference == "tie" else task[f"candidate_{preference.lower()}_model"]
                row[f"normalized_{dimension}_winner"] = winner
                normalized[(dimension, winner)] += 1
                pair_name = task.get("model_pair", "__vs__".join(sorted((task["candidate_a_model"], task["candidate_b_model"]))))
                normalized_by_pair[(pair_name, dimension, winner)] += 1
            rows.append(row)
        summary["normalized_preferences"] = {
            dimension: {winner: normalized[(dimension, winner)] for winner in (*candidate_models, "tie")}
            for dimension in ("instruction", "fidelity", "overall")
        }
        pair_names = sorted({task.get("model_pair", "__vs__".join(sorted((task["candidate_a_model"], task["candidate_b_model"])))) for task in tasks})
        summary["normalized_preferences_by_pair"] = {
            pair_name: {
                dimension: {
                    winner: normalized_by_pair[(pair_name, dimension, winner)]
                    for winner in (*pair_name.split("__vs__"), "tie")
                }
                for dimension in ("instruction", "fidelity", "overall")
            }
            for pair_name in pair_names
        }
        by_sample = defaultdict(list)
        for row in rows:
            pair = tuple(sorted((row["candidate_a_model"], row["candidate_b_model"])))
            by_sample[(row["sample_id"], pair)].append(row)
        summary["order_consistency"] = {
            dimension: sum(len(values) == 2 and values[0][f"normalized_{dimension}_winner"] == values[1][f"normalized_{dimension}_winner"]
                           for values in by_sample.values()) / len(by_sample)
            for dimension in ("instruction", "fidelity", "overall")
        }
        summary["bidirectional_consensus"] = bidirectional_consensus(rows)
    if rows:
        temporary = output / "scores.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(output / "scores.csv")
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OPEN_ROUTER_QWEN")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 8 or (args.limit is not None and args.limit < 1):
        parser.error("invalid retry or limit count")
    tasks = json.loads(args.tasks.read_text())
    manifest_count = len(tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks or len({digest(task) for task in tasks}) != len(tasks):
        raise ValueError("empty or duplicate tasks")
    kinds = {task["kind"] for task in tasks}
    if len(kinds) != 1 or not kinds <= {"absolute", "pairwise", "stability"}:
        raise ValueError(f"unexpected mixed task kinds: {kinds}")
    kind = next(iter(kinds))
    metadata = protocol(kind)
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise RuntimeError(f"missing {args.api_key_env}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = args.output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != metadata:
            raise ValueError("protocol changed; use a new output directory")
        write_json(protocol_path, metadata)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda task: evaluate(task, args.output, metadata, key, args.max_attempts), tasks))
        summary = summarize(args.output, tasks, results)
        summary.update(
            manifest_task_count=manifest_count,
            scope="preflight" if args.limit else "full_manifest",
            api_key_env=args.api_key_env,
        )
        if args.limit and summary["status"] == "complete":
            summary["status"] = "preflight_complete"
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return int(summary["tasks_failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
