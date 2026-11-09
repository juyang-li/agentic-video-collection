#!/usr/bin/env python3
"""Run a resumable Qwen2.5-VL full-pair editing-success judge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SPECIALIST_ROOT = PROJECT_ROOT / "1_specialist_filtering"
DEFAULT_INPUT = SPECIALIST_ROOT / "artifacts/filter_run_v2/stage1_next_input.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/vlm_run_v1/qwen_pair_scores.csv"
DEFAULT_SUMMARY = ROOT / "artifacts/vlm_run_v1/qwen_pair_score_summary.json"
DEFAULT_PROMPT = ROOT / "judge_prompt_v1.txt"
JUDGE_VERSION = "qwen2_5_vl_7b_pair_judge_v1"
VALID_DECISIONS = {"accepted", "rejected", "uncertain"}
VALID_CHECKS = {"pass", "fail", "uncertain"}
CHECK_FIELDS = [
    "instruction_following",
    "edit_completeness",
    "content_preservation",
    "temporal_consistency",
    "visible_artifacts",
]
FIELDS = [
    "judge_version",
    "model_id",
    "model_backend",
    "prompt_sha256",
    "sample_id",
    "dataset",
    "branch",
    "operation",
    "target_scope",
    "original_stage1_1_decision",
    "sampled_source_frames",
    "sampled_target_frames",
    "input_tokens",
    "peak_gpu_memory_mb",
    "model_decision",
    *CHECK_FIELDS,
    "reason_codes_json",
    "model_confidence",
    "short_evidence",
    "parse_status",
    "raw_response",
    "error",
    "elapsed_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_media(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SPECIALIST_ROOT / path


def resize_long_edge(frame: np.ndarray, maximum: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def uniform_frames(path: Path, count: int, max_long_edge: int):
    from decord import VideoReader, cpu
    from PIL import Image

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(reader) < 1:
        raise ValueError(f"empty video: {path}")
    indices = np.unique(np.linspace(0, len(reader) - 1, min(count, len(reader))).round().astype(int))
    frames = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(resize_long_edge(frame, max_long_edge)) for frame in frames]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I | re.S)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")


def normalize_judgment(value: dict[str, Any]) -> dict[str, Any]:
    decision = str(value.get("decision", "uncertain")).lower().strip()
    if decision not in VALID_DECISIONS:
        decision = "uncertain"
    result: dict[str, Any] = {"decision": decision}
    for field in CHECK_FIELDS:
        check = str(value.get(field, "uncertain")).lower().strip()
        result[field] = check if check in VALID_CHECKS else "uncertain"
    raw_codes = value.get("reason_codes", [])
    if isinstance(raw_codes, str):
        raw_codes = [raw_codes]
    result["reason_codes"] = [
        re.sub(r"[^a-z0-9_]+", "_", str(code).lower()).strip("_")[:80]
        for code in raw_codes
        if str(code).strip()
    ][:8]
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = min(1.0, max(0.0, confidence))
    result["short_evidence"] = str(value.get("short_evidence", "")).strip()[:500]
    if any(result[field] == "fail" for field in CHECK_FIELDS) and result["decision"] == "accepted":
        result["decision"] = "uncertain"
        result["reason_codes"].append("inconsistent_checklist")
    return result


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class QwenPairJudge:
    def __init__(self, model_id: str, device: str, local_files_only: bool) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, attn_implementation="flash_attention_2", **kwargs
            )
        except (ImportError, ValueError):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, attn_implementation="sdpa", **kwargs
            )
        self.model.to(self.device).eval()

    def judge(
        self,
        source_frames: list[Any],
        target_frames: list[Any],
        instruction: str,
        operation: str,
        rubric: str,
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        content: list[dict[str, Any]] = [{"type": "text", "text": "SOURCE frames, in time order:"}]
        content.extend({"type": "image", "image": image} for image in source_frames)
        content.append({"type": "text", "text": "TARGET frames, in matching time order:"})
        content.extend({"type": "image", "image": image} for image in target_frames)
        content.append(
            {
                "type": "text",
                "text": f"Operation: {operation}\nInstruction: {instruction}\n\nRubric:\n{rubric}",
            }
        )
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = source_frames + target_frames
        inputs = self.processor(text=[rendered], images=images, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        continuation = generated[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        peak_mb = (
            self.torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )
        return response, int(inputs["input_ids"].shape[1]), float(peak_mb)


def mock_response(row: dict[str, str]) -> str:
    digest = hashlib.sha256(row["sample_id"].encode()).digest()
    unit = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    if unit < 0.2:
        decision, check, reason = "rejected", "fail", "mock_instruction_failure"
    elif unit < 0.4:
        decision, check, reason = "uncertain", "uncertain", "mock_insufficient_evidence"
    else:
        decision, check, reason = "accepted", "pass", "mock_pass"
    return json.dumps(
        {
            "decision": decision,
            "instruction_following": check,
            "edit_completeness": check,
            "content_preservation": check,
            "temporal_consistency": check,
            "visible_artifacts": check,
            "reason_codes": [reason],
            "confidence": round(0.55 + 0.44 * abs(unit - 0.3), 3),
            "short_evidence": "Deterministic mock response for pipeline validation.",
        }
    )


def blank_result(row: dict[str, str], model_id: str, backend: str, rubric_hash: str) -> dict[str, str]:
    result = {field: "" for field in FIELDS}
    result.update(
        {
            "judge_version": JUDGE_VERSION,
            "model_id": model_id,
            "model_backend": backend,
            "prompt_sha256": rubric_hash,
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "branch": row["branch"],
            "operation": row["operation"],
            "target_scope": row.get("target_scope", ""),
            "original_stage1_1_decision": row.get("stage1_1_decision", ""),
        }
    )
    return result


def score_row(
    row: dict[str, str],
    judge: QwenPairJudge | None,
    model_id: str,
    rubric: str,
    frame_samples: int,
    max_long_edge: int,
    max_new_tokens: int,
    mock: bool,
) -> dict[str, str]:
    started = time.monotonic()
    backend = "mock" if mock else "transformers"
    result = blank_result(row, model_id, backend, prompt_hash(rubric))
    try:
        if mock:
            raw = mock_response(row)
            result["sampled_source_frames"] = str(frame_samples)
            result["sampled_target_frames"] = str(frame_samples)
            result["input_tokens"] = "0"
            result["peak_gpu_memory_mb"] = "0.000"
        else:
            source = uniform_frames(resolve_media(row["source_local"]), frame_samples, max_long_edge)
            target = uniform_frames(resolve_media(row["target_local"]), frame_samples, max_long_edge)
            result["sampled_source_frames"] = str(len(source))
            result["sampled_target_frames"] = str(len(target))
            raw, input_tokens, peak_mb = judge.judge(  # type: ignore[union-attr]
                source, target, row["instruction"], row["operation"], rubric, max_new_tokens
            )
            result["input_tokens"] = str(input_tokens)
            result["peak_gpu_memory_mb"] = f"{peak_mb:.3f}"
        result["raw_response"] = raw
        normalized = normalize_judgment(extract_json_object(raw))
        result.update(
            {
                "model_decision": normalized["decision"],
                **{field: normalized[field] for field in CHECK_FIELDS},
                "reason_codes_json": json.dumps(normalized["reason_codes"], ensure_ascii=False),
                "model_confidence": f"{normalized['confidence']:.6f}",
                "short_evidence": normalized["short_evidence"],
                "parse_status": "pass",
            }
        )
    except Exception as error:
        result["model_decision"] = "uncertain"
        result["parse_status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = f"{time.monotonic() - started:.6f}"
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-samples", type=int, default=8)
    parser.add_argument("--max-long-edge", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    rubric = args.prompt.read_text(encoding="utf-8").strip()
    rows = read_csv(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing: dict[str, dict[str, str]] = {}
    if args.resume and args.output.is_file():
        existing = {
            row["sample_id"]: row
            for row in read_csv(args.output)
            if row.get("parse_status") in {"pass", "error"}
        }
    pending = [row for row in rows if row["sample_id"] not in existing]
    judge = None if args.mock_model else QwenPairJudge(
        args.model, args.device, args.local_files_only
    )
    results = list(existing.values())
    run_started = time.monotonic()
    for index, row in enumerate(pending, 1):
        results.append(
            score_row(
                row,
                judge,
                args.model,
                rubric,
                args.frame_samples,
                args.max_long_edge,
                args.max_new_tokens,
                args.mock_model,
            )
        )
        write_csv(args.output, results)
        print(f"qwen_judged={index}/{len(pending)} total={len(results)}", flush=True)

    elapsed = time.monotonic() - run_started
    summary = {
        "judge_version": JUDGE_VERSION,
        "model_id": args.model,
        "model_backend": "mock" if args.mock_model else "transformers",
        "prompt_sha256": prompt_hash(rubric),
        "input_rows": len(rows),
        "completed": len(results),
        "parse_errors": sum(row.get("parse_status") != "pass" for row in results),
        "decision_counts": {
            decision: sum(row.get("model_decision") == decision for row in results)
            for decision in sorted(VALID_DECISIONS)
        },
        "new_run_elapsed_seconds": elapsed,
        "seconds_per_new_pair": elapsed / len(pending) if pending else 0.0,
        "mean_input_tokens": (
            sum(int(row.get("input_tokens") or 0) for row in results) / len(results)
            if results else 0.0
        ),
        "peak_gpu_memory_mb": max(
            (float(row.get("peak_gpu_memory_mb") or 0) for row in results), default=0.0
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
