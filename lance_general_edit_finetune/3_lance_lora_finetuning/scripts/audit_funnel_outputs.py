#!/usr/bin/env python3
"""Verify frozen jobs, adapters and current video bytes against full-decode inventories; no inference/API."""

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

from judge_spatial import sha256_file, write_json
from run_funnel_evaluation import BASE, COUNTS, EVALUATION, ROOT, verify_contract


@lru_cache(maxsize=None)
def digest(path: Path) -> str:
    return sha256_file(path)


def audit(evaluation: Path) -> dict:
    contract = verify_contract(evaluation)
    decision = json.loads((evaluation / "promotion_decision.json").read_text())
    if decision["promotion_steps"] != [677]:
        raise ValueError("frozen winner changed")
    base = json.loads((BASE / "run.json").read_text())
    weight = ROOT / "runtime/lance/downloads/Lance_3B_Video/model.safetensors"
    if digest(weight) != base["main_weight_sha256"]:
        raise ValueError("finetune inference base weight differs from Base evaluation")
    upstream = BASE / "runtime/upstream/inference_lance.py"
    local = ROOT / "runtime/lance/inference_lance.py"
    if digest(local) != digest(upstream):
        raise ValueError("shared underlying inference implementation differs")
    stages, checked_paths = {}, set()
    for stage, count in COUNTS.items():
        jobs_path = evaluation / f"{stage}_jobs.json"
        tasks_path = evaluation / f"{stage}_generation_tasks.json"
        specification = json.loads(jobs_path.read_text())
        tasks = json.loads(tasks_path.read_text())
        if (len(tasks) != count or specification["generation_count"] != count
                or specification["generation_seed"] != 42 or specification["shards_per_checkpoint"] != 4):
            raise ValueError("frozen stage specification mismatch")
        job_names = {job["run_name"] for job in specification["jobs"]}
        if len(job_names) != len(specification["jobs"]) or set(t["run_name"] for t in tasks) != job_names:
            raise ValueError("task/job membership mismatch")
        for step in specification["checkpoint_steps"]:
            if sorted(j["gpu_slot"] for j in specification["jobs"] if j["checkpoint_step"] == step) != [0, 1, 2, 3]:
                raise ValueError("checkpoint did not have four unique GPU shards")
        checked, inventories, adapter_hashes = 0, {}, {}
        for job in specification["jobs"]:
            name = job["run_name"]
            checkpoint = Path(job["adapter_checkpoint"])
            marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
            if marker["status"] != "complete" or marker["completed_steps"] != job["checkpoint_step"]:
                raise ValueError("wrong or incomplete checkpoint")
            config_path = Path(job["config_path"])
            config = json.loads(config_path.read_text())
            output = evaluation / "outputs_raw" / name
            adapter_sha = digest(checkpoint / "adapter.safetensors")
            expected_provenance = {"job": job, "adapter_sha256": adapter_sha,
                                   "config_sha256": digest(config_path), "seed": 42}
            if json.loads((output / "generation_provenance.json").read_text()) != expected_provenance:
                raise ValueError("generation provenance differs from frozen job/current adapter")
            adapter_hashes[str(job["checkpoint_step"])] = adapter_sha
            inventory = evaluation / "technical" / f"{name}.csv"
            with inventory.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            technical = {int(row["sample_index"]): row for row in records}
            subset = {t["sample_index"]: t for t in tasks if t["run_name"] == name}
            if (len(records) != len(technical) or len(subset) != job["sample_count"]
                    or set(subset) != set(technical) or set(subset) != {int(k) for k in config}):
                raise ValueError("technical inventory/config/task indices differ")
            if {p.name for p in output.glob("*.mp4")} != {f"{i:06d}.mp4" for i in subset}:
                raise ValueError("missing or extra generated videos")
            config_by_index = {int(k): value for k, value in config.items()}
            for index, task in subset.items():
                row, item = technical[index], config_by_index[index]
                video = output / f"{index:06d}.mp4"
                if (video.resolve() != Path(task["generated_video"]).resolve() or video in checked_paths
                        or item["element_dtype_array"] != ["text", "video"]
                        or item["interleave_array"] != [task["instruction"], task["source_path"]]
                        or item["istarget_in_interleave"] != [0, 0]):
                    raise ValueError("video/task/input mapping mismatch")
                if (row["file"] != video.name or row["full_decode_ok"] != "1" or float(row["fps"]) != 12
                        or int(row["frames"]) <= 0 or int(row["bytes"]) != video.stat().st_size
                        or row["sha256"] != digest(video)):
                    raise ValueError("current video differs from its full-decode inventory")
                checked_paths.add(video)
                checked += 1
            inventories[name] = digest(inventory)
        if checked != count:
            raise ValueError("stage audit coverage incomplete")
        stages[stage] = {"expected": count, "verified_videos": checked,
                         "jobs_sha256": digest(jobs_path), "tasks_sha256": digest(tasks_path),
                         "technical_inventory_sha256": inventories, "adapter_sha256": adapter_hashes}
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "output_integrity_status": "passed",
            "verified_videos": len(checked_paths), "stages": stages,
            "base_weight_sha256": digest(weight), "shared_inference_code_sha256": digest(local),
            "judge_protocol_sha256": contract["canonical_protocol_sha256"],
            "scope": "all 735 current videos match exact frozen tasks/jobs/adapters and full-decode inventory SHA256; prior full-decode evidence applies to unchanged bytes, not a new decode pass",
            "scoring_completeness": "separate; passing output integrity does not fill a missing score"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.evaluation_root)
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["output_integrity_status"],
                      "verified_videos": report["verified_videos"]}))


if __name__ == "__main__":
    main()
