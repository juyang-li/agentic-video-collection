#!/usr/bin/env python3
"""Freeze resumable IVEBench generation jobs for the two step-677 ablations."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO_COLLECTION = ROOT.parents[1]
FINETUNE = VIDEO_COLLECTION / "lance_general_edit_finetune/3_lance_lora_finetuning"
RUNTIME = FINETUNE / "runtime/lance"
MODEL = RUNTIME / "downloads/Lance_3B_Video"
RUNNER = FINETUNE / "scripts/run_lora_video_edit.sh"
PLAN_ROOT = ROOT / "artifacts/coinve_ablation_v1"
MODELS = {
    "coinve4k_step677": FINETUNE / "runs/lance_coinve_4k_r64_seed42_e2_v1/checkpoints/0000677",
    "mixed2k2k_step677": FINETUNE / "runs/lance_mixed_2k2k_r64_seed42_e2_v1/checkpoints/0000677",
}
STAGES = {
    "short_subject": {"manifest": "short_subject_manifest.csv", "id": "official_id", "count": 91, "phase1": 46, "duration": 6.0},
    "long_subject": {"manifest": "manifest.csv", "id": "generation_id", "count": 43, "phase1": 21, "duration": 10.1},
    "attribute_all": {"manifest": "manifest.csv", "id": "generation_id", "count": 94, "phase1": 47, "duration": 10.1},
}
GENERATION_ORDER = ("attribute_all", "long_subject", "short_subject")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stratified_half(rows: list[dict[str, str]], target: int, stage: str) -> set[str]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["subcategory"]].append(row)
    raw = {key: len(values) * target / len(rows) for key, values in groups.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(groups, key=lambda key: (-(raw[key] - counts[key]), stable_key(42, stage, key)))
    for key in order[:remaining]:
        counts[key] += 1
    selected = set()
    for key, values in groups.items():
        ordered = sorted(values, key=lambda row: stable_key(42, "ivebench-ablation-half", stage, row["sample_id"]))
        selected.update(row["sample_id"] for row in ordered[:counts[key]])
    if len(selected) != target:
        raise RuntimeError(f"phase-1 size mismatch for {stage}")
    return selected


def validate_adapter(path: Path) -> str:
    marker = json.loads((path / "checkpoint_complete.json").read_text())
    if marker.get("status") != "complete" or marker.get("completed_steps") != 677:
        raise ValueError(f"incomplete step-677 checkpoint: {path}")
    return sha256_file(path / "adapter.safetensors")


def main() -> None:
    if not MODEL.joinpath("model.safetensors").is_file():
        raise FileNotFoundError(MODEL)
    adapter_hashes = {kind: validate_adapter(path) for kind, path in MODELS.items()}
    groups = []
    phase_manifest = {}
    stage_rows = {}
    for stage in GENERATION_ORDER:
        spec = STAGES[stage]
        artifact = ROOT / "artifacts" / stage
        manifest_path = artifact / spec["manifest"]
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != spec["count"]:
            raise ValueError(f"{stage}: expected {spec['count']} rows")
        stage_rows[stage] = rows
        first_ids = stratified_half(rows, spec["phase1"], stage)
        for phase in (1, 2):
            phase_rows = [row for row in rows if (row["sample_id"] in first_ids) == (phase == 1)]
            phase_manifest[f"{stage}_phase{phase}"] = {
                "samples": len(phase_rows),
                "sample_ids": [row["sample_id"] for row in phase_rows],
                "subcategories": dict(Counter(row["subcategory"] for row in phase_rows)),
            }

    # Keep phase membership for judge ordering, while generating each stage in a
    # single model load. Splitting inference would reload both models three extra
    # times without changing any generated sample.
    for stage in GENERATION_ORDER:
        spec = STAGES[stage]
        artifact = ROOT / "artifacts" / stage
        jobs = []
        for model_index, (kind, adapter) in enumerate(MODELS.items()):
            for shard in range(4):
                shard_rows = stage_rows[stage][shard::4]
                run_name = f"{stage}_{kind}_shard{shard:02d}"
                config_path = artifact / "configs" / f"{run_name}.json"
                config = {
                    f"{int(row[spec['id']]):06d}": {
                        "element_dtype_array": ["text", "video"],
                        "interleave_array": [row["edit_prompt"], row["source_path"]],
                        "istarget_in_interleave": [0, 0],
                    }
                    for row in shard_rows
                }
                write_json(config_path, config)
                jobs.append({
                    "stage": stage, "phase": "all", "model_kind": kind,
                    "gpu_slot": model_index * 4 + shard,
                    "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
                    "run_name": run_name, "sample_count": len(shard_rows),
                    "adapter_path": str(adapter.resolve()), "adapter_sha256": adapter_hashes[kind],
                    "max_duration": spec["duration"],
                    "command": [
                        "bash", str(RUNNER.resolve()), str(artifact.resolve()), str(RUNTIME.resolve()),
                        str(MODEL.resolve()), str(adapter.resolve()), str(config_path.resolve()), run_name,
                    ],
                })
        groups.append({"name": f"full_{stage}", "phase": "all", "stage": stage, "jobs": jobs})
    write_json(PLAN_ROOT / "phase_manifest.json", {
        "policy": "deterministic subcategory-stratified first half; 114 samples per model before complements",
        "seed": 42, "stages": phase_manifest,
    })
    plan = {
        "schema_version": "ivebench_coinve_ablation_generation_v1",
        "benchmark": "IVEBench", "generation_seed": 42,
        "scheduling": "three full-manifest stage groups, longest first; phase split is judge-order metadata only",
        "base_model_path": str(MODEL.resolve()),
        "base_model_sha256": sha256_file(MODEL / "model.safetensors"),
        "models": {kind: {"checkpoint": str(path.resolve()), "adapter_sha256": adapter_hashes[kind]} for kind, path in MODELS.items()},
        "inference": {"resolution": "video_480p", "save_fps": 12, "steps": 30, "timestep_shift": 3.5, "cfg_text_scale": 4.0, "seed": 42, "dtype": "bf16"},
        "groups": groups,
    }
    write_json(PLAN_ROOT / "generation_plan.json", plan)
    print(json.dumps({"groups": len(groups), "jobs": sum(len(g["jobs"]) for g in groups), "phase1_samples_per_model": 114, "full_samples_per_model": 228}, indent=2))


if __name__ == "__main__":
    main()
