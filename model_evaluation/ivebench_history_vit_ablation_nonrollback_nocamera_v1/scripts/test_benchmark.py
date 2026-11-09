#!/usr/bin/env python3
"""Validate the frozen IVEBench non-rollback/no-camera benchmark."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark.json"
REWRITE_AUDIT = ROOT / "camera_dependency_rewrites.json"
EXPECTED_SOURCE_SHA256 = "1f9edd319806e546eef367d82facc14cdb2240b872ad41135e980cd2fe62998a"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    audit = json.loads(REWRITE_AUDIT.read_text(encoding="utf-8"))
    expected_header = {
        "schema_version": "ivebench_nonrollback_nocamera_history_vit_v1",
        "status": "frozen",
        "source_benchmark_sha256": EXPECTED_SOURCE_SHA256,
        "chain_count": 24,
        "total_turns": 65,
        "formal_checkpoints": 41,
        "axis_counts": {"appearance": 23, "action": 20, "outfit": 22},
        "removed_camera_turn_count": 21,
        "post_camera_rewrite_count": 33,
    }
    for field, expected in expected_header.items():
        assert benchmark.get(field) == expected, (field, benchmark.get(field), expected)
    cases = benchmark["cases"]
    assert [case["chain_id"] for case in cases] == [f"rpmte_{index:03d}" for index in range(1, 25)]
    assert all(case["chain_type"] != "rollback" for case in cases)

    axes = Counter()
    turn_count = checkpoint_count = 0
    source_round_pairs: set[tuple[str, int]] = set()
    for case in cases:
        source = Path(case["source"])
        assert source.is_absolute() and source.is_file(), source
        assert sha256(source) == case["source_sha256"] == case["source_metadata"]["local_sha256"]
        outputs = case["outputs"]
        conversation = case["conversation"]["turns"]
        mapping = case["round_mapping"]
        assert len(outputs) == len(conversation) == len(mapping)
        assert [item["round"] for item in outputs] == list(range(1, len(outputs) + 1))
        assert [item["source_round"] for item in outputs] == case["kept_source_rounds"]
        assert [item["source_round"] for item in mapping] == case["kept_source_rounds"]
        assert [item["derived_round"] for item in mapping] == list(range(1, len(outputs) + 1))
        assert len(case["removed_source_rounds"]) in (0, 1)
        if case["removed_source_rounds"]:
            assert len(outputs) + 1 in (3, 4, 5)
        for index, (output, turn) in enumerate(zip(outputs, conversation), 1):
            assert output["axis"] in {"appearance", "action", "outfit"}
            assert output["history_reference_round"] is None
            assert turn["role"] == "user" and turn["action"] == "edit_video"
            texts = [item["value"] for item in turn["content"] if item["type"] == "text"]
            videos = [item["value"] for item in turn["content"] if item["type"] == "video"]
            assert texts == [output["instruction"]]
            assert videos == ([str(source)] if index == 1 else [])
            assert turn["meta"]["round"] == output["round"]
            assert turn["meta"]["source_round"] == output["source_round"]
            assert turn["meta"]["axis"] == output["axis"]
            assert turn["meta"]["preserve_requirements"] == output["preserve_requirements"]
            assert turn["meta"]["target_state"] == output["target_state"]
            axes[output["axis"]] += 1
            source_round_pairs.add((case["chain_id"], output["source_round"]))
        turn_count += len(outputs)
        checkpoint_count += max(0, len(outputs) - 1)

    assert turn_count == 65 and checkpoint_count == 41
    assert axes == Counter({"appearance": 23, "outfit": 22, "action": 20})
    assert len(benchmark["removed_camera_turns"]) == 21
    assert all(item["removed_turn"]["axis"] == "camera" for item in benchmark["removed_camera_turns"])
    rewrites = benchmark["camera_dependency_rewrites"]
    assert len(rewrites) == 33
    assert audit["rewrites"] == rewrites
    for item in rewrites:
        key = (item["chain_id"], item["source_round"])
        assert key in source_round_pairs
        assert item["source_round"] > item["camera_source_round_removed"]
        assert item["changed_fields"]
        assert all(item["original"][field] != item["revised"][field] for field in item["changed_fields"])
        assert "preserve_requirements" in item["changed_fields"]

    assert benchmark["derivation"]["ground_truth_available"] is False
    print(json.dumps({
        "status": "passed",
        "benchmark_sha256": sha256(BENCHMARK),
        "chains": len(cases),
        "turns": turn_count,
        "formal_checkpoints": checkpoint_count,
        "axis_counts": dict(axes),
        "source_media_verified": len(cases),
        "removed_camera_turns": len(benchmark["removed_camera_turns"]),
        "post_camera_rewrites_verified": len(rewrites),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
