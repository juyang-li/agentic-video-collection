#!/usr/bin/env python3
"""Regression tests for the frozen IVEBench evaluation contract."""

from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aggregate = load("aggregate_short_subject.py")
        cls.review = load("build_short_subject_review.py")
        cls.judge = load("judge_gemini.py")
        cls.ablation_prepare = load("prepare_coinve_ablation_eval.py")
        cls.ablation_report = load("render_coinve_ablation_report.py")
        cls.metric_bundle = load("run_metric_bundle.py")
        verifier_path = HERE.parents[1] / "lance_tools" / "verify_outputs.py"
        spec = importlib.util.spec_from_file_location("verify_outputs", verifier_path)
        cls.verifier = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.verifier)

    def test_official_metric_weights(self) -> None:
        row = {
            "subject_consistency": 0.2,
            "background_consistency": 0.3,
            "temporal_flickering": 0.4,
            "motion_smoothness": 0.5,
            "vtss": 0.025,
            "overall_semantic_consistency": 0.6,
            "phrase_semantic_consistency": 0.7,
            "instruction_satisfaction": 3,
            "semantic_fidelity": 0.8,
            "motion_fidelity": 0.9,
            "content_fidelity": 4,
        }
        quality = (0.2 + 0.3 + 0.4 + 0.5 + 4 * 0.5) / 8
        compliance = (0.6 + 0.7 + 3 * 0.5) / 5
        fidelity = (0.8 + 0.9 + 3 * 0.75) / 5
        expected = (quality + compliance + fidelity) / 3
        report = self.aggregate.aggregate_values([row])
        self.assertAlmostEqual(report["dimensions"]["quality"], quality)
        self.assertAlmostEqual(report["dimensions"]["compliance"], compliance)
        self.assertAlmostEqual(report["dimensions"]["fidelity"], fidelity)
        self.assertAlmostEqual(report["total_score"], expected)
        prefixed = {"base_" + key: value for key, value in row.items()}
        self.assertAlmostEqual(self.review.sample_total(prefixed, "base"), expected)

    def test_dynamic_local_metric_counts(self) -> None:
        aggregate = self.aggregate
        original_local = aggregate.LOCAL
        original_expected = aggregate.EXPECTED_SAMPLES
        try:
            for count in (43, 94):
                with tempfile.TemporaryDirectory() as directory:
                    aggregate.LOCAL = Path(directory)
                    aggregate.EXPECTED_SAMPLES = count
                    for metric, bundle in aggregate.LOCAL_METRICS.items():
                        path = aggregate.LOCAL / "base" / bundle / f"{metric}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({
                            "status": "complete",
                            "records": count,
                            "error_records": 0,
                            "details": [
                                {"video_name": f"sample_{index:03d}.mp4", "video_results": 0.5}
                                for index in range(count)
                            ],
                        }))
                    scores = aggregate.local_scores("base")
                    self.assertEqual(len(scores), count)
                    self.assertTrue(all(len(values) == 9 for values in scores.values()))
        finally:
            aggregate.LOCAL = original_local
            aggregate.EXPECTED_SAMPLES = original_expected

    def test_three_stage_review_metadata(self) -> None:
        self.assertEqual(set(self.review.STAGE_REVIEW_META), set(self.review.STAGES))
        self.assertEqual(self.review.STAGE_REVIEW_META["short_subject"][0], "IVEBench Short")
        self.assertEqual(self.review.STAGE_REVIEW_META["long_subject"][0], "IVEBench Long")
        self.assertEqual(self.review.STAGE_REVIEW_META["attribute_all"][0], "IVEBench Short+Long")

    def test_judge_route_is_frozen(self) -> None:
        settings = self.judge.SETTINGS
        self.assertEqual(settings["model"], "google/gemini-3.8-flash")
        self.assertEqual(settings["temperature"], 0.0)
        self.assertEqual(settings["reasoning"], {"effort": "low"})
        self.assertEqual(settings["provider"]["only"], ["google-ai-studio"])
        self.assertFalse(settings["provider"]["allow_fallbacks"])
        self.assertTrue(settings["provider"]["require_parameters"])

    def test_pairwise_api_payload_is_blinded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = []
            for name in ("source.mp4", "candidate_a.mp4", "candidate_b.mp4"):
                path = Path(directory) / name
                path.write_bytes(b"anonymous-video-bytes")
                media.append(str(path))
            task = {
                "kind": "pairwise",
                "sample_id": "secret-sample-id",
                "instruction": "Make the object blue.",
                "source_video": media[0],
                "candidate_a_video": media[1],
                "candidate_b_video": media[2],
                "candidate_a_model": "coinve4k_step677",
                "candidate_b_model": "mixed2k2k_step677",
                "model_pair": "coinve4k_step677__vs__mixed2k2k_step677",
            }
            serialized = json.dumps(self.judge.payload(task), sort_keys=True)
            self.assertNotIn("secret-sample-id", serialized)
            self.assertNotIn("coinve4k_step677", serialized)
            self.assertNotIn("mixed2k2k_step677", serialized)
            self.assertIn("Candidate A:", serialized)
            self.assertIn("Candidate B:", serialized)
            self.assertEqual(serialized.count("data:video/mp4;base64,"), 3)

    def test_transient_google_region_error_is_retryable(self) -> None:
        transient = {"error": {"message": "User location is not supported for the API use."}}
        self.assertTrue(self.judge.retryable_provider_error(400, transient))
        self.assertFalse(self.judge.retryable_provider_error(400, {"error": {"message": "Bad request"}}))
        self.assertFalse(self.judge.retryable_provider_error(403, transient))

    def test_bidirectional_consensus_normalizes_candidate_order(self) -> None:
        rows = [
            {
                "sample_id": "sample-1", "stage": "short_subject", "model_pair": "model_x__vs__model_y",
                "candidate_a_model": "model_x", "candidate_b_model": "model_y",
                "normalized_instruction_winner": "model_x",
                "normalized_fidelity_winner": "tie", "normalized_overall_winner": "model_x",
            },
            {
                "sample_id": "sample-1", "stage": "short_subject", "model_pair": "model_x__vs__model_y",
                "candidate_a_model": "model_y", "candidate_b_model": "model_x",
                "normalized_instruction_winner": "model_x",
                "normalized_fidelity_winner": "tie", "normalized_overall_winner": "model_y",
            },
        ]
        consensus = self.judge.bidirectional_consensus(rows)
        self.assertEqual(consensus["groups"], 1)
        self.assertEqual(consensus["counts"]["instruction"]["model_x"], 1)
        self.assertEqual(consensus["counts"]["fidelity"]["tie"], 1)
        self.assertEqual(consensus["counts"]["overall"]["order_inconsistent"], 1)
        self.assertEqual(consensus["by_pair"]["model_x__vs__model_y"]["groups"], 1)
        self.assertEqual(
            consensus["by_pair_stage"]["model_x__vs__model_y"]["short_subject"]["counts"]["instruction"]["model_x"],
            1,
        )

    def test_ablation_generation_uses_three_full_stage_groups(self) -> None:
        prepare = self.ablation_prepare
        names = ("ROOT", "FINETUNE", "RUNTIME", "MODEL", "RUNNER", "PLAN_ROOT", "MODELS")
        original = {name: getattr(prepare, name) for name in names}
        try:
            with tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                prepare.ROOT = temporary / "eval"
                prepare.FINETUNE = temporary / "finetune"
                prepare.RUNTIME = temporary / "runtime"
                prepare.MODEL = temporary / "model"
                prepare.RUNNER = temporary / "run.sh"
                prepare.PLAN_ROOT = prepare.ROOT / "artifacts/coinve_ablation_v1"
                prepare.MODELS = {
                    "coinve4k_step677": temporary / "coinve",
                    "mixed2k2k_step677": temporary / "mixed",
                }
                prepare.MODEL.mkdir(parents=True)
                (prepare.MODEL / "model.safetensors").write_bytes(b"base")
                prepare.RUNNER.write_text("#!/bin/sh\n")
                for checkpoint in prepare.MODELS.values():
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
                    (checkpoint / "checkpoint_complete.json").write_text(json.dumps({
                        "status": "complete", "completed_steps": 677,
                    }))
                for stage, spec in prepare.STAGES.items():
                    artifact = prepare.ROOT / "artifacts" / stage
                    artifact.mkdir(parents=True)
                    path = artifact / spec["manifest"]
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        fieldnames = [
                            "sample_id", "subcategory", "official_id", "generation_id",
                            "edit_prompt", "source_path",
                        ]
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        for index in range(spec["count"]):
                            writer.writerow({
                                "sample_id": f"{stage}-{index:03d}",
                                "subcategory": f"subcategory-{index % 5}",
                                "official_id": index,
                                "generation_id": index,
                                "edit_prompt": "edit",
                                "source_path": f"/source/{index}.mp4",
                            })
                prepare.main()
                plan = json.loads((prepare.PLAN_ROOT / "generation_plan.json").read_text())
                jobs = [job for group in plan["groups"] for job in group["jobs"]]
                self.assertEqual([group["name"] for group in plan["groups"]], [
                    "full_attribute_all", "full_long_subject", "full_short_subject",
                ])
                self.assertEqual(len(jobs), 24)
                self.assertTrue(all(len(group["jobs"]) == 8 for group in plan["groups"]))
                self.assertTrue(all({job["gpu_slot"] for job in group["jobs"]} == set(range(8)) for group in plan["groups"]))
                for model in prepare.MODELS:
                    model_jobs = [job for job in jobs if job["model_kind"] == model]
                    self.assertEqual(len(model_jobs), 12)
                    self.assertEqual(sum(job["sample_count"] for job in model_jobs), 228)
        finally:
            for name, value in original.items():
                setattr(prepare, name, value)

    def test_ablation_report_renders_pair_and_stage_consensus(self) -> None:
        rows = []
        for stage in ("short_subject", "long_subject", "attribute_all"):
            for candidate_a, candidate_b in (
                ("single4k_step677", "coinve4k_step677"),
                ("coinve4k_step677", "single4k_step677"),
            ):
                rows.append({
                    "sample_id": f"sample-{stage}", "stage": stage,
                    "model_pair": "single4k_step677__vs__coinve4k_step677",
                    "candidate_a_model": candidate_a, "candidate_b_model": candidate_b,
                    "normalized_instruction_winner": "coinve4k_step677",
                    "normalized_fidelity_winner": "single4k_step677",
                    "normalized_overall_winner": "coinve4k_step677",
                })
        consensus = self.judge.bidirectional_consensus(rows)
        markdown = self.ablation_report.build_markdown({
            "tasks_succeeded": 6, "tasks_requested": 6, "reported_cost_usd": 0.01,
            "order_consistency": {"instruction": 1.0, "fidelity": 1.0, "overall": 1.0},
            "bidirectional_consensus": consensus,
        })
        self.assertIn("Single-4K vs CoinVE-4K", markdown)
        self.assertIn("| Attribute | 0 | 1 | 0 | 0 | 0.0% / 100.0% |", markdown)

    def test_local_metric_reuse_requires_a_complete_full_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metric.json"
            complete = {
                "metric": "example", "status": "complete", "records": 3,
                "valid_records": 3, "error_records": 0,
                "details": [{"video_results": 0.5} for _ in range(3)],
            }
            path.write_text(json.dumps(complete))
            self.assertEqual(
                self.metric_bundle.reusable_metric_result(path, "example", 3), complete)
            for field, bad_value in (("status", "running"), ("records", 2), ("error_records", 1)):
                invalid = dict(complete); invalid[field] = bad_value
                path.write_text(json.dumps(invalid))
                self.assertIsNone(
                    self.metric_bundle.reusable_metric_result(path, "example", 3))

    def test_lance_sampled_frame_count_matches_short_and_long_protocols(self) -> None:
        self.assertEqual(self.verifier.lance_sampled_frame_count(105, 25.0, 10.1), 49)
        self.assertEqual(self.verifier.lance_sampled_frame_count(73, 23.976024, 10.1), 33)
        self.assertEqual(self.verifier.lance_sampled_frame_count(121, 12.0, 10.1), 121)
        self.assertEqual(self.verifier.lance_sampled_frame_count(123, 25.0, 6.0), 57)


if __name__ == "__main__":
    unittest.main()
