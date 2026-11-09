import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summarize_funnel_evaluation import (
    METRICS, canonical_hash, check_score, paired_omni, read_results, split_summary,
)


def score(accuracy=2):
    return {"accuracy": accuracy, "preservation": 3, "realism": 3, "consistency": 3,
            "overall_score": round(0.8 * accuracy, 4), "reason": "fixture"}


def pairs():
    base, winner = [], []
    for group in range(24):
        for index in range(10):
            task = {"sample_id": f"g{group}_{index}", "instruction": "fixture", "generation_seed": 42,
                    "_source_sha256": f"source-{group}-{index}"}
            base.append({**task, **score(2), "category": f"g{group}", "checkpoint_step": "base"})
            winner.append({**task, **score(3), "branch": f"g{group}", "checkpoint_step": 677})
    return base, winner


class SummaryTests(unittest.TestCase):
    def test_formula_and_integer_dimensions(self):
        check_score(score())
        for invalid in ({**score(), "accuracy": True}, {**score(), "accuracy": 2.0},
                        {**score(), "accuracy": 6}, {**score(), "overall_score": float("nan")},
                        {**score(), "overall_score": 4}):
            with self.assertRaises(ValueError):
                check_score(invalid)

    def test_partial_is_not_zero_and_macro_is_branch_weighted(self):
        tasks = [{"sample_id": "a1", "branch": "a"}, {"sample_id": "a2", "branch": "a"},
                 {"sample_id": "b1", "branch": "b"}, {"sample_id": "b2", "branch": "b"}]
        rows = [{**tasks[0], **score(0)}, {**tasks[1], **score(0)}, {**tasks[2], **score(5)}]
        result = split_summary(tasks, rows)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_sample_ids"], ["b2"])
        self.assertAlmostEqual(result["micro_mean_scores_successes_only"]["overall_score"], 4 / 3)
        self.assertEqual(result["group_macro_mean_scores_successes_only"]["overall_score"], 2)

    def test_duplicate_split_ids_rejected(self):
        task = {"sample_id": "a", "branch": "a"}
        with self.assertRaises(ValueError):
            split_summary([task, task], [])
        with self.assertRaises(ValueError):
            split_summary([task], [{**task, **score()}] * 2)

    def test_omni_pairing_sign_seed_and_confidence(self):
        base, winner = pairs()
        first = paired_omni(base, winner, resamples=50)
        self.assertEqual(first, paired_omni(base, winner, resamples=50))
        self.assertAlmostEqual(first["mean_differences"]["overall_score"], 0.8)
        self.assertEqual(first["overall_sample_wins_ties_losses"], {"wins": 240, "ties": 0, "losses": 0})
        for bound in first["paired_stratified_bootstrap_95_ci"]["overall_score"]:
            self.assertAlmostEqual(bound, 0.8)
        self.assertEqual(set(first["mean_differences"]), set(METRICS))

    def test_omni_refuses_partial_duplicate_or_changed_inputs(self):
        base, winner = pairs()
        with self.assertRaises(ValueError):
            paired_omni(base, winner[:-1], resamples=5)
        with self.assertRaises(ValueError):
            paired_omni(base, winner + winner[:1], resamples=5)
        for field, value in (("_source_sha256", "changed"), ("instruction", "changed"),
                             ("generation_seed", 43), ("checkpoint_step", 902), ("branch", "changed")):
            changed = copy.deepcopy(winner)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                paired_omni(base, changed, resamples=5)

    def test_receipts_require_frozen_protocol_and_actual_media_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"settings": {"model": "google/gemini-3.8-flash"}, "expected_response_provider": "Google"}
            (root / "protocol.json").write_text(json.dumps(protocol))
            task = {"stage": "test", "checkpoint_step": 677, "sample_id": "a",
                    "source_path": "source", "generated_video": "generated"}
            record = {"status": "success", "returned_model": "google/gemini-3.8-flash",
                      "returned_provider": "Google", "score": score(),
                      "identity": {"task": task, "source_sha256": "source-hash", "generated_sha256": "output-hash",
                                   "protocol_sha256": canonical_hash(protocol)}}
            target = root / "records/item/result.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(record))
            (target.parent / "attempt_001.json").write_text(json.dumps({"usage": {"cost": 0.002}}))
            (target.parent / "attempt_002.json").write_text(json.dumps({"usage": {}}))
            with patch("summarize_funnel_evaluation.media_sha", side_effect=lambda p: {"source": "source-hash", "generated": "output-hash"}[p]):
                rows, cost = read_results(root, [task], protocol)
                self.assertEqual(len(rows), 1)
                self.assertEqual(cost["known_cost_usd"], 0.002)
                self.assertEqual(cost["requests_without_reported_cost"], 1)
            with patch("summarize_funnel_evaluation.media_sha", return_value="changed"):
                with self.assertRaises(ValueError):
                    read_results(root, [task], protocol)
            record["identity"]["protocol_sha256"] = "wrong-protocol"
            target.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                read_results(root, [task], protocol)


if __name__ == "__main__":
    unittest.main()
