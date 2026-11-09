from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


manifest_tool = load_script("build_survey_manifest.py")
bundle_tool = load_script("build_review_bundle.py")
summary_tool = load_script("summarize_survey.py")
sqlite_tool = load_script("import_survey_sqlite.py")
openve_tool = load_script("extend_survey_with_openve.py")
reco_tool = load_script("extend_survey_with_reco.py")
filter_common = load_script("filter_common.py")
stage1_metadata = load_script("run_stage1_metadata.py")
score_merge = load_script("merge_stage1_scores.py")
specialist_queues = load_script("build_specialist_queues.py")
audit_sample = load_script("build_stage1_audit_sample.py")
phase_b = load_script("build_filter_phase_b.py")


class ManifestTests(unittest.TestCase):
    def test_egoedit_directions(self) -> None:
        self.assertEqual(
            manifest_tool.ego_direction("scene_no_change", "scene_edit"),
            "original_to_edited",
        )
        self.assertEqual(
            manifest_tool.ego_direction("scene_edit", "scene_no_change"),
            "edited_to_original",
        )
        self.assertEqual(
            manifest_tool.ego_direction("scene_a", "scene_b"),
            "edited_to_edited",
        )

    def test_fixed_dataset_provenance_controls_quality_side(self) -> None:
        goku_add = manifest_tool.dataset_provenance(
            "bigfacing/GOKU-2M",
            "f289f4b245648dc57db04afaaeb68659440c0fa6",
            "goku_add",
        )
        self.assertEqual(goku_add["source_provenance"], "edited")
        self.assertEqual(goku_add["target_provenance"], "original")
        self.assertEqual(goku_add["quality_checked_roles"], "source")
        self.assertEqual(goku_add["direction"], "edited_to_original")

        unknown = manifest_tool.dataset_provenance(
            "bigfacing/GOKU-2M", "new-revision", "goku_add"
        )
        self.assertEqual(unknown["source_provenance"], "unknown")
        self.assertEqual(unknown["target_provenance"], "unknown")
        self.assertEqual(unknown["quality_checked_roles"], "both")
        self.assertEqual(unknown["source_is_original"], "")

    def test_validate_requires_exact_balanced_400(self) -> None:
        branches = {
            name: {"quota": 100}
            for name in ("insave_remove", "goku_remove", "goku_add", "egoedit_replace")
        }
        rows = []
        for branch in branches:
            for index in range(100):
                rows.append({"branch": branch, "sample_id": f"{branch}-{index}"})
        manifest_tool.validate(rows, {"branches": branches})
        with self.assertRaises(ValueError):
            manifest_tool.validate(rows[:-1], {"branches": branches})

    def test_proxy_name_deduplicates_by_video_id(self) -> None:
        self.assertEqual(bundle_tool.proxy_name("video:1"), bundle_tool.proxy_name("video:1"))
        self.assertNotEqual(bundle_tool.proxy_name("video:1"), bundle_tool.proxy_name("video:2"))

    def test_duration_pattern(self) -> None:
        match = bundle_tool.DURATION_RE.search("Duration: 00:00:03.42, start: 0.0")
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), ("00", "00", "03.42"))

    def test_openve_extension_applies_duration_and_balances_available_tasks(self) -> None:
        rows = openve_tool.select_openve_rows(min_duration=3.0)
        self.assertEqual(len(rows), 11)
        self.assertEqual(
            Counter(row["task"] for row in rows),
            Counter({"add": 3, "remove": 3, "replace": 3, "change": 2}),
        )
        self.assertTrue(
            all(
                min(
                    float(row["source_duration_seconds"]),
                    float(row["target_duration_seconds"]),
                )
                >= 3.0
                for row in rows
            )
        )

    def test_reviewer_accepts_cross_version_import_by_stable_ids(self) -> None:
        template = (ROOT / "reviewer_template.html").read_text(encoding="utf-8")
        self.assertIn("mergeImportedState", template)
        self.assertIn("itemById.has(sampleId)", template)
        self.assertNotIn("survey version mismatch", template)

    def test_reco_scope_and_remove_exclusion(self) -> None:
        self.assertEqual(reco_tool.classify_scope("add", "Add a woman."), "person")
        self.assertEqual(reco_tool.classify_scope("add", "Add sunglasses to the man."), "wearable")
        self.assertEqual(reco_tool.classify_scope("replace", "Replace the blue shirt."), "wearable")
        self.assertIsNone(reco_tool.classify_scope("remove", "Remove the man's shoes."))
        self.assertEqual(reco_tool.classify_scope("remove", "Remove the man's hat."), "wearable")

    def test_reco_provenance_is_derived_from_fixed_revision(self) -> None:
        known = reco_tool.fixed_provenance(
            reco_tool.RECO_REPO, reco_tool.RECO_REVISION, "reco_replace"
        )
        self.assertEqual(known["source_provenance"], "original")
        self.assertEqual(known["target_provenance"], "edited")
        self.assertEqual(known["provenance_basis"], "official_dataset_rule")
        self.assertEqual(known["quality_checked_roles"], "target")

        unknown = reco_tool.fixed_provenance(
            reco_tool.RECO_REPO, "future-revision", "reco_replace"
        )
        self.assertEqual(unknown["source_provenance"], "unknown")
        self.assertEqual(unknown["target_provenance"], "unknown")
        self.assertEqual(unknown["provenance_basis"], "unknown")
        self.assertEqual(unknown["quality_checked_roles"], "both")

    def test_target_scope_uses_direct_object_not_description(self) -> None:
        cases = {
            "Remove the man in the black t-shirt.": "person",
            "Remove the toilet bowl near the man.": "object",
            "Add a car in the background.": "object",
            "Add sunglasses to the man.": "accessory",
            "Remove the man's shoes.": "clothes",
            "Remove a men's collared shirt.": "clothes",
            "Replace the man's face with another face.": "appearance",
            "Add a woman wearing a shirt.": "person",
            "Add the woman is wearing a white button-down shirt.": "clothes",
            "Remove the man in the center is wearing a grey sweatshirt.": "clothes",
        }
        for instruction, expected in cases.items():
            with self.subTest(instruction=instruction):
                self.assertEqual(filter_common.classify_target_scope(instruction), expected)

    def test_garment_exclusion_is_dataset_specific(self) -> None:
        instruction = "Remove the man's shoes."
        self.assertIn(
            "excluded_garment_add_remove",
            filter_common.metadata_rejection_reasons("goku", "remove", instruction),
        )
        self.assertNotIn(
            "excluded_garment_add_remove",
            filter_common.metadata_rejection_reasons("reco", "remove", instruction),
        )
        self.assertNotIn(
            "excluded_garment_add_remove",
            filter_common.metadata_rejection_reasons(
                "goku", "add", "Add a woman wearing a shirt."
            ),
        )
        self.assertIn(
            "excluded_garment_add_remove",
            filter_common.metadata_rejection_reasons(
                "goku", "add", "Add the woman is wearing a white button-down shirt."
            ),
        )

    def test_stage1_metadata_short_circuits_rejected_rows(self) -> None:
        row = {
            "dataset": "goku",
            "branch": "goku_remove",
            "operation": "remove",
            "instruction": "Remove the man's shoes.",
            "target_scope": "clothes",
            "source_locator": "source.mp4",
            "target_locator": "target.mp4",
            "source_local": "",
            "target_local": "",
            "source_provenance": "original",
            "target_provenance": "edited",
            "quality_checked_roles": "target",
        }
        result = stage1_metadata.evaluate(row)
        self.assertEqual(result["metadata_decision"], "rejected")
        self.assertEqual(result["next_stage"], "stop")
        self.assertIn("excluded_garment_add_remove", result["metadata_reasons"])

    def test_score_merge_prefers_full_scores(self) -> None:
        technical = {"technical_status": "pass", "musiq_p10": "", "dino_max_feature_jump": ""}
        full = {"technical_status": "pass", "musiq_p10": "55", "dino_max_feature_jump": "0.2"}
        self.assertGreater(score_merge.completeness(full), score_merge.completeness(technical))

    def test_robot_routing_is_reco_only(self) -> None:
        instruction = "Replace the person with a detailed humanoid robot."
        self.assertTrue(
            specialist_queues.is_robot_candidate(
                {"dataset": "reco", "pilot_group": "branch_random", "instruction": instruction}
            )
        )
        self.assertFalse(
            specialist_queues.is_robot_candidate(
                {"dataset": "egoedit", "pilot_group": "branch_random", "instruction": instruction}
            )
        )

    def test_audit_sampling_round_robins_strata(self) -> None:
        rows = [
            {"branch": branch, "pilot_group": "base", "target_scope": "object", "id": str(index)}
            for branch in ("a", "b")
            for index in range(4)
        ]
        chosen = audit_sample.stratified_sample(rows, 4, 7)
        self.assertEqual(Counter(row["branch"] for row in chosen), Counter({"a": 2, "b": 2}))

    def test_phase_b_quota_is_exact_and_preserves_branch_floor(self) -> None:
        rows = [
            {"branch": branch, "stage1_1_decision": "uncertain" if branch == "insave_remove" else "accepted"}
            for branch in phase_b.BRANCHES
            for _ in range(10)
        ]
        quotas = phase_b.adaptive_quotas(rows)
        self.assertEqual(sum(quotas.values()), 500)
        self.assertTrue(all(quota >= 50 for quota in quotas.values()))
        self.assertGreater(quotas["insave_remove"], quotas["goku_add"])


class SummaryTests(unittest.TestCase):
    def test_review_validation_and_trigger_rules(self) -> None:
        manifest = [{"sample_id": "a", "branch": "x"}]
        review = [
            {
                "sample_id": "a",
                "target_status": "pass",
                "source_status": "uncertain",
                "pair_status": "hard_fail",
            }
        ]
        rows = summary_tool.validate(manifest, review, False)
        self.assertEqual(rows[0]["pair_status"], "hard_fail")
        self.assertIn("追加 50", summary_tool.trigger_decision(2, summary_tool.Counter({"x": 2})))
        self.assertIn("触发", summary_tool.trigger_decision(3, summary_tool.Counter({"x": 3})))

    def test_csv_reader_handles_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["sample_id", "notes"])
                writer.writeheader()
                writer.writerow({"sample_id": "a", "notes": "局部涂抹, 闪烁"})
            self.assertEqual(summary_tool.read_csv(path)[0]["notes"], "局部涂抹, 闪烁")


class SQLiteTests(unittest.TestCase):
    def test_manifest_and_review_are_idempotent(self) -> None:
        manifest = [
            {
                "survey_version": "v1", "sample_id": "sample-a", "sample_rank": "1",
                "dataset": "demo", "repo_id": "org/demo", "revision": "abc",
                "pair_id": "pair-a", "source_video_id": "video-source",
                "target_video_id": "video-target", "source_locator": "source.mp4",
                "target_locator": "target.mp4", "source_local": "source.mp4",
                "target_local": "target.mp4", "instruction": "Replace the cup.",
                "operation": "replace", "direction": "edited_to_edited",
            }
        ]
        review = [
            {
                "sample_id": "sample-a", "source_status": "uncertain",
                "source_defects": "smear_ghost", "source_flags": "suspected_edited|edit_residual",
                "source_notes": "", "source_seconds": "3", "target_status": "pass",
                "target_defects": "", "target_notes": "", "target_seconds": "2",
                "pair_status": "hard_fail", "pair_errors": "unrelated_pair",
                "pair_notes": "wrong clip", "pair_seconds": "4",
            }
        ]
        connection = sqlite3.connect(":memory:")
        connection.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
        for _ in range(2):
            sqlite_tool.upsert_manifest(connection, manifest)
            self.assertEqual(sqlite_tool.upsert_reviews(connection, manifest, review), 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0], 1)
        flags = connection.execute("SELECT source_flags_json FROM pair_audits").fetchone()[0]
        self.assertIn("edit_residual", flags)
        connection.close()


if __name__ == "__main__":
    unittest.main()
