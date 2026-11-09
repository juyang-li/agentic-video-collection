from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE1_SCRIPTS = PROJECT / "1_specialist_filtering/scripts"
STAGE2_SCRIPTS = ROOT / "scripts"
for directory in (STAGE1_SCRIPTS, STAGE2_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


continuation = load(STAGE1_SCRIPTS / "build_stage1_continuation.py")
qwen = load(STAGE2_SCRIPTS / "run_qwen_judge.py")
import run_qwen_rect_judge as rect_qwen
import run_openrouter_qwen3_judge as qwen3
import evaluate_qwen3_calibration as evaluate_qwen3
import apply_qwen3_gates
import compose_best_routed_filter_v12 as routed_v12
apply_qwen = load(STAGE2_SCRIPTS / "apply_qwen_thresholds.py")
finalize = load(STAGE2_SCRIPTS / "finalize_stage2.py")


class ContinuationTests(unittest.TestCase):
    def test_structural_exclusions_are_specific(self) -> None:
        insave = {"branch": "insave_remove", "instruction": "Remove a man."}
        handheld = {
            "branch": "goku_remove",
            "operation": "remove",
            "target_scope": "object",
            "instruction": "Remove the cup held by the woman.",
        }
        person = {
            "branch": "goku_remove",
            "operation": "remove",
            "target_scope": "person",
            "instruction": "Remove the woman holding a cup.",
        }
        malformed_garment = {
            "dataset": "goku",
            "branch": "goku_add",
            "operation": "add",
            "instruction": "Add the woman is wearing a white button-down shirt.",
        }
        self.assertEqual(
            continuation.structural_reasons(insave), ["excluded_insave_person_remove_branch"]
        )
        self.assertEqual(
            continuation.structural_reasons(handheld), ["excluded_handheld_object_remove"]
        )
        self.assertEqual(continuation.structural_reasons(person), [])
        self.assertEqual(
            continuation.structural_reasons(malformed_garment),
            ["excluded_garment_add_remove"],
        )


class QwenParsingTests(unittest.TestCase):
    def test_extracts_fenced_json_and_normalizes(self) -> None:
        raw = """```json
        {"decision":"accepted","instruction_following":"pass","edit_completeness":"pass",
        "content_preservation":"pass","temporal_consistency":"pass","visible_artifacts":"pass",
        "reason_codes":["Looks Good!"],"confidence":1.4,"short_evidence":"ok"}
        ```"""
        value = qwen.normalize_judgment(qwen.extract_json_object(raw))
        self.assertEqual(value["decision"], "accepted")
        self.assertEqual(value["reason_codes"], ["looks_good"])
        self.assertEqual(value["confidence"], 1.0)

    def test_inconsistent_accept_is_downgraded(self) -> None:
        value = qwen.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "fail",
                "edit_completeness": "pass",
                "content_preservation": "pass",
                "temporal_consistency": "pass",
                "visible_artifacts": "pass",
                "confidence": 0.9,
            }
        )
        self.assertEqual(value["decision"], "uncertain")


class RectQwenTests(unittest.TestCase):
    def test_change_region_encloses_and_expands_local_edit(self) -> None:
        source = np.full((240, 320, 3), 110, dtype=np.uint8)
        target = source.copy()
        target[80:140, 120:180] = (230, 20, 30)
        proposal = rect_qwen.estimate_change_region(source, target, index=0)
        x1, y1, x2, y2 = proposal.expanded_box
        self.assertLessEqual(x1, 120 / 320)
        self.assertLessEqual(y1, 80 / 240)
        self.assertGreaterEqual(x2, 180 / 320)
        self.assertGreaterEqual(y2, 140 / 240)
        self.assertLess((x2 - x1) * (y2 - y1), 0.9)

    def test_identical_pair_uses_full_frame_fallback(self) -> None:
        frame = np.full((120, 160, 3), 127, dtype=np.uint8)
        proposal = rect_qwen.estimate_change_region(frame, frame.copy(), index=0)
        self.assertTrue(proposal.fallback)
        self.assertEqual(proposal.expanded_box, (0.0, 0.0, 1.0, 1.0))

    def test_overall_decision_is_derived_from_three_checks(self) -> None:
        value = rect_qwen.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "pass",
                "visible_artifact": "fail",
                "content_preservation": "pass",
                "confidence": 0.91,
            }
        )
        self.assertEqual(value["reported_decision"], "accepted")
        self.assertEqual(value["decision"], "rejected")
        self.assertIn("inconsistent_overall_decision", value["reason_codes"])


class OpenRouterQwen3Tests(unittest.TestCase):
    def test_generated_side_ignores_redundant_quality_checked_roles(self) -> None:
        row = {
            "quality_checked_roles": "source",
            "source_provenance": "original",
            "target_provenance": "edited",
        }
        self.assertEqual(qwen3.derive_generated_side(row), "TARGET")

    def test_generated_side_falls_back_to_provenance(self) -> None:
        self.assertEqual(
            qwen3.derive_generated_side(
                {"source_provenance": "original", "target_provenance": "edited"}
            ),
            "TARGET",
        )
        self.assertEqual(
            qwen3.derive_generated_side(
                {"source_provenance": "edited", "target_provenance": "edited"}
            ),
            "BOTH",
        )

    def test_target_generated_uses_forward_reference_generated_view(self) -> None:
        view = qwen3.build_audit_view(
            {
                "source_provenance": "original",
                "target_provenance": "edited",
                "operation": "add",
                "instruction": "Place a tiny silver pin beside the collar.",
            }
        )
        self.assertEqual(view.left_training_role, "source")
        self.assertEqual(view.right_training_role, "target")
        self.assertEqual(view.left_label, "REFERENCE_VIDEO")
        self.assertEqual(view.right_label, "GENERATED_VIDEO")
        self.assertEqual(view.audit_operation, "add")
        self.assertEqual(view.audit_relation, "forward")

    def test_source_generated_inverts_add_structurally_without_rewriting_text(self) -> None:
        instruction = "Could you place a tiny silver pin beside the collar, please?"
        row = {
            "source_provenance": "edited",
            "target_provenance": "original",
            "operation": "add",
            "instruction": instruction,
        }
        view = qwen3.build_audit_view(row)
        self.assertEqual(view.left_training_role, "target")
        self.assertEqual(view.right_training_role, "source")
        self.assertEqual(view.audit_operation, "remove")
        self.assertEqual(view.audit_relation, "inverse")
        self.assertEqual(
            view.routing_transform,
            "swap_media_and_invert_structured_operation",
        )
        content = qwen3.build_content([], [], "add", instruction, view, "rubric", 90)
        prompt_text = "\n".join(
            item["text"] for item in content if item.get("type") == "text"
        )
        self.assertIn(instruction, prompt_text)
        self.assertIn("AUDIT_OPERATION：remove", prompt_text)
        self.assertNotIn("Remove a tiny silver pin", prompt_text)

    def test_unambiguous_audit_task_turns_training_command_into_description(self) -> None:
        instruction = "Add a pair of safety goggles worn by the man."
        view = qwen3.build_audit_view(
            {
                "source_provenance": "edited",
                "target_provenance": "original",
                "operation": "add",
                "instruction": instruction,
            }
        )
        task = qwen3.audit_task_text("add", instruction, view)
        self.assertIn("唯一审查任务", task)
        self.assertIn("完整去除了", task)
        self.assertIn("a pair of safety goggles", task)
        self.assertNotIn("Add a pair", task)

    def test_eyewear_route_requires_eyewear_to_be_the_remove_target(self) -> None:
        remove_score = {"audit_operation": "remove"}
        self.assertTrue(
            routed_v12.is_eyewear_remove_target(
                {"instruction": "Remove a pair of black rectangular eyeglasses."},
                remove_score,
            )
        )
        self.assertTrue(
            routed_v12.is_eyewear_remove_target(
                {"instruction": "Remove the dark eyewear from the woman's face."},
                remove_score,
            )
        )
        self.assertFalse(
            routed_v12.is_eyewear_remove_target(
                {"instruction": "Remove a shelving unit containing jars and glasses."},
                remove_score,
            )
        )
        self.assertFalse(
            routed_v12.is_eyewear_remove_target(
                {"instruction": "Remove the robot wearing heart-shaped sunglasses."},
                remove_score,
            )
        )
        self.assertFalse(
            routed_v12.is_eyewear_remove_target(
                {"instruction": "Add a pair of clear safety goggles."},
                {"audit_operation": "add"},
            )
        )

    def test_source_generated_inverts_remove_as_structured_add(self) -> None:
        view = qwen3.build_audit_view(
            {
                "source_provenance": "edited",
                "target_provenance": "original",
                "operation": "remove",
                "instruction": "Please make the patterned scarf disappear.",
            }
        )
        self.assertEqual(view.audit_operation, "add")
        self.assertEqual(view.audit_relation, "inverse")

    def test_inverse_replace_does_not_parse_or_mutate_complex_instruction(self) -> None:
        instruction = (
            "Keep the pose unchanged, but make the old amber brooch look like a "
            "small matte-blue badge with two asymmetric pins."
        )
        view = qwen3.build_audit_view(
            {
                "source_provenance": "edited",
                "target_provenance": "original",
                "operation": "replacement",
                "instruction": instruction,
            }
        )
        self.assertEqual(view.audit_operation, "replace")
        self.assertEqual(view.audit_relation, "inverse_replace")
        content = qwen3.build_content(
            [], [], "replacement", instruction, view, "rubric", 90
        )
        prompt_text = "\n".join(
            item["text"] for item in content if item.get("type") == "text"
        )
        self.assertIn(instruction, prompt_text)
        self.assertNotIn("AUDIT_INSTRUCTION", prompt_text)

    def test_both_generated_is_explicit_and_keeps_training_order(self) -> None:
        view = qwen3.build_audit_view(
            {
                "source_provenance": "edited",
                "target_provenance": "edited",
                "operation": "change",
                "instruction": "Change the cup into a vase.",
            }
        )
        self.assertEqual(view.left_training_role, "source")
        self.assertEqual(view.right_training_role, "target")
        self.assertEqual(view.audit_operation, "replace")
        self.assertEqual(view.audit_relation, "training_direction_both_generated")
        self.assertEqual(view.generated_position, "BOTH")

    def test_unknown_provenance_is_not_silently_treated_as_both_generated(self) -> None:
        row = {
            "source_provenance": "unknown",
            "target_provenance": "unknown",
            "operation": "add",
            "instruction": "Add a cup.",
        }
        self.assertEqual(qwen3.derive_generated_side(row), "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "cannot route audit media"):
            qwen3.build_audit_view(row)

    def test_explicit_unresolved_generation_side_uses_training_direction(self) -> None:
        row = {
            "source_provenance": "generation_side_unresolved",
            "target_provenance": "generation_side_unresolved",
            "operation": "remove",
            "instruction": "Remove the blue robot on the left.",
        }
        self.assertEqual(qwen3.derive_generated_side(row), "UNRESOLVED")
        view = qwen3.build_audit_view(row)
        self.assertEqual(view.left_training_role, "source")
        self.assertEqual(view.right_training_role, "target")
        self.assertEqual(view.left_label, "SRC_VIDEO")
        self.assertEqual(view.right_label, "TARGET_VIDEO")
        self.assertEqual(view.audit_operation, "remove")
        self.assertEqual(
            view.audit_relation,
            "training_direction_generation_side_unresolved",
        )
        self.assertEqual(view.generated_position, "UNRESOLVED_CHECK_BOTH")
        task = qwen3.audit_task_text("remove", row["instruction"], view)
        self.assertIn("SRC_VIDEO -> 右侧 TARGET_VIDEO", task)
        self.assertIn("visible_artifact 独立检查两侧", task)

    def test_roi_selection_omits_full_fallback_and_propagates_anchor_to_endpoints(self) -> None:
        proposals = [
            rect_qwen.RegionProposal(0, (0, 0, 1, 1), (0, 0, 1, 1), 0.99, 0.9, True),
            rect_qwen.RegionProposal(1, (0, 0, 1, 1), (0, 0, 1, 1), 0.70, 0.8, False),
            rect_qwen.RegionProposal(2, (0, 0, 1, 1), (0, 0, 1, 1), 0.69, 0.8, False),
            rect_qwen.RegionProposal(4, (0, 0, 1, 1), (0, 0, 1, 1), 0.68, 0.7, False),
        ]
        selected = qwen3.select_reliable_proposals(proposals, 2, 0.12)
        self.assertEqual([proposal.index for proposal in selected], [0, 4])
        self.assertEqual(selected[0].expanded_box, proposals[1].expanded_box)
        self.assertEqual(selected[0].change_score, 0.0)

    def test_five_roi_slots_cover_every_full_frame(self) -> None:
        proposals = [
            rect_qwen.RegionProposal(
                index,
                (0.2, 0.2, 0.5, 0.5),
                (0.1, 0.1, 0.6, 0.6),
                0.8 if index == 3 else 0.0,
                0.7 if index == 3 else 0.0,
                index != 3,
            )
            for index in range(5)
        ]
        selected = qwen3.select_reliable_proposals(proposals, 5, 0.12)
        self.assertEqual([proposal.index for proposal in selected], [0, 1, 2, 3, 4])
        self.assertEqual(sum(proposal.change_score == 0 for proposal in selected), 4)

    def test_tight_roi_is_capped_at_thirty_percent(self) -> None:
        box = qwen3.tighten_box((0.0, 0.0, 1.0, 1.0))
        width = box[2] - box[0]
        height = box[3] - box[1]
        self.assertLessEqual(width * height, 0.300001)
        self.assertGreaterEqual(box[0], 0.0)
        self.assertGreaterEqual(box[1], 0.0)
        self.assertLessEqual(box[2], 1.0)
        self.assertLessEqual(box[3], 1.0)

    def test_time_cards_keep_five_roi_slots_when_localizer_falls_back(self) -> None:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        cards, _proposals, selected = qwen3.prepare_time_cards(
            [(frame, frame.copy()) for _ in range(9)], 5, 64, 64, 64, 0.12
        )
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual([card.frame_index for card in cards], [1, 2, 4, 6, 8])
        self.assertEqual([card.time_percent for card in cards], [12, 25, 50, 75, 100])
        self.assertTrue(all(card.roi_left is not None for card in cards))
        self.assertTrue(all(proposal.fallback for proposal in selected))

    def test_overall_decision_is_mechanical_and_uncertain_is_narrow(self) -> None:
        rejected = qwen3.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "pass",
                "visible_artifact": "fail",
                "content_preservation": "pass",
                "reason_codes": ["Visible Artifact!"],
                "frame_audit": {},
                "temporal_audit": {},
                "evidence": {},
            }
        )
        self.assertEqual(rejected["decision"], "rejected")
        self.assertIn("inconsistent_overall_decision", rejected["reason_codes"])

        uncertain = qwen3.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "pass",
                "visible_artifact": "uncertain",
                "content_preservation": "pass",
                "reason_codes": [],
                "frame_audit": {},
                "temporal_audit": {},
                "evidence": {},
            }
        )
        self.assertEqual(uncertain["decision"], "uncertain")

    def test_frame_audit_failure_overrides_inconsistent_global_pass(self) -> None:
        frame_audit = {
            f"F{index}": {field: "pass" for field in qwen3.CHECK_FIELDS}
            for index in range(1, 6)
        }
        frame_audit["F2"]["instruction_following"] = "fail"
        value = qwen3.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "pass",
                "visible_artifact": "pass",
                "content_preservation": "pass",
                "frame_audit": frame_audit,
                "temporal_audit": {},
                "reason_codes": [],
                "evidence": {},
            }
        )
        self.assertEqual(value["instruction_following"], "fail")
        self.assertEqual(value["decision"], "rejected")

    def test_structured_removal_failure_overrides_global_pass(self) -> None:
        value = qwen3.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "pass",
                "visible_artifact": "pass",
                "content_preservation": "pass",
                "removal_audit": {
                    "applicable": "yes",
                    "reference_context": "hair around a cap",
                    "generated_fill": "a smooth cap-shaped patch",
                    "removed_region_quality": "fail",
                    "large_blur_fill": "pass",
                    "body_reconstruction": "not_applicable",
                    "hair_reconstruction": "fail",
                },
                "reason_codes": [],
                "evidence": {},
            },
            removal_expected=True,
        )
        self.assertEqual(value["visible_artifact"], "fail")
        self.assertEqual(value["decision"], "rejected")
        self.assertIn("hair_reconstruction", value["reason_codes"])

    def test_temporal_audit_failure_overrides_global_pass(self) -> None:
        temporal_audit = {
            f"T{index}": {field: "pass" for field in qwen3.TEMPORAL_CHECK_FIELDS}
            for index in range(1, 10)
        }
        temporal_audit["T4"]["instruction_following"] = "fail"
        value = qwen3.normalize_judgment(
            {
                "decision": "accepted",
                "instruction_following": "pass",
                "visible_artifact": "pass",
                "content_preservation": "pass",
                "frame_audit": {},
                "temporal_audit": temporal_audit,
                "reason_codes": [],
                "evidence": {},
            }
        )
        self.assertEqual(value["instruction_following"], "fail")
        self.assertEqual(value["decision"], "rejected")

    def test_openrouter_routing_is_private_and_has_no_fallback(self) -> None:
        payload = qwen3.build_payload("qwen/test", [], 256, "deny")
        self.assertEqual(payload["provider"]["data_collection"], "deny")
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertNotIn("removal_audit", schema["properties"])
        self.assertNotIn("removal_audit", schema["required"])
        self.assertNotIn("frame_audit", schema["properties"])
        self.assertNotIn("frame_audit", schema["required"])
        self.assertNotIn("temporal_audit", schema["properties"])
        self.assertNotIn("temporal_audit", schema["required"])

    def test_api_key_file_requires_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "openrouter_key"
            key_file.write_text("test-secret\n", encoding="utf-8")
            key_file.chmod(0o644)
            with self.assertRaises(PermissionError):
                qwen3.load_api_key("MISSING_TEST_OPENROUTER_KEY", key_file)
            key_file.chmod(0o600)
            self.assertEqual(
                qwen3.load_api_key("MISSING_TEST_OPENROUTER_KEY", key_file),
                "test-secret",
            )

    def test_api_key_environment_takes_precedence_without_logging(self) -> None:
        name = "TEST_OPENROUTER_KEY_PRECEDENCE"
        os.environ[name] = "environment-secret"
        try:
            self.assertEqual(qwen3.load_api_key(name, Path("/missing")), "environment-secret")
        finally:
            os.environ.pop(name, None)

    def test_calibration_requires_complete_scores_and_both_precision_gates(self) -> None:
        labels = []
        scores = []
        for operation in ("add", "remove", "replace"):
            for index in range(3):
                sample_id = f"{operation}-usable-{index}"
                labels.append({"sample_id": sample_id, "human_label": "usable"})
                scores.append(
                    {
                        "sample_id": sample_id,
                        "operation": operation,
                        "parse_status": "pass",
                        "model_decision": "accepted",
                        "judge_version": "test-v1",
                        "prompt_sha256": "test-hash",
                        "model_id": "qwen/test",
                        "model_backend": "openrouter",
                    }
                )
            sample_id = f"{operation}-unusable"
            labels.append({"sample_id": sample_id, "human_label": "unusable"})
            scores.append(
                {
                    "sample_id": sample_id,
                    "operation": operation,
                    "parse_status": "pass",
                    "model_decision": "rejected",
                    "judge_version": "test-v1",
                    "prompt_sha256": "test-hash",
                    "model_id": "qwen/test",
                    "model_backend": "openrouter",
                }
            )
        report, gates = evaluate_qwen3.evaluate(
            scores,
            labels,
            accept_precision_min=0.95,
            reject_precision_min=0.90,
            operation_accept_precision_min=0.90,
            minimum_operation_accepts=3,
        )
        self.assertEqual(report["calibration_status"], "enabled")
        self.assertEqual(gates["enabled_operations"], ["add", "remove", "replace"])

        _, incomplete = evaluate_qwen3.evaluate(
            scores[:-1],
            labels,
            accept_precision_min=0.95,
            reject_precision_min=0.90,
            operation_accept_precision_min=0.90,
            minimum_operation_accepts=3,
        )
        self.assertEqual(incomplete["calibration_status"], "pending_or_failed")

    def test_qwen3_formal_decision_obeys_frozen_gates(self) -> None:
        gates = {
            "calibration_status": "enabled",
            "accept": {"enabled": True},
            "reject": {"enabled": True},
            "operation_acceptance": {
                "add": {"enabled": True},
                "remove": {"enabled": False},
            },
        }
        accepted = {
            "model_backend": "openrouter",
            "parse_status": "pass",
            "model_decision": "accepted",
            "operation": "add",
        }
        self.assertEqual(apply_qwen3_gates.decide(accepted, gates)[0], "accepted")
        accepted["operation"] = "remove"
        self.assertEqual(apply_qwen3_gates.decide(accepted, gates)[0], "uncertain")
        accepted["model_decision"] = "rejected"
        self.assertEqual(apply_qwen3_gates.decide(accepted, gates)[0], "rejected")


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = {
            "calibration_status": "enabled",
            "accept": {"confidence_min": 0.8},
            "reject": {"confidence_min": 0.7},
            "operation_acceptance": {"remove": {"enabled": True}},
        }

    def test_qwen_accept_requires_calibrated_operation(self) -> None:
        row = {
            "model_backend": "transformers",
            "parse_status": "pass",
            "model_decision": "accepted",
            "model_confidence": "0.9",
            "operation": "remove",
        }
        self.assertEqual(apply_qwen.decide(row, self.thresholds)[0], "accepted")
        row["operation"] = "replace"
        self.assertEqual(apply_qwen.decide(row, self.thresholds)[0], "uncertain")

    def test_mock_scores_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            apply_qwen.decide({"model_backend": "mock"}, self.thresholds)

    def test_original_uncertain_can_be_promoted(self) -> None:
        row = {"route_stage1_2": "0", "stage1_1_decision": "uncertain"}
        decision = finalize.merge_decision(
            row,
            None,
            {"qwen_decision": "accepted", "qwen_reason": "qwen_edit_success"},
        )
        self.assertEqual(decision[0], "accepted")

    def test_robot_boundary_prevents_promotion(self) -> None:
        row = {"route_stage1_2": "1", "stage1_1_decision": "accepted"}
        decision = finalize.merge_decision(
            row,
            {"stage1_2_decision": "uncertain", "stage1_2_reason": "boundary"},
            {"qwen_decision": "accepted", "qwen_reason": "pass"},
        )
        self.assertEqual(decision[0], "uncertain")


if __name__ == "__main__":
    unittest.main()
