import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import monitor_evaluation as monitor


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = {"stage": "test", "checkpoint_step": 677, "sample_id": "sample",
                     "generated_video": str(self.root / "sample.mp4")}
        (self.root / "sample.mp4").write_bytes(b"fixture-not-a-real-video")
        self.dump("test_generation_tasks.json", [self.task])
        self.dump("test_jobs.json", {"jobs": [{"run_name": "test_shard00"}]})
        technical = self.root / "technical/test_shard00.csv"
        technical.parent.mkdir()
        technical.write_text("sample_index,full_decode_ok\n1,1\n")
        self.summary = {"scope": "full_manifest", "status": "complete",
                        "tasks_succeeded": 1, "tasks_failed": 0}
        self.dump("test_scores_vertex_v2/summary.json", self.summary)
        self.record = {"status": "success", "identity": {
            "task": self.task, "protocol_sha256": "frozen"}}
        self.dump("test_scores_vertex_v2/records/sample/result.json", self.record)
        self.counts = patch.dict(monitor.COUNTS, {"test": 1})
        self.counts.start()
        self.addCleanup(self.counts.stop)

    def dump(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def status(self):
        return monitor.stage_snapshot(self.root, "test", "frozen")

    def test_complete_is_only_reported_not_a_decode_audit(self):
        self.assertTrue(self.status()["complete_reported"])

    def test_score_failure_does_not_hide_completed_generation(self):
        self.record["status"] = "failed"
        self.dump("test_scores_vertex_v2/records/sample/result.json", self.record)
        self.summary.update(status="incomplete", tasks_succeeded=0, tasks_failed=1)
        self.dump("test_scores_vertex_v2/summary.json", self.summary)
        result = self.status()
        self.assertEqual(result["generated_files"], 1)
        self.assertEqual(result["successful_receipts"], 0)
        self.assertFalse(result["complete_reported"])

    def test_wrong_protocol_is_not_counted(self):
        self.record["identity"]["protocol_sha256"] = "other"
        self.dump("test_scores_vertex_v2/records/sample/result.json", self.record)
        self.assertEqual(self.status()["invalid_receipts"], 1)
        self.assertFalse(self.status()["complete_reported"])

    def test_duplicate_tasks_are_not_complete(self):
        self.dump("test_generation_tasks.json", [self.task, self.task])
        self.assertFalse(self.status()["complete_reported"])

    def test_missing_video_is_not_complete(self):
        (self.root / "sample.mp4").unlink()
        self.assertFalse(self.status()["complete_reported"])


if __name__ == "__main__":
    unittest.main()
