"""A failed observation must not be treated as a stopped generation task."""

import subprocess
import unittest
from unittest.mock import patch

import watch_generation_and_evaluate as monitor


class MonitorTests(unittest.TestCase):
    def test_live_pane(self):
        with patch.object(monitor.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            self.assertTrue(monitor.generation_pane_live("test"))

    def test_missing_pane(self):
        for message in ("can't find session: generation", "no server running on /tmp/test", "No such file or directory"):
            with patch.object(monitor.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", message)):
                self.assertFalse(monitor.generation_pane_live("test"))

    def test_observation_failure(self):
        for message in ("Operation not permitted", "Permission denied", "Connection reset by peer", "timeout"):
            with patch.object(monitor.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", message)):
                with self.assertRaises(RuntimeError):
                    monitor.generation_pane_live("test")


if __name__ == "__main__":
    unittest.main()
