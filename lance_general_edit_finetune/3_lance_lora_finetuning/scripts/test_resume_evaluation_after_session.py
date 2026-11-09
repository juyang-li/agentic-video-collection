import unittest
from unittest.mock import patch

import resume_evaluation_after_session as recovery


class ProcessIdentityTests(unittest.TestCase):
    def test_original_process_is_live(self):
        with patch.object(recovery, "process_state", return_value=(123, "S")):
            self.assertTrue(recovery.original_alive(10, 123))

    def test_missing_process_is_terminal(self):
        with patch.object(recovery, "process_state", return_value=None):
            self.assertFalse(recovery.original_alive(10, 123))

    def test_pid_reuse_does_not_adopt_another_process(self):
        with patch.object(recovery, "process_state", return_value=(124, "R")):
            self.assertFalse(recovery.original_alive(10, 123))

    def test_zombie_is_not_a_running_worker(self):
        with patch.object(recovery, "process_state", return_value=(123, "Z")):
            self.assertFalse(recovery.original_alive(10, 123))

    def test_observation_failure_is_not_treated_as_exit(self):
        with patch.object(recovery, "process_state", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                recovery.original_alive(10, 123)

    def test_duplicate_wrapper_pids_rejected(self):
        with self.assertRaises(ValueError):
            recovery.identify_wrappers([10, 10, 11, 12])


if __name__ == "__main__":
    unittest.main()
