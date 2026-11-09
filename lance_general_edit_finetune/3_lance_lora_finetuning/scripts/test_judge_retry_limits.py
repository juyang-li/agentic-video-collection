from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model_evaluation/lance_tools"))
import judge_spatial_openrouter as judge


class RetryLimitTests(unittest.TestCase):
    def check_limit(self, count, accepted):
        argv = ["judge", "--tasks", "unused.json", "--output", "unused", "--max-attempts", str(count)]
        with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            # Stop immediately after argument validation: no filesystem/API work.
            with patch.object(judge, "configure_route", side_effect=RuntimeError("bounds accepted")):
                if accepted:
                    with self.assertRaisesRegex(RuntimeError, "bounds accepted"):
                        judge.main()
                else:
                    with self.assertRaises(SystemExit) as result:
                        judge.main()
                    self.assertEqual(result.exception.code, 2)

    def test_explicit_bounded_recovery(self):
        for count in (1, 4, 8, 9, 12):
            with self.subTest(count=count):
                self.check_limit(count, True)

    def test_invalid_or_excessive_attempts_rejected(self):
        for count in (0, -1, 13, 100):
            with self.subTest(count=count):
                self.check_limit(count, False)


if __name__ == "__main__":
    unittest.main()
