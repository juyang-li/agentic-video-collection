"""Portable CPU regression tests; GPU integration is exercised with the same demo CLI."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from common import crop_box, lossless_video, read_manifest, select_frames
from gt_metrics import frame_metrics
from make_demo import make_demo


class ExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_demo(cls.root / "data")
        cls.times = [i / 12 for i in range(121)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_kernels(self):
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        same = frame_metrics((a, a))
        self.assertEqual(same["ssim"], 1)
        self.assertTrue(same["psnr_infinite"])
        self.assertIsNone(same["psnr_db"])
        different = frame_metrics((a, a + 20))
        self.assertAlmostEqual(different["psnr_db"], 20 * np.log10(255 / 20), places=5)
        self.assertLess(different["ssim"], 1)

    def test_pts_and_tail(self):
        a, _ = select_frames(self.root / "data/gt.mp4", self.times, "gt")
        b, audit = select_frames(self.root / "data/prediction24.mp4", self.times, "prediction")
        self.assertEqual([x["native_index"] for x in audit["selected"]], list(range(0, 241, 2)))
        self.assertEqual(audit["native_frame_count"], 243)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_endpoint_and_missing_prediction(self):
        frame = np.full((36, 64, 3), 80, dtype=np.uint8)
        gt = self.root / "gt30.mp4"
        lossless_video(gt, [frame] * 300, 30)
        _, audit = select_frames(gt, self.times, "gt")
        self.assertTrue(audit["selected"][-1]["gt_endpoint_quantized"])
        short = self.root / "short.mp4"
        lossless_video(short, [frame] * 120, 12)
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            select_frames(short, self.times, "prediction")

    def test_contain_and_portrait(self):
        self.assertEqual(crop_box(1280, 736, contain_aspect=(1920, 1080)), [0, 8, 1280, 720])
        self.assertEqual(crop_box(736, 1280, contain_aspect=(1080, 1920)), [8, 0, 720, 1280])
        with self.assertRaises(ValueError):
            crop_box(64, 36, explicit=[0, 0, 65, 36])

    def test_duplicate_manifest(self):
        row = json.loads((self.root / "data/manifest.jsonl").read_text())
        row.update(prediction=str(self.root / "data/prediction24.mp4"), gt=str(self.root / "data/gt.mp4"))
        p = self.root / "duplicate.jsonl"
        p.write_text((json.dumps(row) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_manifest(p)

    def test_detached_cli_and_timing(self):
        # Copy the actual distributed scripts, not an import from this repository.
        copied = self.root / "detached"
        copied.mkdir(exist_ok=True)
        for name in ["common.py", "gt_metrics.py"]:
            shutil.copyfile(Path(__file__).parent / name, copied / name)
        output = self.root / "cpu_result"
        subprocess.run([sys.executable, str(copied / "gt_metrics.py"), "--manifest", str(self.root / "data/manifest.jsonl"),
                        "--output-dir", str(output), "--metrics", "ssim,psnr", "--device", "cpu", "--size", "128x72"],
                       cwd=copied, check=True)
        result = json.loads((output / "results.jsonl").read_text())
        self.assertEqual(result["mean"]["ssim"], 1)
        self.assertTrue(result["mean"]["psnr_infinite"])
        timing = json.loads((output / "timing.json").read_text())
        self.assertEqual(timing["successful_videos"], 1)
        self.assertEqual(timing["frame_pairs_or_output_frames"], 121)
        self.assertAlmostEqual(timing["observed_videos_per_hour"], 3600 / timing["total_wall_seconds"])
        self.assertLessEqual(sum(timing["stage_seconds"].values()), timing["total_wall_seconds"] + 1e-6)
        self.assertTrue((output / "timing_report.md").exists())


if __name__ == "__main__":
    unittest.main()
