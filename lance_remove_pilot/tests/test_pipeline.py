from __future__ import annotations

import csv
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


PILOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_ROOT / "scripts"))

import plot_metrics  # noqa: E402
import prepare_smoke_data  # noqa: E402
import verify_training_run  # noqa: E402

RUNTIME_ROOT = PILOT_ROOT / "runtime" / "upstream"
sys.path.insert(0, str(RUNTIME_ROOT))
from train.pilot_monitor import TrainingMonitor  # noqa: E402


class PipelineTest(unittest.TestCase):
    def make_video(self, path: Path, color: tuple[int, int, int]) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            8.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for index in range(8):
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            frame[:, index : index + 4] = (255, 255, 255)
            writer.write(frame)
        writer.release()

    def make_tar(self, path: Path, members: dict[str, Path]) -> None:
        with tarfile.open(path, "w") as archive:
            for member_name, source in members.items():
                archive.add(source, arcname=member_name)

    def test_data_pipeline_and_loss_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            target = root / "target.mp4"
            self.make_video(source, (30, 60, 90))
            self.make_video(target, (90, 60, 30))

            insave_tar = root / "insave.tar"
            goku_tar = root / "goku.tar"
            self.make_tar(
                insave_tar,
                {"00001_original.mp4": source, "00001_traget.mp4": target},
            )
            self.make_tar(
                goku_tar,
                {"videos/case/source.mp4": source, "videos/case/edited.mp4": target},
            )
            rows = [
                {
                    "pair_id": "insave-00001",
                    "dataset": "insave",
                    "repo_id": "suimu/InsAVE-80K",
                    "revision": "rev-insave",
                    "official_id": "1",
                    "instruction": "Remove the person.",
                    "source_member": "00001_original.mp4",
                    "target_member": "00001_traget.mp4",
                },
                {
                    "pair_id": "goku-case",
                    "dataset": "goku",
                    "repo_id": "bigfacing/GOKU-2M",
                    "revision": "rev-goku",
                    "official_id": "case",
                    "instruction": "Remove the object.",
                    "source_member": "videos/case/source.mp4",
                    "target_member": "videos/case/edited.mp4",
                },
            ]
            prepare_smoke_data.extract_media(
                rows, {"insave": insave_tar, "goku": goku_tar}, root
            )
            inventory, lookup = prepare_smoke_data.build_inventory(rows, root)
            self.assertEqual(len(inventory), 4)
            self.assertTrue(all(row["decode_ok"] == "1" for row in inventory))
            db_path = root / "removal.sqlite"
            prepare_smoke_data.initialize_database(
                db_path, PILOT_ROOT / "schema.sql", rows, lookup, rebuild=False
            )
            outputs = prepare_smoke_data.export_parquets(db_path, root)
            prepare_smoke_data.verify_outputs(db_path, outputs, root, 2)
            self.assertEqual(set(outputs), {"insave", "goku"})
            self.assertEqual(pq.read_table(outputs["insave"]).num_rows, 1)
            with sqlite3.connect(db_path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0], 2)

            metrics = root / "metrics.csv"
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["step", "total_loss", "mse_loss"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"step": 0, "total_loss": 1.5, "mse_loss": 1.5},
                        {"step": 1, "total_loss": 1.2, "mse_loss": 1.2},
                    ]
                )
            svg_path = root / "loss_curve.svg"
            plot_metrics.render_svg(plot_metrics.read_metrics(metrics), svg_path, "test")
            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn("total_loss", svg)
            self.assertIn("mse_loss", svg)
            self.assertIn("<polyline", svg)

    def test_monitor_resume_and_strict_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            datasets = [
                {"dataset_name": "x2v_interleave_local:insave_smoke"},
                {"dataset_name": "x2v_interleave_local:goku_smoke"},
            ]
            first = TrainingMonitor(str(run_dir))
            for step in (0, 1):
                first.log(
                    step,
                    {
                        "total_loss": 1.5 - step * 0.1,
                        "mse_loss": 1.5 - step * 0.1,
                        "lr": 1e-5,
                        "grad_norm": 0.5,
                        "total_samples": 2,
                        "total_mse_tokens": 32,
                        "max_memory_allocated_mb": 1024,
                        "step_wall_time_seconds": 1,
                    },
                )
                first.log_datasets(step, datasets)
            first.close()

            resumed = TrainingMonitor(str(run_dir), purge_step=2)
            resumed.log(
                2,
                {
                    "total_loss": 1.2,
                    "mse_loss": 1.2,
                    "lr": 1e-5,
                    "grad_norm": 0.4,
                    "total_samples": 2,
                    "total_mse_tokens": 32,
                    "max_memory_allocated_mb": 1024,
                    "step_wall_time_seconds": 1,
                },
            )
            resumed.log_datasets(2, datasets)
            resumed.write_summary(3, 0)
            resumed.close()

            checkpoint = run_dir / "checkpoints" / "0000002"
            checkpoint.mkdir(parents=True)
            for name in verify_training_run.CHECKPOINT_FILES:
                (checkpoint / name).write_bytes(b"test")

            verify_training_run.verify_run(run_dir, 3)
            self.assertTrue((run_dir / "loss_curve.svg").is_file())


if __name__ == "__main__":
    unittest.main()
