#!/usr/bin/env python3
"""Package 36 optional paired reviews from the scored val48 outputs; no new inference."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import imageio_ffmpeg
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "model_evaluation/lance_tools"))
from build_cascade_review import build_html, probe_video
from verify_outputs import probe
from decide_funnel_promotion import BRANCHES, CANDIDATES
from judge_spatial import sha256_file, write_json


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = defaultdict(dict)
    for row in rows:
        step = int(row["checkpoint_step"])
        if step not in CANDIDATES or step in grouped[row["sample_id"]]:
            raise ValueError("unexpected checkpoint or duplicate score")
        grouped[row["sample_id"]][step] = row
    if len(rows) != 144 or len(grouped) != 48 or any(set(v) != set(CANDIDATES) for v in grouped.values()):
        raise ValueError("audit requires all 48 paired samples across three checkpoints")
    selected = []
    tie = lambda sid: hashlib.sha256(f"42\0audit\0{sid}".encode()).hexdigest()
    for branch in BRANCHES:
        samples = [sid for sid, values in grouped.items() if values[CANDIDATES[0]]["branch"] == branch]
        if len(samples) != 8:
            raise ValueError(f"expected 8 screen samples in {branch}")
        def scores(sid):
            return [float(grouped[sid][step]["overall_score"]) for step in CANDIDATES]
        worst = min(samples, key=lambda sid: (sum(scores(sid)), tie(sid)))
        spread = min((sid for sid in samples if sid != worst),
                     key=lambda sid: (-(max(scores(sid)) - min(scores(sid))), tie(sid)))
        for sid, category in ((worst, "lowest_mean_score"), (spread, "largest_checkpoint_difference")):
            for step in CANDIDATES:
                selected.append({**grouped[sid][step], "audit_category": category})
    return selected


def source_proxy(source: Path, destination: Path, frames: int, fps: float) -> list[int]:
    reader = VideoReader(str(source), ctx=cpu(0), num_threads=2)
    indices = np.linspace(0, len(reader) - 1, frames, dtype=int).tolist()
    # Lance's single-clip sampler uses the same linspace over the entire source.
    first = reader[0].asnumpy()
    height, width = first.shape[:2]
    scale = min(1.0, 720 / width, 480 / height)
    size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
    writer = imageio_ffmpeg.write_frames(str(destination), size, fps=fps, codec="libx264",
                                       pix_fmt_out="yuv420p", macro_block_size=2,
                                       output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart", "-threads", "2"])
    writer.send(None)
    try:
        for index in indices:
            frame = Image.fromarray(reader[index].asnumpy()).resize(size, Image.Resampling.LANCZOS)
            writer.send(np.asarray(frame))
    finally:
        writer.close()
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8", newline="") as stream:
        selected = select_rows(list(csv.DictReader(stream)))
    args.output.mkdir(parents=True, exist_ok=False)
    media = args.output / "media"
    media.mkdir()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    sources = {}
    review = []
    for index, row in enumerate(selected, 1):
        generated = Path(row["generated_video"])
        info = probe(generated)
        if float(info["fps"]) != 12.0:
            raise ValueError(f"unexpected inference FPS: {generated}")
        source = Path(row["source_path"])
        source_key = (str(source), int(info["frames"]))
        if source_key not in sources:
            source_id = hashlib.sha256(str(source_key).encode()).hexdigest()[:16]
            proxy = media / f"source_{source_id}.mp4"
            indices = source_proxy(source, proxy, int(info["frames"]), 12.0)
            sources[source_key] = (proxy, indices)
        proxy, indices = sources[source_key]
        output = media / f"output_{index:03d}_step{int(row['checkpoint_step'])}.mp4"
        codec, pixels = probe_video(ffmpeg, generated)
        encoding = ["-c:v", "copy"] if codec == "h264" and pixels == "yuv420p" else [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"]
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(generated),
                        "-map", "0:v:0", "-an", *encoding, "-movflags", "+faststart", "-threads", "2", str(output)], check=True)
        original_id = row["sample_id"]
        review.append({
            **row, "source_sample_id": original_id,
            "sample_id": f"step{row['checkpoint_step']}_{original_id}",
            "dataset": row["branch"].split("_")[0], "evaluation_split": "validation_screen48",
            "source_provenance": "model_input", "target_provenance": "lance_generated",
            "audit_left_label": "INPUT_VIDEO (frames used by Lance)",
            "audit_right_label": f"MODEL_OUTPUT step {row['checkpoint_step']}",
            "audit_generated_position": "RIGHT", "review_group": row["audit_category"],
            "review_source_path": str(proxy.relative_to(args.output)),
            "review_target_path": str(output.relative_to(args.output)),
            "source_frame_indices_json": json.dumps(indices),
            "source_original_sha256": sha256_file(source), "generated_original_sha256": info["sha256"],
        })
    version = "lance_4k_seed42_val48_audit_" + sha256_file(args.scores)[:12]
    build_html(args.output, review, "Lance 4K · checkpoint 对照审核",
               "同一分支选择低分与分歧样本，逐一对照 1/1.5/2 epoch；右侧为模型输出。审核仅供留证，不改变自动选择结果。",
               version, model_evaluation=True)
    with (args.output / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review[0]))
        writer.writeheader(); writer.writerows(review)
    html = (args.output / "index.html").read_text()
    embedded = re.search(r'<script id="surveyData" type="application/json">(.*?)</script>', html, re.S)
    if embedded is None or len(json.loads(embedded[1])["items"]) != 36:
        raise ValueError("invalid embedded review data")
    for script in re.findall(r"<script>(.*?)</script>", html, re.S):
        subprocess.run(["node", "--check"], input=script, text=True, check=True)
    files = []
    for path in sorted(media.glob("*.mp4")):
        codec, pixels = probe_video(ffmpeg, path)
        if codec != "h264" or pixels != "yuv420p":
            raise ValueError(f"review media incompatible: {path}")
        files.append({"file": str(path.relative_to(args.output)), **probe(path)})
    for row in review:
        left = probe(args.output / row["review_source_path"])
        right = probe(args.output / row["review_target_path"])
        if (left["frames"], left["fps"]) != (right["frames"], right["fps"]):
            raise ValueError("review pair is not temporally aligned")
    write_json(args.output / "summary.json", {
        "review_version": version, "rows": 36, "unique_samples": 12,
        "selection_seed": 42, "human_review_required": False,
        "selection_changes_checkpoint": False, "source_scores_sha256": sha256_file(args.scores),
        "source_proxy_policy": "Lance full-clip linspace frame indices, played at output FPS; display resize only",
        "media": files,
    })
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(args.output.name) / path.relative_to(args.output))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or {name.split("/")[0] for name in bundle.namelist()} != {args.output.name}:
            raise ValueError("archive CRC or root layout check failed")
    write_json(archive.with_suffix(".zip.sha256.json"), {
        "file": str(archive.resolve()), "bytes": archive.stat().st_size, "sha256": sha256_file(archive),
    })
    print(json.dumps({"archive": str(archive), "review_pairs": 36, "new_inference_videos": 0}))


if __name__ == "__main__":
    main()
