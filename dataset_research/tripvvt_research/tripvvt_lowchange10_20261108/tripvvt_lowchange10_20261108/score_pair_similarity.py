#!/usr/bin/env python3
"""CPU-only TripVVT pair metrics. Scores low change; does not label semantic no-op.

CSV input: sample_id, category, source, target; optional reference, cohort,
source_frames, target_frames, source_fps, target_fps. Paths are relative to the CSV.
Dependencies: Python, numpy, Pillow, FFmpeg/ffprobe. No learned weights or GPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

VERSION = "tripvvt_cpu_lowchange_v1"
FRACTIONS = (.1, .3, .5, .7, .9)
REGIONS = {"upper_body": (.15, .60), "lower_body": (.45, .95), "dresses": (.15, .95)}
W, H = 144, 256
DCT = np.cos(np.pi * (2 * np.arange(32)[None, :] + 1) * np.arange(8)[:, None] / 64)
DCT[0] *= 1 / np.sqrt(2)
DCT *= np.sqrt(2 / 32)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def probe(path, ffprobe):
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate", "-of", "json", str(path)],
        capture_output=True, check=True, text=True)
    info = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(float, info["avg_frame_rate"].split("/"))
    return int(info["nb_frames"]), numerator / denominator


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty manifest")
    ids = set()
    for row in rows:
        if not row.get("sample_id") or row["sample_id"] in ids:
            raise ValueError("sample_id must be nonempty and unique")
        ids.add(row["sample_id"])
        if row.get("category") not in REGIONS:
            raise ValueError("category must be upper_body, lower_body or dresses")
        for key in ("source", "target", "reference"):
            if row.get(key):
                p = Path(row[key])
                p = p if p.is_absolute() else path.parent / p
                if not p.is_file():
                    raise FileNotFoundError(p)
                row[key] = str(p.resolve())
    return rows


def frame_indices(row, ffprobe):
    specs = {}
    for role in ("source", "target"):
        if row.get(role + "_frames") and row.get(role + "_fps"):
            frames, fps = int(row[role + "_frames"]), float(row[role + "_fps"])
        else:
            frames, fps = probe(row[role], ffprobe)
        if frames <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("invalid frame count or FPS")
        specs[role] = (frames, fps)
    # Same absolute times on both sides, not independent whole-clip percentiles.
    last_time = min((frames - 1) / fps for frames, fps in specs.values())
    times = [last_time * fraction for fraction in FRACTIONS]
    indices = {role: [min(frames - 1, int(np.rint(t * fps))) for t in times]
               for role, (frames, fps) in specs.items()}
    return specs, times, indices


def decode(path, indices, ffmpeg):
    unique = sorted(set(indices))
    vf = "select=" + "+".join(f"eq(n\\,{i})" for i in unique)
    vf += f",scale={W}:{H}:flags=area,format=rgb24"
    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
        "-threads", "1", "-filter_threads", "1", "-i", str(path), "-vf", vf,
        "-fps_mode", "passthrough", "-an", "-threads", "1", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
    values = np.frombuffer(result.stdout, dtype=np.uint8)
    if values.size != len(unique) * H * W * 3:
        raise ValueError("unexpected extracted frame count")
    frames = values.reshape(len(unique), H, W, 3)
    positions = {n: i for i, n in enumerate(unique)}
    return frames[[positions[n] for n in indices]]


def phash63(frame):
    small = np.asarray(Image.fromarray(frame).convert("L").resize(
        (32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = (DCT @ small @ DCT.T).ravel()[1:]
    return coefficients > np.median(coefficients)


def block_ssim8(source, target):
    # Explicit 8x8 non-overlapping block SSIM; not skimage's sliding-window SSIM.
    a = (source.astype(np.float32) * [.299, .587, .114]).sum(-1) / 255
    b = (target.astype(np.float32) * [.299, .587, .114]).sum(-1) / 255
    a = a.reshape(5, H // 8, 8, W // 8, 8).transpose(0, 1, 3, 2, 4).reshape(5, -1, 64)
    b = b.reshape(5, H // 8, 8, W // 8, 8).transpose(0, 1, 3, 2, 4).reshape(5, -1, 64)
    ma, mb = a.mean(-1), b.mean(-1)
    va, vb = a.var(-1), b.var(-1)
    covariance = ((a - ma[..., None]) * (b - mb[..., None])).mean(-1)
    return (((2 * ma * mb + .01 ** 2) * (2 * covariance + .03 ** 2)) /
            ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2))).mean(-1)


def calculate_metrics(source, target, category):
    difference = np.abs(source.astype(np.float32) - target.astype(np.float32)) / 255
    pixel_difference = difference.mean(-1)
    ssim = block_ssim8(source, target)
    hamming = [int(np.count_nonzero(phash63(a) != phash63(b))) for a, b in zip(source, target)]
    y0, y1 = REGIONS[category]
    roi = pixel_difference[:, int(H * y0):int(H * y1), int(W * .15):int(W * .85)]
    return dict(ssim_median=float(np.median(ssim)), ssim_min=float(np.min(ssim)),
        phash_median=float(np.median(hamming)), phash_max=max(hamming),
        rgb_mae_median=float(np.median(difference.mean((1, 2, 3)))),
        roi_mae_median=float(np.median(roi.mean((1, 2)))),
        changed_fraction_median=float(np.median((pixel_difference > .05).mean((1, 2)))),
        roi_changed_fraction_median=float(np.median((roi > .05).mean((1, 2)))),
        per_frame_ssim=ssim.tolist(), per_frame_phash=hamming,
        per_frame_roi_mae=roi.mean((1, 2)).tolist())


def score_pair(row, ffmpeg, ffprobe):
    start = time.perf_counter()
    specs, times, indices = frame_indices(row, ffprobe)
    after_probe = time.perf_counter()
    source = decode(row["source"], indices["source"], ffmpeg)
    target = decode(row["target"], indices["target"], ffmpeg)
    after_decode = time.perf_counter()
    scores = calculate_metrics(source, target, row["category"])
    end = time.perf_counter()
    unique_count = min(len(set(v)) for v in indices.values())
    return dict(**row, status="ok", **scores, sample_times_seconds=times,
        source_indices=indices["source"], target_indices=indices["target"],
        unique_sample_count=unique_count, insufficient_unique_frames=unique_count < 3,
        probe_seconds=after_probe-start, decode_seconds=after_decode-after_probe,
        metric_seconds=end-after_decode, wall_seconds=end-start)


def safe_score(row, ffmpeg, ffprobe):
    try:
        return score_pair(row, ffmpeg, ffprobe)
    except Exception as error:
        return dict(**row, status="error", error_type=type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output prefix (.csv and .json)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true", help="Also time the first N pairs serially")
    parser.add_argument("--serial-samples", type=int, default=10)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/scratch/prj0000000275/jcy/.conda/envs/vivid/bin/ffmpeg")
    parser.add_argument("--ffprobe", default=None)
    args = parser.parse_args()
    if args.workers < 1 or args.serial_samples < 1:
        parser.error("worker and serial sample counts must be positive")
    ffprobe = args.ffprobe or str(Path(args.ffmpeg).with_name("ffprobe"))
    rows = read_manifest(args.manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    serial = []
    if args.benchmark:
        started = time.perf_counter()
        serial = [safe_score(r, args.ffmpeg, ffprobe) for r in rows[:args.serial_samples]]
        serial_wall = time.perf_counter() - started
        print(f"serial {len(serial)} pairs: {serial_wall:.3f}s", flush=True)
    started = time.perf_counter()
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(safe_score, r, args.ffmpeg, ffprobe): r for r in rows}
        for future in as_completed(futures):
            result = future.result()
            results[result["sample_id"]] = result
            if len(results) % 15 == 0:
                print(f"scored {len(results)}/{len(rows)}", flush=True)
    parallel_wall = time.perf_counter() - started
    ordered = [results[r["sample_id"]] for r in rows]
    good = [r for r in ordered if r["status"] == "ok"]
    report = dict(algorithm_version=VERSION, script_sha256=sha(__file__),
        input_manifest_sha256=sha(args.manifest), pair_count=len(rows), successful=len(good),
        errors=len(rows)-len(good), workers=args.workers, cpu_only=True,
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
            pillow=PILLOW_VERSION, available_cpu_count=len(os.sched_getaffinity(0)),
            ffmpeg=subprocess.run([args.ffmpeg, "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]),
        parameters=dict(fractions=FRACTIONS, time_alignment="shared last-frame timestamp",
            thumbnail_width=W, thumbnail_height=H, resize="FFmpeg area RGB24",
            phash="63 AC DCT coefficients from 8x8 of 32x32 grayscale, median threshold",
            ssim="non-overlapping 8x8 grayscale blocks, C1=.01^2 C2=.03^2, population variance",
            region_x=[.15,.85], region_y=REGIONS, changed_pixel_rgb_mae_threshold=.05),
        parallel_seconds=parallel_wall, parallel1000_seconds=parallel_wall/len(rows)*1000,
        serial_records=serial, records=ordered,
        limitations=["No semantic no-op labels or automatic exclusion.",
            "Region is a fixed category-specific crop, not garment segmentation.",
            "Timing excludes downloads, uses already downloaded local files and may hit OS cache.",
            "Very short clips may repeat sampled frame indices; unique_sample_count records this."])
    if args.benchmark:
        report.update(serial10_seconds=serial_wall, serial_sample_count=len(serial),
            serial1000_seconds=serial_wall/len(serial)*1000)
    if good:
        report["metric_quantiles"] = {k: dict(zip(["min","p05","p10","p25","p50","p75","p90","p95","max"],
            np.quantile([r[k] for r in good], [0,.05,.1,.25,.5,.75,.9,.95,1]).tolist()))
            for k in ("ssim_median","phash_median","rgb_mae_median","roi_mae_median")}
    columns = list(dict.fromkeys(key for row in ordered for key in row))
    csv_path = args.out.with_suffix(".csv")
    with csv_path.with_suffix(".csv.part").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    csv_path.with_suffix(".csv.part").replace(csv_path)
    save_json(args.out.with_suffix(".json"), report)
    print(json.dumps(dict(pairs=len(rows), errors=report["errors"], parallel_seconds=parallel_wall,
        estimated_1000_seconds=report["parallel1000_seconds"], output=str(args.out)), ensure_ascii=False), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
