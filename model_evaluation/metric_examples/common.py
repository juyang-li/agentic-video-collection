"""Small shared video/measurement helpers. No dataset or scheduler dependencies."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import av
import numpy as np
from PIL import Image


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path, require_gt=False):
    path = Path(path).resolve()
    rows, keys = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Every row needs a non-empty string id")
        row.setdefault("model", "default")
        if not isinstance(row["model"], str) or not row["model"]:
            raise ValueError("model must be a non-empty string")
        key = (row["id"], row["model"])
        if key in keys:
            raise ValueError(f"Duplicate (id, model): {key}")
        keys.add(key)
        for field in ("prediction", "gt") if require_gt else ("prediction",):
            if not row.get(field):
                raise ValueError(f"{key}: missing {field}")
            p = Path(row[field])
            p = p if p.is_absolute() else path.parent / p
            if not p.is_file():
                raise ValueError(f"{key}: file does not exist: {p}")
            row[field] = str(p.resolve())
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def common_arguments(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Pilot: process the first N rows; arrange representative rows first")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=10)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--endpoint", choices=["inclusive", "exclusive"], default="inclusive")
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cache-state", choices=["unknown", "cold", "warm"], default="unknown",
                        help="User-declared cache condition, not automatically inferred")


def targets(args):
    if not (math.isfinite(args.start) and math.isfinite(args.end) and math.isfinite(args.fps)):
        raise ValueError("Timeline arguments must be finite")
    count = (args.end - args.start) * args.fps
    if args.start < 0 or args.fps <= 0 or count <= 0 or abs(count - round(count)) > 1e-7:
        raise ValueError("Require start>=0, end>start, fps>0 and integer (end-start)*fps")
    if min(args.decode_threads, args.torch_threads) < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Thread count and limit must be positive")
    return [args.start + i / args.fps for i in range(round(count) + (args.endpoint == "inclusive"))]


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return {"width": s.width, "height": s.height,
                "nominal_fps": float(s.average_rate) if s.average_rate else None,
                "codec": s.codec_context.name, "native_frames_header": s.frames}


def crop_box(width, height, explicit=None, contain_aspect=None):
    if explicit is not None:
        if len(explicit) != 4 or any(type(x) is not int for x in explicit):
            raise ValueError("crop must be integer [x,y,width,height]")
        x, y, w, h = explicit
    elif contain_aspect is not None:
        ow, oh = contain_aspect
        scale = min(width / ow, height / oh)
        w, h = round(ow * scale), round(oh * scale)
        x, y = (width - w) // 2, (height - h) // 2
    else:
        x, y, w, h = 0, 0, width, height
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        raise ValueError("crop lies outside the decoded image")
    return [x, y, w, h]


def matching_aspect(a, b):
    # Permit at most two pixels of integer contain-rounding, not arbitrary stretching.
    return abs(a[0] * b[1] - a[1] * b[0]) <= 2 * max(a[0], a[1], b[0], b[1])


def select_frames(path, times, role, threads=2, box=None, size=None):
    """Decode the full stream; choose nearest relative PTS, ties later. No output upsampling."""
    chosen, selections = [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.codec_context.thread_count = threads
        nominal = float(stream.average_rate) if stream.average_rate else 0
        if nominal <= 0:
            raise ValueError("A positive nominal fps is required by this small example")
        interval = 1 / nominal
        state = {"count": 0, "origin": None, "last": -math.inf, "end_display": None, "wh": None}

        def decode():
            for index, frame in enumerate(c.decode(stream)):
                if frame.pts is None:
                    raise ValueError("Missing presentation timestamp")
                absolute = float(frame.pts * frame.time_base)
                if state["origin"] is None:
                    state["origin"] = absolute
                pts = absolute - state["origin"]
                if pts <= state["last"]:
                    raise ValueError("Non-increasing presentation timestamps")
                wh = [frame.width, frame.height]
                if state["wh"] is not None and wh != state["wh"]:
                    raise ValueError("Resolution changes inside one video")
                state.update(count=index + 1, last=pts, wh=wh)
                duration = getattr(frame, "duration", 0)
                duration = float(duration * frame.time_base) if duration else interval
                state["end_display"] = pts + duration
                yield frame, pts, index

        iterator = iter(decode())
        current, previous = next(iterator, None), None
        used = set()
        for target in times:
            while current is not None and current[1] < target - 1e-9:
                previous, current = current, next(iterator, None)
            candidates = [x for x in (previous, current) if x is not None]
            if not candidates:
                raise ValueError("No decodable video frames")
            frame, pts, index = min(candidates, key=lambda x: (round(abs(x[1] - target), 9), -x[1]))
            edge = (role == "gt" and current is None and target == times[-1]
                    and abs(pts - target) <= interval + 1e-6
                    and state["end_display"] + 1e-6 >= target)
            if abs(pts - target) > interval / 2 + 1e-6 and not edge:
                raise ValueError(f"{role} cannot cover target {target:.6f}s (nearest {pts:.6f}s)")
            if index in used and not edge:
                raise ValueError(f"{role}: duplicate selected frame; lower sampling fps or fix coverage")
            used.add(index)
            image = frame.to_image()
            if box is not None:
                x, y, w, h = box
                image = image.crop((x, y, x + w, y + h))
            if size is not None and image.size != tuple(size):
                image = image.resize(tuple(size), Image.Resampling.LANCZOS)
            chosen.append(np.array(image, dtype=np.uint8))
            selections.append({"target_seconds": target, "relative_pts": pts, "native_index": index,
                               "gt_endpoint_quantized": edge})
        # Also validate native frames after the evaluation interval, without retaining their RGB arrays.
        for _ in iterator:
            pass
    return chosen, {"native_frame_count": state["count"], "nominal_fps": nominal,
                    "native_wh": state["wh"], "origin_pts_seconds": state["origin"],
                    "end_display_seconds": state["end_display"], "selected": selections}


def lossless_video(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264rgb", rate=Fraction(str(fps)).limit_denominator(10000))
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt, s.thread_count = "rgb24", 2
        s.options = {"crf": "0", "preset": "fast"}
        for array in frames:
            for packet in s.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    expected = [hashlib.sha256(a.tobytes()).hexdigest() for a in frames]
    with av.open(str(path)) as c:
        actual = [hashlib.sha256(f.to_ndarray(format="rgb24").tobytes()).hexdigest() for f in c.decode(video=0)]
    if expected != actual:
        raise ValueError("Lossless scoring video failed RGB pixel verification")
    return {"sha256": digest(path), "pixel_preservation_verified": True, "frames": len(actual)}


def sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def hardware():
    versions = {}
    for name in ["torch", "torchvision", "numpy", "av", "Pillow", "scikit-image", "lpips", "pyiqa"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"os": platform.system(), "architecture": platform.machine(), "cpu": cpu,
            "available_logical_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
            "versions": versions}


class RunTimer:
    def __init__(self, start, imports_seconds):
        self.start, self.events = start, [{"stage": "startup_imports_and_arguments", "seconds": imports_seconds}]
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.hardware = hardware()

    @contextmanager
    def stage(self, name, **details):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.events.append({"stage": name, "seconds": time.perf_counter() - start, **details})

    def finish(self, output, records, total_rows, protocol, signatures, device, notes=()):
        elapsed = time.perf_counter() - self.start
        successful = [r for r in records if r["status"] == "complete"]
        frames = sum(r.get("num_frames", 0) for r in successful)
        estimate = {"estimated_total_seconds": None, "method": "inclusive pilot wall time scaled by video count"}
        if len(successful) != len(records):
            estimate["reason"] = "Failed/incomplete rows: correct inputs before forecasting"
        elif len(set(signatures)) != 1 or len(signatures) != total_rows:
            estimate["reason"] = "Mixed/unknown frame geometry: make separate representative pilots per resolution"
        elif records:
            estimate.update(estimated_total_seconds=elapsed * total_rows / len(records),
                            reason="Same geometry and timeline only; includes first-run loading/download overhead; not a confidence bound")
        totals = defaultdict(float)
        for event in self.events:
            totals[event["stage"]] += event["seconds"]
        gpu = None
        if str(device).startswith("cuda"):
            import torch
            gpu = {"device": str(device), "name": torch.cuda.get_device_name(device),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "torch_cuda": torch.version.cuda}
        result = {"status": "complete" if len(successful) == len(records) else "incomplete",
                  "started_utc": self.started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_videos": total_rows, "sampled_videos": len(records), "successful_videos": len(successful),
                  "failed_videos": len(records) - len(successful), "frame_pairs_or_output_frames": frames,
                  "total_wall_seconds": elapsed, "observed_videos_per_hour": len(successful) * 3600 / elapsed,
                  "observed_frames_per_second": frames / elapsed, "stage_seconds": dict(totals),
                  "unattributed_overhead_seconds": max(0, elapsed - sum(totals.values())),
                  "events": self.events, "hardware": self.hardware, "gpu": gpu,
                  "peak_process_rss_bytes_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "protocol": protocol, "estimate": estimate, "notes": list(notes)}
        write_json(Path(output) / "timing.json", result)
        lines = ["# Measurement timing report", "", f"Status: {result['status']}",
                 f"Videos: {len(successful)}/{len(records)} sampled; manifest has {total_rows} output videos.",
                 f"Measured wall time: {elapsed:.3f} s; throughput: {result['observed_videos_per_hour']:.2f} videos/h; {frames / elapsed:.2f} frames/s.",
                 "", "| Measured stage | Seconds |", "|---|---:|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in totals.items()]
        lines += [f"| Other overhead | {result['unattributed_overhead_seconds']:.3f} |", "",
                  "## Estimate (not measured full-run time)", "", json.dumps(estimate, ensure_ascii=False), "",
                  "## Conditions", "", "```json", json.dumps({"hardware": result['hardware'], "gpu": gpu, "protocol": protocol}, ensure_ascii=False, indent=2), "```", "",
                  "The first sample and cold-start costs are retained. Stages include CPU work/IO where documented.",
                  "The report itself is written after the wall-clock endpoint; report serialization and process shutdown are not timed.",
                  "A video is one model output, not one metric scalar or one multi-model task."]
        lines += ["", *list(notes)]
        (Path(output) / "timing_report.md").write_text("\n".join(lines) + "\n")
        return result


def save_results(output, records, metrics):
    output = Path(output)
    (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
    summary = []
    for model in dict.fromkeys(r["model"] for r in records):
        subset = [r for r in records if r["model"] == model]
        for metric in metrics:
            available = [r for r in subset if metric in r.get("mean", {})]
            infinite = metric == "psnr_db" and any(r["mean"].get("psnr_infinite") for r in available)
            values = [r["mean"][metric] for r in available]
            summary.append({"model": model, "metric": metric, "n": len(values), "expected": len(subset),
                            "mean": None if not values or infinite else math.fsum(values) / len(values),
                            "psnr_infinite": infinite, "ids": [r["id"] for r in available]})
    write_json(output / "summary.json", summary)
    import csv
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "n", "expected", "mean", "psnr_infinite"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in summary)
