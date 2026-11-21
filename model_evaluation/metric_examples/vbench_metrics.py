"""Official VBench custom_input example. Full raw flicker; no static filtering."""
import time
PROCESS_START = time.perf_counter()
import argparse
import importlib
import math
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
from common import (RunTimer, common_arguments, digest, fresh_output, lossless_video,
                    probe, read_manifest, save_results, select_frames, sync, targets, write_json)

REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
DIMS = ["subject_consistency", "background_consistency", "aesthetic_quality",
        "imaging_quality", "temporal_flickering", "motion_smoothness"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--vbench-root", type=Path, required=True, help="Official source checkout at the pinned revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dimensions", default=",".join(DIMS))
    parser.add_argument("--local-weights", action="store_true", help="Use official local=True weight layout; otherwise allow official downloads")
    args = parser.parse_args()
    dims = args.dimensions.split(",")
    if not dims or len(set(dims)) != len(dims) or not set(dims) <= set(DIMS):
        parser.error("Choose a unique comma-separated subset of the six documented dimensions")
    times = targets(args)
    rows = read_manifest(args.manifest)
    selected = rows[:args.limit] if args.limit else rows
    output = fresh_output(args.output_dir).resolve()
    timer = RunTimer(PROCESS_START, time.perf_counter() - PROCESS_START)
    vendor = args.vbench_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected VBench {REVISION}, got {revision}")
    sys.path.insert(0, str(vendor))
    with timer.stage("official_import_and_device_setup"):
        import torch
        from vbench import VBench
        torch.set_num_threads(args.torch_threads)
        import cv2
        cv2.setNumThreads(args.torch_threads)
        if Path(importlib.import_module("vbench").__file__).resolve().parent != vendor / "vbench":
            raise ValueError("A different installed VBench was imported")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if any(d != "temporal_flickering" for d in dims):
            if not args.device.startswith("cuda") or not torch.cuda.is_available():
                raise ValueError("The five quality dimensions require a working CUDA device in this example")
            torch.cuda.set_device(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
    signatures = []
    with timer.stage("manifest_header_probe"):
        for row in rows:
            try:
                signatures.append(str(probe(row["prediction"])))
            except Exception:
                pass
    records, lookup = [], {}
    media = output / "scoring_media"
    media.mkdir()
    for index, row in enumerate(selected):
        record = {"id": row["id"], "model": row["model"], "status": "failed", "mean": {}, "raw_score": {}}
        try:
            with timer.stage("prepare_scoring_media", id=row["id"], model=row["model"]):
                frames, alignment = select_frames(row["prediction"], times, "prediction", args.decode_threads)
                path = media / f"video_{index:08d}.mp4"
                receipt = lossless_video(path, frames, args.fps)
                record.update(num_frames=len(times), native_wh=alignment["native_wh"], alignment=alignment,
                              original_sha256=digest(row["prediction"]), scoring_media=receipt)
                lookup[path.stem] = record
                del frames
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    bench = VBench(torch.device(args.device), str(vendor / "vbench/VBench_full_info.json"), str(output / "official_results"))
    for dim in dims:
        try:
            gpu = dim != "temporal_flickering"
            if gpu:
                sync(args.device)
            with timer.stage("dimension_api_wall", dimension=dim, includes_model_load_decode_compute=True):
                if not gpu:
                    from vbench.temporal_flickering import cal_score
                    values = [{"video_path": str(path), "video_results": float(cal_score(str(path)))} for path in sorted(media.glob("*.mp4"))]
                elif lookup:
                    name = dim
                    bench.evaluate(str(media), name, dimension_list=[dim], local=args.local_weights,
                                   mode="custom_input", imaging_quality_preprocessing_mode="longer")
                    import json
                    values = json.loads((output / f"official_results/{name}_eval_results.json").read_text())[dim][1]
                else:
                    values = []
                if gpu:
                    sync(args.device)
            seen = set()
            for value in values:
                key = Path(value["video_path"]).stem
                raw = float(value["video_results"])
                if key not in lookup or key in seen or not math.isfinite(raw):
                    raise ValueError("Official response has unknown/duplicate video or non-finite score")
                seen.add(key)
                lookup[key]["raw_score"][dim] = raw
                lookup[key]["mean"][dim] = raw / 100 if dim == "imaging_quality" else raw
            for key in set(lookup) - seen:
                lookup[key].setdefault("errors", []).append(f"Official evaluator omitted {dim}")
        except Exception as exc:
            for record in lookup.values():
                record.setdefault("errors", []).append(f"{dim}: {type(exc).__name__}: {exc}")
    for record in records:
        record["status"] = "complete" if set(record["mean"]) == set(dims) else "failed"
    with timer.stage("result_write"):
        save_results(output, records, dims)
        write_json(output / "implementation.json", {"vbench_revision": revision, "mode": "custom_input",
                   "imaging_quality_preprocessing_mode": "longer", "temporal_static_filter": "not_requested",
                   "dimension_source_sha256": {d: digest(vendor / f"vbench/{d}.py") for d in dims}})
    report = timer.finish(output, records, len(rows),
                          {"suite": "VBench custom_input", "start": args.start, "end": args.end,
                           "fps": args.fps, "endpoint": args.endpoint, "samples_per_video": len(times),
                           "dimensions": dims, "canvas": "native", "static_filter": "not_requested",
                           "local_weights": args.local_weights, "decode_threads": args.decode_threads, "seed": args.seed,
                           "torch_threads": args.torch_threads, "user_declared_cache_state": args.cache_state},
                          signatures, args.device if any(d != "temporal_flickering" for d in dims) else "cpu",
                          ["Dimension API wall times include loading, decoding and computation; internal substages are not separately measured.",
                           "Official model downloads may occur without --local-weights; they are retained in the first-run API wall time.",
                           "Temporal Flickering is the full raw score, affected by normal motion; no static filtering or Total Score."])
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
