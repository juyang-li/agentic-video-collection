"""Run paired GT metrics on arbitrary JSONL manifests; see the bundled guide."""
import time
PROCESS_START = time.perf_counter()
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
from common import (RunTimer, common_arguments, crop_box, digest, fresh_output,
                    matching_aspect, probe, read_manifest, save_results, select_frames, sync, targets)


def frame_metrics(pair):
    from skimage.metrics import structural_similarity
    a, b = [x.astype(np.float32) / 255 for x in pair]
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return {"ssim": float(structural_similarity(a, b, data_range=1, channel_axis=-1,
                                               gaussian_weights=True, sigma=1.5, win_size=11,
                                               use_sample_covariance=False)),
            "psnr_db": -10 * math.log10(mse) if mse else None, "psnr_infinite": mse == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--metrics", default="ssim,psnr,lpips")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--size", default="auto720", help="auto720 or WIDTHxHEIGHT")
    parser.add_argument("--prediction-padding", choices=["none", "contain"], default="none")
    args = parser.parse_args()
    metrics = args.metrics.split(",")
    if len(set(metrics)) != len(metrics) or not set(metrics) <= {"ssim", "psnr", "lpips"} or not metrics:
        parser.error("metrics must be a unique comma-separated subset of ssim,psnr,lpips")
    if min(args.cpu_workers, args.lpips_batch_size) < 1:
        parser.error("worker count and batch size must be positive")
    times = targets(args)
    rows = read_manifest(args.manifest, require_gt=True)
    output = fresh_output(args.output_dir)
    timer = RunTimer(PROCESS_START, time.perf_counter() - PROCESS_START)
    selected = rows[:args.limit] if args.limit else rows
    metadata, signatures = {}, []
    with timer.stage("preflight"):
        for row in rows:
            try:
                gt, pred = probe(row["gt"]), probe(row["prediction"])
                gb = crop_box(gt["width"], gt["height"], row.get("gt_crop"))
                if row.get("prediction_crop") is not None and args.prediction_padding != "none":
                    raise ValueError("Use prediction_crop OR contain padding, not both")
                pb = crop_box(pred["width"], pred["height"], row.get("prediction_crop"),
                              gb[2:] if args.prediction_padding == "contain" else None)
                size = ([1280, 720] if gb[2] >= gb[3] else [720, 1280]) if args.size == "auto720" else [int(x) for x in args.size.split("x")]
                if len(size) != 2 or min(size) < 11 or not matching_aspect(gb[2:], pb[2:]) or not matching_aspect(gb[2:], size):
                    raise ValueError("Aspect/size mismatch: declare the known padding/crop and an aspect-preserving size")
                metadata[row["id"], row["model"]] = (gb, pb, size)
                signatures.append(str((gt, pred, size, gb, pb)))
            except Exception as exc:
                metadata[row["id"], row["model"]] = str(exc)
    network, torch = None, None
    if "lpips" in metrics:
        with timer.stage("lpips_initialization_including_possible_download"):
            import torch
            import lpips
            torch.set_num_threads(args.torch_threads)
            torch.manual_seed(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if args.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; install matching PyTorch, or choose CPU metrics/device")
                torch.cuda.set_device(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            network = lpips.LPIPS(net="alex", version="0.1").to(args.device).float().eval()
            sync(args.device)
    records = []
    for row in selected:
        record = {"id": row["id"], "model": row["model"], "status": "failed", "mean": {}, "frames": []}
        try:
            config = metadata[row["id"], row["model"]]
            if isinstance(config, str):
                raise ValueError(config)
            gb, pb, size = config
            with timer.stage("decode_align_and_hash", id=row["id"], model=row["model"]):
                gt, ga = select_frames(row["gt"], times, "gt", args.decode_threads, gb, size)
                pred, pa = select_frames(row["prediction"], times, "prediction", args.decode_threads, pb, size)
                record.update(num_frames=len(times), effective_wh=size, alignment={"gt": ga, "prediction": pa},
                              gt_sha256=digest(row["gt"]), prediction_sha256=digest(row["prediction"]))
            values = [{"frame": i} for i in range(len(times))]
            record["frames"] = values
            if set(metrics) & {"ssim", "psnr"}:
                with timer.stage("ssim_psnr_joint", id=row["id"], model=row["model"]):
                    with ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
                        basic = list(pool.map(frame_metrics, zip(gt, pred)))
                    for out, basic_row in zip(values, basic):
                        out.update({k: v for k, v in basic_row.items() if k == "ssim" and "ssim" in metrics or k.startswith("psnr") and "psnr" in metrics})
                    if "ssim" in metrics:
                        record["mean"]["ssim"] = math.fsum(v["ssim"] for v in basic) / len(basic)
                    if "psnr" in metrics:
                        inf = any(v["psnr_infinite"] for v in basic)
                        record["mean"].update(psnr_db=None if inf else math.fsum(v["psnr_db"] for v in basic) / len(basic), psnr_infinite=inf)
            if network is not None:
                sync(args.device)
                with timer.stage("lpips_with_transfer", id=row["id"], model=row["model"]):
                    pos, batch_size = 0, args.lpips_batch_size
                    distances = []
                    while pos < len(times):
                        x = y = None
                        try:
                            x = torch.from_numpy(np.stack(pred[pos:pos + batch_size])).permute(0, 3, 1, 2).to(args.device).float() / 127.5 - 1
                            y = torch.from_numpy(np.stack(gt[pos:pos + batch_size])).permute(0, 3, 1, 2).to(args.device).float() / 127.5 - 1
                            with torch.inference_mode():
                                result = network(x, y).flatten().cpu().tolist()
                            if not all(math.isfinite(v) for v in result):
                                raise ValueError("Non-finite LPIPS result")
                            distances.extend(result)
                            pos += len(result)
                        except torch.cuda.OutOfMemoryError:
                            if batch_size == 1:
                                raise
                            del x, y
                            batch_size = max(1, batch_size // 2)
                            torch.cuda.empty_cache()
                    sync(args.device)
                    for out, distance in zip(values, distances):
                        out["lpips"] = distance
                    record["mean"]["lpips"] = math.fsum(distances) / len(distances)
                    record["actual_lpips_batch_size"] = batch_size
            record["status"] = "complete"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        print(row["id"], row["model"], record["status"], flush=True)
    names = ["psnr_db" if x == "psnr" else x for x in metrics]
    with timer.stage("result_write"):
        save_results(output, records, names)
    actual_device = args.device if network is not None else "cpu"
    report = timer.finish(output, records, len(rows),
                          {"suite": "GT", "start": args.start, "end": args.end, "fps": args.fps,
                           "endpoint": args.endpoint, "samples_per_video": len(times), "size": args.size,
                           "prediction_padding": args.prediction_padding, "cpu_workers": args.cpu_workers,
                           "decode_threads": args.decode_threads, "lpips_batch_size": args.lpips_batch_size,
                           "metrics": metrics, "seed": args.seed, "tf32": False, "amp": False,
                           "torch_threads": args.torch_threads, "user_declared_cache_state": args.cache_state},
                          signatures, actual_device,
                          ["SSIM and PSNR use a joint traversal; no independent SSIM-only/PSNR-only timing is claimed.",
                           "LPIPS initialization may include official weight downloads; this first-run cost is retained."])
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
