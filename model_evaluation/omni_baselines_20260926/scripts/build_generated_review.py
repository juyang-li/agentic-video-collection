#!/usr/bin/env python3
"""Build validated, score-free VACE or Bernini human-review archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FIELDS = (
    "instruction_correctness", "logical_correctness", "content_preservation",
    "temporal_consistency", "visual_quality", "technical_quality", "overall_decision", "notes",
)
CONFIG = {
    "vace": {
        "work": ROOT / "vace", "manifest": ROOT / "vace/generation_manifest.jsonl",
        "slug": "vace14b_mask", "title": "VACE 14B Mask", "model": "Wan-AI/Wan2.1-VACE-14B",
    },
    "bernini": {
        "work": ROOT / "bernini_r14b", "manifest": ROOT / "bernini_r14b/manifests/generation_manifest.jsonl",
        "slug": "bernini_r14b", "title": "Bernini-R 14B", "model": "ByteDance/Bernini-R-Diffusers",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def full_decode(path: Path) -> None:
    run([FFMPEG, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"])


def probe(path: Path) -> dict:
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cannot decode {path}: {p.stderr[-2000:]}")
    v = re.search(r"Video:\s*([^,]+),\s*([^,\s]+).*?(\d{2,5})x(\d{2,5})", p.stderr)
    duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not v or not duration:
        raise RuntimeError(f"cannot probe {path}: {p.stderr[-2000:]}")
    return {"codec_name": v.group(1).strip().split()[0], "pix_fmt": v.group(2).split("(")[0],
            "width": int(v.group(3)), "height": int(v.group(4)),
            "duration_seconds": round(int(duration.group(1))*3600 + int(duration.group(2))*60 + float(duration.group(3)), 6)}


def proxy(src: Path, dst: Path, width: int, height: int) -> None:
    vf = (f"fps=16,trim=start_frame=0:end_frame=81,setpts=N/(16*TB),"
          f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p")
    run([FFMPEG, "-y", "-v", "error", "-nostdin", "-i", str(src), "-vf", vf,
         "-an", "-frames:v", "81", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
         "-movflags", "+faststart", str(dst)])
    full_decode(dst)
    p = probe(dst)
    if p["codec_name"] != "h264" or p["pix_fmt"] != "yuv420p" or (p["width"], p["height"]) != (width, height):
        raise RuntimeError(f"invalid review proxy {dst}: {p}")


def bind_checksums(package: Path) -> None:
    lines = [f"{sha256(p)}  {p.relative_to(package).as_posix()}" for p in sorted(package.rglob("*"))
             if p.is_file() and p.name != "SHA256SUMS"]
    (package / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=CONFIG)
    ap.add_argument("--count", type=int, choices=(24, 48), required=True)
    args = ap.parse_args(); cfg = CONFIG[args.model]
    aligned = read_csv(ROOT / "aligned_subset.csv")
    selected = [r for r in aligned if args.count == 48 or r["tier"] == "A"]
    assert len(selected) == args.count and len({r["category"] for r in selected}) == 24
    generated = [json.loads(x) for x in cfg["manifest"].read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {x["eval_id"]: x for x in generated}
    missing = [r["eval_id"] for r in selected if r["eval_id"] not in by_id or not Path(by_id[r["eval_id"]]["output_path"]).is_file()]
    if missing:
        raise RuntimeError(f"refusing incomplete package; missing {len(missing)} outputs: {missing[:8]}")
    package_name = f"omni{args.count}_{cfg['slug']}_seed42_human_review_20260926"
    package = cfg["work"] / package_name; archive = cfg["work"] / f"{package_name}.zip"
    if package.exists() or archive.exists():
        raise FileExistsError(f"refusing to overwrite {package} / {archive}")
    media = package / "media"; media.mkdir(parents=True)
    review_rows, output_manifest, inventory, items = [], [], [], []
    for position, row in enumerate(selected, 1):
        item = dict(by_id[row["eval_id"]])
        source, output = Path(row["official_source_video"]), Path(item["output_path"])
        if sha256(source) != row["official_source_sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
        actual_output_sha = sha256(output)
        if item.get("output_sha256") and item["output_sha256"] != actual_output_sha:
            raise RuntimeError(f"output hash mismatch: {output}")
        item["output_sha256"] = actual_output_sha
        raw_probe = probe(output); width, height = raw_probe["width"], raw_probe["height"]
        panels, roles = [], [("source", source), ("output", output)]
        if args.model == "vace":
            mask = Path(item["mask_path"]); overlay = Path(item["overlay_path"])
            for key, path in (("mask_sha256", mask), ("overlay_sha256", overlay)):
                if not path.is_file(): raise FileNotFoundError(path)
                actual = sha256(path)
                if item.get(key) and item[key] != actual: raise RuntimeError(f"{key} mismatch: {path}")
                item[key] = actual
            roles = [("source", source), ("mask", mask), ("mask_overlay", overlay), ("output", output)]
        labels = {"source": "SOURCE · OFFICIAL INPUT", "mask": "VACE · BINARY MASK (WHITE=REDRAW)",
                  "mask_overlay": "VACE · MASK OVERLAY", "output": f"{cfg['title'].upper()} · MODEL OUTPUT"}
        for role, original in roles:
            rel = Path("media") / f"{position:03d}_{role}.mp4"
            proxy(original, package / rel, width, height)
            pp = probe(package / rel)
            inventory.append({"eval_id": row["eval_id"], "role": role, "path": rel.as_posix(),
                              "sha256": sha256(package / rel), "bytes": (package / rel).stat().st_size,
                              "full_decode_ok": True, **pp})
            panels.append({"label": labels[role], "path": rel.as_posix()})
        base = {"eval_id": row["eval_id"], "tier": row["tier"], "aligned_index": int(row["aligned_index"]),
                "sample_index": int(row["sample_index"]), "category": row["category"],
                "instruction": row["editing_instruction"], "dataset": "OmniEdit Spatial",
                "dataset_revision": row["dataset_revision"], "model_id": item.get("model_id", cfg["model"]),
                "model_revision": item.get("model_revision", ""), "code_commit": item.get("code_commit", ""),
                "task_type": item.get("task_type", "vace" if args.model == "vace" else "v2v"),
                "generation_seed": item.get("seed", 42), "source_original_sha256": row["official_source_sha256"],
                "generated_original_sha256": actual_output_sha}
        if args.model == "vace":
            base.update(mask_original_sha256=item["mask_sha256"], mask_method=item.get("mask_method", ""),
                        mask_qc=json.dumps(item.get("mask_qc", ""), ensure_ascii=False))
        review_rows.append({**base, **{f: "" for f in FIELDS}})
        output_manifest.append(item)
        items.append({"eval_id": row["eval_id"], "tier": row["tier"], "category": row["category"],
                      "instruction": row["editing_instruction"], "task_type": base["task_type"], "panels": panels})
    write_csv(package / "review_manifest.csv", review_rows)
    with (package / "generation_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in output_manifest: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(package / "media_inventory.json", inventory)
    write_json(package / "summary.json", {"status": "complete", "review_type": "human_only_unlabeled",
        "model_id": cfg["model"], "sample_count": args.count, "category_count": 24,
        "samples_per_category": args.count // 24, "tiers": {"A": 24, "B": args.count - 24},
        "automated_judge_included": False, "aligned_subset_sha256": sha256(ROOT / "aligned_subset.csv")})
    payload = {"review_version": package_name, "title": f"{cfg['title']} · OmniEdit Spatial {args.count} · 人工审核", "items": items}
    template = (Path(__file__).with_name("human_review.html")).read_text(encoding="utf-8")
    document = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (package / "index.html").write_text(document, encoding="utf-8")
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
    subprocess.run(["node", "--check"], input=script, text=True, check=True)
    extra = "VACE 包同时展示二值 mask 与 mask overlay；白色区域为允许重绘。\n" if args.model == "vace" else "Bernini 包展示官方 source 与模型输出，并标明 v2v/mv2v 路由。\n"
    (package / "README.md").write_text(f"# {cfg['title']} · OmniEdit Spatial {args.count} 人工审核包\n\n"
        "完整解压后用 Chrome / Edge / Safari 打开 `index.html`；本包完全离线，不调用 API。\n\n" + extra +
        "审核项初始均为空。1=通过，0=不通过，2=存疑；请定期导出 CSV 或 JSON。\n", encoding="utf-8")
    bind_checksums(package)
    # Browser validation is run separately, then this script can be re-run only through finalize_review.py.
    print(json.dumps({"status": "built_pending_browser_validation", "package": str(package), "samples": args.count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
