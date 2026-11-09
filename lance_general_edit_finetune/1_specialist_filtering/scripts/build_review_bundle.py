#!/usr/bin/env python3
"""Transcode unique media and build the self-contained offline review HTML."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/survey_manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/review_bundle"
TEMPLATE = ROOT / "compact_pair_reviewer_template.html"
DURATION_RE = re.compile(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)")


def find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError(
            "ffmpeg was not found; pass --ffmpeg or install imageio-ffmpeg"
        ) from error


def proxy_name(video_id: str) -> str:
    return hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:20] + ".mp4"


def probe_duration(ffmpeg: str, source: Path) -> float:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(source)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    match = DURATION_RE.search(result.stderr)
    if match is None:
        raise RuntimeError(f"cannot find a valid video duration: {source}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def transcode(
    ffmpeg: str, source: Path, output: Path, max_long_edge: int, crf: int, rebuild: bool
) -> None:
    if output.is_file() and not rebuild:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.mp4")
    scale = (
        f"scale='if(gt(iw,ih),min({max_long_edge},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({max_long_edge},ih))'"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            scale,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg")
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE,
        help="HTML template; defaults to the compact one-decision pair reviewer",
    )
    parser.add_argument("--max-long-edge", type=int, default=720)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--expected-size", type=int, default=400)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != args.expected_size:
        raise ValueError(f"expected {args.expected_size} manifest rows, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("manifest contains duplicate sample_id")

    video_sources: dict[str, Path] = {}
    for row in rows:
        for role in ("source", "target"):
            video_id = row[f"{role}_video_id"]
            path = (ROOT / row[f"{role}_local"]).resolve()
            if ROOT.resolve() not in path.parents:
                raise ValueError(f"media path escapes specialist root: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"missing {role} media for {row['sample_id']}: {path}")
            existing = video_sources.setdefault(video_id, path)
            if existing != path:
                raise ValueError(f"video_id maps to multiple paths: {video_id}")

    ffmpeg = find_ffmpeg(args.ffmpeg)
    media_dir = args.output / "media"
    inventory: list[dict[str, object]] = []
    for index, (video_id, source) in enumerate(video_sources.items(), 1):
        destination = media_dir / proxy_name(video_id)
        duration = probe_duration(ffmpeg, source)
        if duration < 3.0:
            raise ValueError(f"video is shorter than 3.0 seconds: {video_id} duration={duration}")
        transcode(
            ffmpeg,
            source,
            destination,
            args.max_long_edge,
            args.crf,
            args.rebuild,
        )
        inventory.append(
            {
                "video_id": video_id,
                "source_path": str(source.relative_to(ROOT)),
                "duration_seconds": round(duration, 3),
                "proxy_path": str(destination.relative_to(args.output)),
            }
        )
        if index % 25 == 0 or index == len(video_sources):
            print(f"proxies={index}/{len(video_sources)}", flush=True)

    items: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["source_media"] = "media/" + proxy_name(row["source_video_id"])
        item["target_media"] = "media/" + proxy_name(row["target_video_id"])
        items.append(item)
    payload = {
        "survey_version": rows[0]["survey_version"] if rows else "",
        "items": items,
        "created_from": args.manifest.name,
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = args.template.read_text(encoding="utf-8").replace("__SURVEY_DATA__", encoded)
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "index.part.html"
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(args.output / "index.html")
    inventory_temp = args.output / "media_inventory.part.csv"
    with inventory_temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    inventory_temp.replace(args.output / "media_inventory.csv")
    print(
        f"bundle={args.output / 'index.html'} pairs={len(rows)} "
        f"unique_videos={len(video_sources)}"
    )


if __name__ == "__main__":
    main()
