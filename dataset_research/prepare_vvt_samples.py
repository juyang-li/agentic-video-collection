#!/usr/bin/env python3
"""Build compact H.264 review videos and inventories for ViViD or Eevee."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
STATIC_FFMPEG = Path(
    "/home/users/astar/cfar/stuchengyou/Flow-Factory/.venv/lib/python3.10/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


CONFIG = {
    "vivid": {
        "root": ROOT / "vivid_research",
        "roles": (("agnostic_member", "MASKED INPUT", "video"),
                  ("garment_member", "GARMENT", "image"),
                  ("target_member", "TARGET", "video")),
        "panel": (400, 720),
    },
    "eevee": {
        "root": ROOT / "eevee_research",
        "roles": (("agnostic_member", "MASKED INPUT", "video"),
                  ("garment_member", "GARMENT", "image"),
                  ("target_member", "FULL SHOT", "video"),
                  ("closeup_member", "CLOSE-UP", "video")),
        "panel": (320, 640),
    },
}


def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if STATIC_FFMPEG.is_file():
        return str(STATIC_FFMPEG)
    raise FileNotFoundError("ffmpeg was not found")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    text = result.stderr
    duration_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text)
    video_match = re.search(
        r"Video:\s*([^,]+).*?,\s*(\d+)x(\d+).*?,\s*([\d.]+) fps", text
    )
    frame_matches = re.findall(r"frame=\s*(\d+)", text)
    if not duration_match or not video_match or not frame_matches:
        raise RuntimeError(f"could not inspect video: {path}\n{text[-1200:]}")
    hours, minutes, seconds = duration_match.groups()
    return {
        "decode_ok": result.returncode == 0,
        "codec": video_match.group(1).strip(),
        "width": int(video_match.group(2)),
        "height": int(video_match.group(3)),
        "fps": float(video_match.group(4)),
        "frame_count": int(frame_matches[-1]),
        "duration_seconds": round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3),
    }


def write_comparison(
    roles: tuple[tuple[str, str, str], ...],
    paths: list[Path],
    output: Path,
    panel: tuple[int, int],
    header: Path,
    rebuild: bool,
) -> None:
    if output.exists() and not rebuild:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y"]
    for (_, _, kind), path in zip(roles, paths):
        if kind == "image":
            command.extend(["-loop", "1", "-framerate", "25"])
        command.extend(["-i", str(path)])
    command.extend(["-loop", "1", "-framerate", "25", "-i", str(header)])
    width, height = panel
    filters = []
    labels = []
    for index, (_, _, _) in enumerate(roles):
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:white[p{index}]"
        )
        labels.append(f"[p{index}]")
    header_index = len(roles)
    filters.append(f"{''.join(labels)}hstack=inputs={len(labels)}:shortest=1[row]")
    filters.append(f"[{header_index}:v][row]vstack=inputs=2:shortest=1[out]")
    temporary = output.with_name(output.stem + ".part.mp4")
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    temporary.replace(output)


def write_header(roles: tuple[tuple[str, str, str], ...], panel: tuple[int, int], path: Path) -> None:
    width, _ = panel
    height = 50
    image = Image.new("RGB", (width * len(roles), height), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    for index, (_, label, _) in enumerate(roles):
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((index * width + (width - text_width) // 2, 12), label, fill="black", font=font)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(CONFIG))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reset-audit", action="store_true")
    args = parser.parse_args()

    config = CONFIG[args.dataset]
    dataset_root: Path = config["root"]
    roles = config["roles"]
    raw_root = dataset_root / "raw_sample"
    comparison_root = dataset_root / "examples" / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    header = dataset_root / f".{args.dataset}_header.png"
    write_header(roles, config["panel"], header)
    manifest = list(csv.DictReader((dataset_root / "random_pairs.csv").open()))
    manifest.sort(key=lambda row: int(row["sample_rank"]))
    inventory = []
    for item in manifest:
        paths = [raw_root / item[field] for field, _, _ in roles]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        target_path = raw_root / item["target_member"]
        short_hash = sha256_file(target_path)[:5]
        comparison = comparison_root / f"{short_hash}-{item['sample_id']}.mp4"
        write_comparison(roles, paths, comparison, config["panel"], header, args.rebuild)
        caption = ""
        if args.dataset == "eevee":
            caption = (raw_root / item["caption_member"]).read_text(encoding="utf-8").strip()
        comparison.with_suffix(".txt").write_text(
            f"task: {item['task']}\n"
            f"instruction: {item['instruction']}\n"
            f"sample: {item['category']}/{item.get('sample_dir', item.get('video', ''))}\n"
            f"caption: {caption}\n",
            encoding="utf-8",
        )
        row: dict[str, object] = {
            "hash": short_hash,
            "task": item["task"],
            "sample_id": item["sample_id"],
            "population_index": item["population_index"],
            "split": item["split"],
            "category": item["category"],
            "comparison_file": str(comparison.relative_to(dataset_root)),
        }
        for (field, _, kind), path in zip(roles, paths):
            role = field.removesuffix("_member")
            row[f"{role}_file"] = str(path.relative_to(dataset_root))
            row[f"{role}_sha256"] = sha256_file(path)
            if kind == "video":
                for key, value in inspect_video(path).items():
                    row[f"{role}_{key}"] = value
        inventory.append(row)

    inventory_path = dataset_root / f"{args.dataset}_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    audit_path = dataset_root / f"{args.dataset}_audit.csv"
    if args.reset_audit or not audit_path.exists():
        fields = [
            "hash", "task", "person_relevance", "instruction_correctness",
            "logical_correctness", "content_preservation", "temporal_consistency",
            "visual_quality", "technical_quality", "notes",
        ]
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in inventory:
                writer.writerow({"hash": row["hash"], "task": row["task"]})
    header.unlink(missing_ok=True)
    print(f"dataset={args.dataset} inventory_rows={len(inventory)}")


if __name__ == "__main__":
    main()
