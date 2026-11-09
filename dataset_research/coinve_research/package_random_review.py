#!/usr/bin/env python3
"""Prepare lossless browser media where possible, and package CoinVE review."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from build_random_review import ROOT, OUTPUT, VERSION, digest, read_csv, write_json

FFMPEG = "/scratch/prj0000000275/jcy/.conda/envs/vivid/bin/ffmpeg"
FFPROBE = "/scratch/prj0000000275/jcy/.conda/envs/vivid/bin/ffprobe"
BUNDLE = OUTPUT / VERSION


def probe(path):
    result = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration,start_time",
        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    v = json.loads(result.stdout)["streams"][0]
    return dict(codec=v["codec_name"], pix_fmt=v["pix_fmt"], width=v["width"], height=v["height"],
                fps=float(Fraction(v["avg_frame_rate"])), fps_rational=v["avg_frame_rate"],
                frames=int(v["nb_frames"]), duration=float(v["duration"]),
                start_time=float(v.get("start_time", 0)))


def faststart(path):
    with path.open("rb") as stream:
        while True:
            header = stream.read(8)
            if len(header) != 8:
                return False
            size, kind = struct.unpack(">I4s", header)
            used = 8
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
                used = 16
            if kind == b"moov":
                return True
            if kind == b"mdat" or size < used:
                return False
            stream.seek(size-used, 1)


def prepare_video(job):
    row, role, cache = job
    original = cache / f"{row['sample_id']}_{role}.mp4"
    target = BUNDLE / "media" / original.name
    receipt = target.with_suffix(".json")
    source_hash = digest(original)
    source = probe(original)
    if receipt.exists() and target.exists():
        previous = json.loads(receipt.read_text())
        if previous["original_sha256"] == source_hash and previous["review_sha256"] == digest(target):
            return previous
    mode = "lossless_video_stream_remux" if (source["codec"], source["pix_fmt"]) == ("h264", "yuv420p") else "h264_crf18_full_resolution"
    temporary = target.with_suffix(".part.mp4")
    command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2", "-i", str(original),
               "-map", "0:v:0", "-an"]
    if mode == "lossless_video_stream_remux":
        command += ["-c:v", "copy"]
    else:
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", "-fps_mode", "passthrough"]
    subprocess.run(command + ["-movflags", "+faststart", str(temporary)], check=True)
    out = probe(temporary)
    assert out["codec"] == "h264" and out["pix_fmt"] == "yuv420p" and faststart(temporary)
    for key in ("width", "height", "frames", "fps_rational"):
        assert out[key] == source[key], (original.name, key, source[key], out[key])
    assert abs(out["duration"]-source["duration"]) < 1/source["fps"]
    decoded = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-xerror", "-nostdin",
                 "-threads", "2", "-i", str(temporary), "-map", "0:v:0", "-f", "null", "-"],
                 capture_output=True, text=True, check=True)
    if decoded.stderr.strip():
        raise RuntimeError(f"decode diagnostics for {temporary.name}")
    temporary.replace(target)
    record = dict(sample_id=row["sample_id"], role=role, original_sha256=source_hash,
                  original_info=source, review_info=out, review_media=f"media/{target.name}",
                  review_sha256=digest(target), review_bytes=target.stat().st_size,
                  conversion=mode, full_decode="passed", faststart=True)
    write_json(receipt, record)
    print(f"prepared {row['sample_id']} {role} {mode}", flush=True)
    return record


def build(cache, workers):
    rows = read_csv(OUTPUT / "sample_manifest.csv")
    assert len(rows) == 50
    BUNDLE.mkdir(exist_ok=True)
    (BUNDLE / "media").mkdir(exist_ok=True)
    missing = [f"{r['sample_id']}_{role}.mp4" for r in rows for role in ("source", "target")
               if not (cache / f"{r['sample_id']}_{role}.mp4").is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} frozen role files; no replacements: {missing}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        media = list(executor.map(prepare_video, [(r, role, cache) for r in rows for role in ("source", "target")]))
    by_role = {(r["sample_id"], r["role"]): r for r in media}
    items, review_rows = [], []
    for r in rows:
        item = dict(sample_id=r["sample_id"], branch=r["branch"], instructions=json.loads(r["instructions_json"]),
                    operations=json.loads(r["operations_json"]), objects=json.loads(r["objects_json"]))
        entry = dict(r)
        for role in ("source", "target"):
            m = by_role[(r["sample_id"], role)]
            item[role+"_media"] = m["review_media"]
            item[role+"_info"] = m["review_info"]
            entry[role+"_media"] = m["review_media"]
            entry[role+"_original_sha256"] = m["original_sha256"]
            entry[role+"_review_sha256"] = m["review_sha256"]
            entry[role+"_duration"] = m["review_info"]["duration"]
        items.append(item)
        review_rows.append(entry)
    summary = json.loads((OUTPUT / "sampling_summary.json").read_text())
    payload = dict(survey_version=VERSION, repo_id=summary["repo_id"], revision=summary["revision"], items=items)
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    font = base64.b64encode((OUTPUT / "review_font.woff").read_bytes()).decode()
    document = (ROOT / "random_review_template.html").read_text().replace("__SURVEY_DATA__", encoded).replace("__CJK_FONT__", font)
    (BUNDLE / "index.html").write_text(document)
    for name, records in (("review_manifest.csv", review_rows), ("audit.csv", [dict(sample_id=r["sample_id"],
                branch=r["branch"], human_decision=1, reviewed=0, notes="") for r in rows])):
        dest = BUNDLE / name
        if name == "audit.csv" and dest.exists():
            continue
        with dest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(BUNDLE / "media_inventory.json", media)
    shutil.copyfile(OUTPUT / "sample_manifest.csv", BUNDLE / "sample_manifest.csv")
    shutil.copyfile(OUTPUT / "font_LICENSE.txt", BUNDLE / "font_LICENSE.txt")
    summary.update(complete_pair_count=50, full_decode_role_count=100,
        review_media_bytes=sum(m["review_bytes"] for m in media),
        duration_mismatch_pairs=sum(abs(i["source_info"]["duration"]-i["target_info"]["duration"])>.05 for i in items),
        source_target_fps_mismatch_pairs=sum(i["source_info"]["fps_rational"]!=i["target_info"]["fps_rational"] for i in items),
        browser_validation="pending", model_judge_calls=0,
        quality_results="awaiting human review; defaults are not confirmed labels")
    write_json(BUNDLE / "summary.json", summary)
    (BUNDLE / "README.md").write_text('''# CoinVE 50 对随机审核

完整解压 ZIP，用 Chrome / Edge 打开 `index.html`，无需服务器或联网。

- 黑色单 pair 页面；source / target 同屏，所有子指令完整保留。
- 空格同步播放，左右键翻页；可以跳过。1 接受、0 拒绝、2 存疑，备注可写具体失败的子指令。
- CSV / JSON 均可导出和重新导入；结果自动保存到当前浏览器。换设备或移动目录前先导出。
- 审核项预填 1，但 `reviewed=0` 表示未审核；仅点击结论后才记为人工已审核。计算通过率必须限定 `reviewed=1`，不能把预填值算作人工通过。
- 两侧完整分辨率、帧率和时长均保留。同步播放共同时间区间，不拉伸视频；时长不同时可独立打开完整片段检查尾段。
- 优先无损重封装 H.264；不兼容时只转码审核副本，细节见 `media_inventory.json`。审核包无音轨，仅供视觉编辑审核。

抽样：固定官方 revision、metadata SHA256 和 seed 20260906；从不含 Stylize 的 145,483 条中排除此前看过的 10 条，在剩余 145,473 条中无放回均匀随机抽 50 条。未按人物、画质、时长或编辑质量筛选，未按结果补抽。2/3/4/5 个任务分别为 23/23/3/1 对。

`sample_manifest.csv` 保存固定抽样顺序与原始来源，`review_manifest.csv` 连接审核媒体，`summary.json` 保存抽样和技术统计。所有模型质量筛选均未运行，等待人工判断是否适合直接入池。
''')
    scripts = re.findall(r"<script>(.*?)</script>", document, re.S)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js") as script:
        script.write("\n".join(scripts)); script.flush()
        subprocess.run(["node", "--check", script.name], check=True)
    print(f"ready for browser acceptance: {BUNDLE}", flush=True)


def package():
    report = json.loads((BUNDLE / "browser_validation.json").read_text())
    assert report["status"] == "passed" and report["sample_count"] == 50
    media = json.loads((BUNDLE / "media_inventory.json").read_text())
    assert len(media) == 100
    for m in media:
        assert digest(BUNDLE / m["review_media"]) == m["review_sha256"]
    summary = json.loads((BUNDLE / "summary.json").read_text())
    summary["browser_validation"] = "passed"
    write_json(BUNDLE / "summary.json", summary)
    files = [p for p in BUNDLE.rglob("*") if p.is_file() and not (p.parent.name == "media" and p.suffix == ".json")]
    archive = OUTPUT / (VERSION+".zip")
    temporary = archive.with_suffix(".part.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in sorted(files):
            z.write(path, Path(VERSION)/path.relative_to(BUNDLE))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert {Path(n).parts[0] for n in names} == {VERSION}
        assert sum(n.endswith(".mp4") for n in names) == 100
        for m in media:
            assert str(Path(VERSION)/m["review_media"]) in names
    temporary.replace(archive)
    receipt = dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest(archive),
                   pair_count=50, media_count=100, zip_crc="passed", single_root_directory=True,
                   browser_validation="passed")
    write_json(archive.with_suffix(".sha256.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "package"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/coinve_random50_20260906"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "build":
        build(args.cache, args.workers)
    else:
        package()


if __name__ == "__main__":
    main()
