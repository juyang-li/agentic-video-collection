#!/usr/bin/env python3
"""Stream OpenVid-1k and build a deterministic, stratified T2V review bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import random
import shutil
import subprocess
import tarfile
from pathlib import Path

import requests


DEFAULT_URL = (
    "https://huggingface.co/datasets/finetrainers/OpenVid-1k-split/"
    "resolve/main/dataset_0000.tar"
)
QUOTAS = {"human": 15, "vehicle": 10, "animal": 5, "scene": 10, "object": 10}
TEXT_FIELDS = {
    "caption",
    "aesthetic_score",
    "motion_score",
    "temporal_consistency_score",
    "camera_motion",
    "frame",
    "fps",
    "seconds",
    "part_id",
}


class HTTPRangeReader(io.RawIOBase):
    """Small seekable HTTP reader for an uncompressed tar hosted with Range support."""

    def __init__(self, url: str, block_size: int = 128 * 1024) -> None:
        super().__init__()
        self.session = requests.Session()
        first = self.session.get(
            url, headers={"Range": "bytes=0-0"}, timeout=(30, 120)
        )
        first.raise_for_status()
        if first.status_code != 206 or "/" not in first.headers.get("Content-Range", ""):
            raise RuntimeError("OpenVid host did not honor HTTP Range")
        self.url = first.url
        self.size = int(first.headers["Content-Range"].rsplit("/", 1)[1])
        self.block_size = block_size
        self.position = 0
        self.cache_start = 0
        self.cache = bytes(first.content)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _fetch(self, start: int, minimum: int) -> None:
        end = min(self.size - 1, start + max(self.block_size, minimum) - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=(30, 120),
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"Range request returned HTTP {response.status_code}")
        self.cache_start = start
        self.cache = response.content

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fetch(self.position, min(remaining, self.block_size))
                cache_end = self.cache_start + len(self.cache)
            take = min(remaining, cache_end - self.position)
            local = self.position - self.cache_start
            chunks.append(self.cache[local : local + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
        super().close()


def category(caption: str) -> str:
    text = f" {caption.lower()} "
    keywords = {
        "animal": (
            " dog ", " dogs ", " puppy ", " cat ", " cats ", " kitten ",
            " horse ", " horses ", " bird ", " birds ", " duck ", " cow ",
            " sheep ", " animal ", " animals ", " lion ", " tiger ", " elephant ",
            " rabbit ", " fish ", " bear ", " monkey ", " deer ", " fox ",
        ),
        "vehicle": (
            " car ", " truck ", " bus ", " train ", " motorcycle ", " bicycle ",
            " airplane ", " aircraft ", " boat ", " ship ", " vehicle ",
        ),
        "human": (
            " man ", " woman ", " person ", " people ", " boy ", " girl ",
            " child ", " children ", " couple ", " crowd ", " athlete ",
        ),
        "scene": (
            " landscape ", " mountain ", " ocean ", " beach ", " forest ", " river ",
            " waterfall ", " sunset ", " sunrise ", " cityscape ", " street ", " sky ",
        ),
    }
    for name in ("animal", "vehicle", "human", "scene"):
        if any(word in text for word in keywords[name]):
            return name
    return "object"


def eligible(row: dict[str, str]) -> bool:
    caption = row.get("caption", "").lower()
    blocked = (
        "animation", "animated", "cartoon", "comic book", "video game", "split-screen",
        "news ticker", "news segment", "subscribe button", "watermark",
    )
    try:
        seconds = float(row["seconds"])
        temporal = float(row["temporal_consistency_score"])
    except (KeyError, ValueError):
        return False
    return 3.0 <= seconds <= 10.0 and temporal >= 0.995 and not any(x in caption for x in blocked)


def safe_key(name: str) -> str:
    return name.replace("/", "_").replace("..", "_")


def stream_sample(url: str, output: Path, seed: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    rngs = {name: random.Random(f"{seed}:{name}") for name in QUOTAS}
    seen = {name: 0 for name in QUOTAS}
    kept: dict[str, list[dict[str, str]]] = {name: [] for name in QUOTAS}
    current_key = ""
    current: dict[str, str] = {}

    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|*") as archive:
            for member in archive:
                if not member.isfile() or "." not in member.name:
                    continue
                key, ext = member.name.rsplit(".", 1)
                if key != current_key:
                    current_key, current = key, {}
                source = archive.extractfile(member)
                if source is None:
                    continue
                if ext in TEXT_FIELDS:
                    current[ext] = source.read().decode("utf-8", errors="replace").strip()
                    continue
                if ext != "video":
                    continue
                if not eligible(current):
                    continue

                kind = category(current.get("caption", ""))
                seen[kind] += 1
                quota = QUOTAS[kind]
                slot = len(kept[kind])
                replace_index = slot if slot < quota else rngs[kind].randrange(seen[kind])
                if replace_index >= quota:
                    continue

                record = dict(current)
                record["source_key"] = current_key
                record["category"] = kind
                filename = f"{kind}_{replace_index + 1:02d}_{safe_key(Path(current_key).name)}.mp4"
                destination = media / filename
                temporary = destination.with_suffix(".part")
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary.replace(destination)
                record["video_path"] = f"media/{filename}"

                if slot < quota:
                    kept[kind].append(record)
                else:
                    old = output / kept[kind][replace_index]["video_path"]
                    if old != destination:
                        old.unlink(missing_ok=True)
                    kept[kind][replace_index] = record

    missing = {name: QUOTAS[name] - len(kept[name]) for name in QUOTAS if len(kept[name]) < QUOTAS[name]}
    if missing:
        raise RuntimeError(f"not enough eligible samples: {missing}; seen={seen}")
    rows = [row for name in QUOTAS for row in kept[name]]
    return rows, seen


def range_sample(
    url: str, output: Path, seed: int, block_size: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Scan only tar metadata via Range, then fetch exactly the selected videos."""
    rngs = {name: random.Random(f"{seed}:{name}") for name in QUOTAS}
    seen = {name: 0 for name in QUOTAS}
    kept: dict[str, list[dict[str, str]]] = {name: [] for name in QUOTAS}
    current_key = ""
    current: dict[str, str] = {}
    reader = HTTPRangeReader(url, block_size=block_size)
    try:
        with tarfile.open(fileobj=reader, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or "." not in member.name:
                    continue
                key, ext = member.name.rsplit(".", 1)
                if key != current_key:
                    current_key, current = key, {}
                if ext in TEXT_FIELDS:
                    source = archive.extractfile(member)
                    if source is not None:
                        current[ext] = source.read().decode("utf-8", errors="replace").strip()
                    continue
                if ext != "video" or not eligible(current):
                    continue
                kind = category(current.get("caption", ""))
                seen[kind] += 1
                quota = QUOTAS[kind]
                slot = len(kept[kind])
                replace_index = slot if slot < quota else rngs[kind].randrange(seen[kind])
                if replace_index >= quota:
                    continue
                record = dict(current)
                record.update(
                    {
                        "source_key": current_key,
                        "category": kind,
                        "video_offset": str(member.offset_data),
                        "video_size": str(member.size),
                    }
                )
                if slot < quota:
                    kept[kind].append(record)
                else:
                    kept[kind][replace_index] = record
    finally:
        reader.close()

    missing = {
        name: QUOTAS[name] - len(kept[name])
        for name in QUOTAS
        if len(kept[name]) < QUOTAS[name]
    }
    if missing:
        raise RuntimeError(f"not enough eligible samples: {missing}; seen={seen}")

    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    rows = [row for name in QUOTAS for row in kept[name]]
    session = requests.Session()
    try:
        for index, row in enumerate(rows, 1):
            filename = f"{row['category']}_{index:02d}_{safe_key(Path(row['source_key']).name)}.mp4"
            destination = media / filename
            start = int(row.pop("video_offset"))
            size = int(row.pop("video_size"))
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{start + size - 1}"},
                stream=True,
                timeout=(30, 300),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"video Range request returned HTTP {response.status_code}")
            temporary = destination.with_suffix(".part")
            with temporary.open("wb") as target:
                shutil.copyfileobj(response.raw, target, length=1024 * 1024)
            if temporary.stat().st_size != size:
                raise RuntimeError(
                    f"video size mismatch for {row['source_key']}: "
                    f"expected={size} actual={temporary.stat().st_size}"
                )
            temporary.replace(destination)
            row["video_path"] = f"media/{filename}"
            print(f"range_download={index}/{len(rows)} bytes={size}", flush=True)
    finally:
        session.close()
    return rows, seen


def probe_video(path: Path) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate:format=duration,size",
            "-of", "json", str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return {"probe_backend": "ffprobe", **json.loads(result.stdout)}

    # Some cluster environments provide decord but no ffmpeg CLI. Decode both
    # ends so this remains a real media integrity check rather than a metadata-only test.
    import decord

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    if len(reader) < 2:
        raise RuntimeError(f"video has fewer than two decodable frames: {path}")
    frames = reader.get_batch([0, len(reader) - 1]).asnumpy()
    if frames.shape[0] != 2:
        raise RuntimeError(f"failed to decode first/last frames: {path}")
    fps = float(reader.get_avg_fps())
    return {
        "probe_backend": "decord",
        "frame_count": len(reader),
        "fps": fps,
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "duration_seconds": len(reader) / fps if fps > 0 else None,
        "size": path.stat().st_size,
        "decoded_first_and_last": True,
    }


def finalize_existing(output: Path, url: str, seed: int) -> None:
    manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    if not isinstance(rows, list) or len(rows) != sum(QUOTAS.values()):
        raise ValueError("existing review manifest is missing or not exactly 50 rows")
    counts: dict[str, int] = {}
    for row in rows:
        category_name = str(row.get("category", ""))
        counts[category_name] = counts.get(category_name, 0) + 1
        path = output / str(row.get("video_path", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("video_sha256"):
            raise RuntimeError(f"SHA256 mismatch: {path}")
    if counts != QUOTAS:
        raise ValueError(f"category quotas differ: expected={QUOTAS}, actual={counts}")
    probes = {
        str(row["sample_id"]): probe_video(output / str(row["video_path"]))
        for row in rows
    }
    summary = {
        "review_version": "openvid_t2v_review_v1",
        "source_url": url,
        "selection_seed": seed,
        "eligibility": {"seconds": "3-10", "temporal_consistency_score": ">=0.995"},
        "quotas": QUOTAS,
        "eligible_seen": "not_recovered_after_post-download_finalize",
        "selected": len(rows),
        "video_probes": probes,
        "all_manifest_sha256_verified": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "selected": len(rows), "finalized": True}, ensure_ascii=False))


def write_manifest(output: Path, rows: list[dict[str, str]], seed: int) -> None:
    fields = [
        "review_index", "sample_id", "dataset", "dataset_view", "category", "caption",
        "aesthetic_score", "motion_score", "temporal_consistency_score", "camera_motion",
        "seconds", "fps", "frame", "part_id", "video_path", "video_sha256",
        "caption_match", "visual_quality", "temporal_quality", "usable_for_t2v", "notes",
    ]
    for index, row in enumerate(rows, 1):
        path = output / row["video_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row.update(
            {
                "review_index": str(index),
                "sample_id": f"openvid-review-{digest[:16]}",
                "dataset": "OpenVid-1M",
                "dataset_view": "finetrainers/OpenVid-1k-split",
                "video_sha256": digest,
                "caption_match": "1",
                "visual_quality": "1",
                "temporal_quality": "1",
                "usable_for_t2v": "1",
                "notes": "",
            }
        )
    with (output / "review_manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "review_version": "openvid_t2v_review_v1",
        "selection_seed": seed,
        "rows": [{key: row.get(key, "") for key in fields} for row in rows],
    }
    (output / "review_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_html(output: Path, rows: list[dict[str, str]], seed: int) -> None:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenVid T2V 审核</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#222}}header{{position:sticky;top:0;background:#fff;padding:12px 18px;border-bottom:1px solid #ddd;z-index:2}}
main{{max-width:1050px;margin:auto;padding:18px}}article{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin-bottom:18px}}video{{width:100%;max-height:580px;background:#111}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:6px;margin:10px 0;font-size:14px}}.caption{{line-height:1.5;background:#f7f7f7;padding:10px}}
.checks{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-top:12px}}select,textarea{{width:100%;padding:7px;box-sizing:border-box}}textarea{{min-height:60px}}button{{padding:8px 12px;margin-right:8px}}.idx{{font-weight:700}}
</style></head><body><header><b>OpenVid-1M T2V 审核包</b> · 50条 · selection seed {seed}<br>
<small>1=通过，0=不通过，2=存疑。每条视频同时检查画质、动态、caption对应和是否适合T2V replay。</small><div style="margin-top:8px"><button onclick="exportCSV()">导出CSV</button><button onclick="exportJSON()">导出JSON</button><button onclick="clearSaved()">清除本地记录</button></div></header><main id="app"></main>
<script>const rows={data};const key='openvid_t2v_review_v1';let saved=JSON.parse(localStorage.getItem(key)||'{{}}');
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function render(){{document.getElementById('app').innerHTML=rows.map((r,i)=>{{const s=saved[r.sample_id]||{{}};return `<article><div class="idx">${{i+1}}/50 · ${{esc(r.category)}} · ${{esc(r.sample_id)}}</div><video controls preload="metadata" src="${{esc(r.video_path)}}"></video><div class="meta"><span>aesthetic: ${{esc(r.aesthetic_score)}}</span><span>motion: ${{esc(r.motion_score)}}</span><span>temporal: ${{esc(r.temporal_consistency_score)}}</span><span>camera: ${{esc(r.camera_motion)}}</span><span>seconds: ${{esc(r.seconds)}}</span><span>fps: ${{esc(r.fps)}}</span></div><div class="caption">${{esc(r.caption)}}</div><div class="checks">${{['caption_match','visual_quality','temporal_quality','usable_for_t2v'].map(f=>`<label>${{f}}<select data-id="${{r.sample_id}}" data-field="${{f}}"><option value="1" ${{(s[f]??'1')==='1'?'selected':''}}>1 通过</option><option value="0" ${{s[f]==='0'?'selected':''}}>0 不通过</option><option value="2" ${{s[f]==='2'?'selected':''}}>2 存疑</option></select></label>`).join('')}}</div><label>notes<textarea data-id="${{r.sample_id}}" data-field="notes">${{esc(s.notes||'')}}</textarea></label></article>`}}).join('');document.querySelectorAll('[data-id]').forEach(e=>e.onchange=()=>{{saved[e.dataset.id]={{...(saved[e.dataset.id]||{{}}),[e.dataset.field]:e.value}};localStorage.setItem(key,JSON.stringify(saved))}})}}
function finalRows(){{return rows.map(r=>({{...r,caption_match:saved[r.sample_id]?.caption_match??'1',visual_quality:saved[r.sample_id]?.visual_quality??'1',temporal_quality:saved[r.sample_id]?.temporal_quality??'1',usable_for_t2v:saved[r.sample_id]?.usable_for_t2v??'1',notes:saved[r.sample_id]?.notes??''}}))}}
function dl(name,text,type){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type}}));a.download=name;a.click();URL.revokeObjectURL(a.href)}}
function exportJSON(){{dl('openvid_t2v_review_v1.json',JSON.stringify({{review_version:key,rows:finalRows()}},null,2),'application/json')}}
function exportCSV(){{const rs=finalRows(),ks=Object.keys(rs[0]),q=v=>'\"'+String(v??'').replaceAll('\"','\"\"')+'\"';dl('openvid_t2v_review_v1.csv','\\ufeff'+ks.join(',')+'\\n'+rs.map(r=>ks.map(k=>q(r[k])).join(',')).join('\\n'),'text/csv')}}
function clearSaved(){{if(confirm('确定清除浏览器中的审核记录？')){{localStorage.removeItem(key);saved={{}};render()}}}}render();</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Verify and finalize a fully downloaded bundle without downloading again.",
    )
    parser.add_argument(
        "--range-scan",
        action="store_true",
        help="Scan tar metadata with HTTP Range, then download only selected videos.",
    )
    parser.add_argument("--range-block-size", type=int, default=128 * 1024)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.finalize_existing:
        finalize_existing(output, args.url, args.seed)
        return
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"output exists: {output}; pass --replace to rebuild")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    rows, seen = (
        range_sample(args.url, output, args.seed, args.range_block_size)
        if args.range_scan
        else stream_sample(args.url, output, args.seed)
    )
    write_manifest(output, rows, args.seed)
    write_html(output, rows, args.seed)
    probes = {row["sample_id"]: probe_video(output / row["video_path"]) for row in rows}
    summary = {
        "review_version": "openvid_t2v_review_v1",
        "source_url": args.url,
        "selection_seed": args.seed,
        "eligibility": {"seconds": "3-10", "temporal_consistency_score": ">=0.995"},
        "quotas": QUOTAS,
        "eligible_seen": seen,
        "selected": len(rows),
        "video_probes": probes,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "selected": len(rows), "seen": seen}, ensure_ascii=False))


if __name__ == "__main__":
    main()
