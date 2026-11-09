#!/usr/bin/env python3
"""Fetch a small, reproducible reference-edit review set, not a training set."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time
import zlib

import requests

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent))
from remote_zip_extract import parse_central_directory

TRIP_REPO = "TripVVT/TripVVT-10K"
TRIP_REV = "596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3"
HMDB_REPO = "MichiganNLP/hmdb"
TOKEN_FILE = Path("/scratch/prj0000000275/jcy/.cache/huggingface/token")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def get(url, *, auth=False, headers=None, stream=False):
    headers = dict(headers or {})
    if auth:
        headers["Authorization"] = "Bearer " + TOKEN_FILE.read_text().strip()
    for attempt in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=(20, 90), stream=stream)
            if r.status_code in (200, 206):
                return r
            status = r.status_code
            r.close()
        except requests.RequestException as e:
            status = type(e).__name__
        if attempt < 3:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"asset request failed: {status}")


def download(member):
    dest = ROOT / "assets" / "tripvvt" / member
    if dest.is_file() and dest.stat().st_size:
        return member, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get(f"https://huggingface.co/datasets/{TRIP_REPO}/resolve/{TRIP_REV}/{member}", auth=True, stream=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with r, tmp.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)
    tmp.replace(dest)
    return member, dest.stat().st_size


def tripvvt():
    meta = ROOT / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    population = []
    for split in ("train", "test"):
        path = meta / f"tripvvt_{split}.jsonl"
        if not path.exists():
            r = get(f"https://huggingface.co/datasets/{TRIP_REPO}/resolve/{TRIP_REV}/{split}.jsonl", auth=True)
            path.write_bytes(r.content)
        for i, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                population.append({**json.loads(line), "official_split": split, "official_row": i})
    manifest = meta / "garment_selection.json"
    if manifest.exists():
        selected = json.loads(manifest.read_text())["samples"]
    else:
        rng = random.Random(20261030)
        selected, seen = [], set()
        for category, count in [("upper_body", 9), ("lower_body", 8), ("dresses", 8)]:
            candidates = [x for x in population if x["cloth_type"] == category]
            rng.shuffle(candidates)
            for row in candidates:
                if row["output"] in seen:
                    continue
                seen.add(row["output"])
                selected.append({"sample_id": f"cloth_{len(selected)+1:03d}", **row})
                if sum(x["cloth_type"] == category for x in selected) == count:
                    break
        assert len(selected) == 25 and len(seen) == 25
        save_json(manifest, {"repo": TRIP_REPO, "revision": TRIP_REV, "seed": 20261030,
                            "population": len(population), "unique_target_grouping": True,
                            "metadata_sha256": {s: sha(meta / f"tripvvt_{s}.jsonl") for s in ("train", "test")},
                            "samples": selected})
    members = sorted({r[k] for r in selected for k in ("input", "output", "cloth_image")})
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, result in enumerate(pool.map(download, members), 1):
            print(f"TripVVT {i}/{len(members)} {result[0]} {result[1]}", flush=True)


def read_range(url, start, end):
    r = get(url, headers={"Range": f"bytes={start}-{end}"})
    if r.status_code != 206:
        r.close()
        raise RuntimeError("remote ZIP ignored Range")
    data = r.content
    if len(data) != end - start + 1:
        raise RuntimeError("remote ZIP range length mismatch")
    return data


def hmdb():
    meta = ROOT / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    info_path = meta / "hmdb_repo.json"
    if not info_path.exists():
        info = get(f"https://huggingface.co/api/datasets/{HMDB_REPO}").json()
        save_json(info_path, {"repo": HMDB_REPO, "revision": info["sha"], "official_page": "https://serre.lab.brown.edu/hmdb51.html"})
    info = json.loads(info_path.read_text())
    url = f"https://huggingface.co/datasets/{HMDB_REPO}/resolve/{info['revision']}/hmdb51_org.zip"
    central = meta / "hmdb_central_directory.bin"
    if not central.exists():
        r = get(url, headers={"Range": "bytes=-65536"})
        assert r.status_code == 206
        tail = r.content
        pos = tail.rfind(b"PK\x05\x06")
        assert pos >= 0
        eocd = struct.unpack_from("<4s4H2IH", tail, pos)
        size, offset = eocd[5:7]
        assert size != 0xffffffff and offset != 0xffffffff
        central.write_bytes(read_range(url, offset, offset + size - 1))
    entries = parse_central_directory(central)
    manifest = meta / "motion_reference_selection.json"
    counts = {"wave": 7, "clap": 6, "turn": 6, "walk": 6}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get("selection_version") != 2:
            save_json(meta / "hmdb_initial_unfiltered_selection.json", old)
            manifest.unlink()
    if manifest.exists():
        selected = json.loads(manifest.read_text())["samples"]
    else:
        selected = []
        rng = random.Random(20261031)
        for cls, count in counts.items():
            unique = {}
            for e in sorted(entries, key=lambda x: (x.name.count('/'), x.name)):
                b = Path(e.name).name
                if not b.endswith('.avi') or f'/{cls}/' not in '/'+e.name or b.startswith('._') or '__MACOSX/' in e.name:
                    continue
                if '_np1_' not in b or '_bad_' in b or any(x in b.lower() for x in ['baby','lil__man','theslowclap1','maya_beim','kids','girl_']):
                    continue
                if cls in ('turn','walk') and f'_{cls}_f_' not in b:
                    continue
                if cls in ('wave','clap') and f'_{cls}_h_' in b:
                    continue
                unique.setdefault(b,e)
            candidates = list(unique.values())
            print("HMDB population", cls, len(candidates), flush=True)
            if len(candidates) < count:
                raise RuntimeError("unexpected ZIP layout: " + repr([e.name for e in entries[:10]]))
            # Extra candidates allow technical/visual rejection without changing the sampling seed.
            for e in rng.sample(candidates, count + 5):
                selected.append({"class": cls, "member": e.name, "crc32": e.crc32,
                                 "compressed_size": e.compressed_size, "uncompressed_size": e.uncompressed_size,
                                 "local_offset": e.local_offset, "method": e.method})
        save_json(manifest, {**info, "selection_version": 2, "seed": 20261031, "quota": counts, "samples": selected,
                            "filters": "deduplicate basename; np1; exclude bad quality and head-only; full-body turn/walk; adult-oriented candidates; then direct visual selection",
                            "zip_central_sha256": sha(central), "semantics": "action reference, not edited target"})

    def extract(row):
        dest = ROOT / "assets" / "hmdb" / row["class"] / Path(row["member"]).name
        if dest.exists() and dest.stat().st_size == row["uncompressed_size"]:
            return row["member"]
        offset = row["local_offset"]
        header = read_range(url, offset, offset + 29)
        values = struct.unpack("<4s5H3I2H", header)
        assert values[0] == b"PK\x03\x04"
        start = offset + 30 + values[-2] + values[-1]
        raw = read_range(url, start, start + row["compressed_size"] - 1)
        data = zlib.decompress(raw, -15) if row["method"] == 8 else raw
        assert len(data) == row["uncompressed_size"] and zlib.crc32(data) & 0xffffffff == row["crc32"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return row["member"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, future in enumerate(as_completed([pool.submit(extract, r) for r in selected]), 1):
            print(f"HMDB {i}/{len(selected)} {future.result()}", flush=True)


def font():
    directory=ROOT/'assets/font';directory.mkdir(parents=True,exist_ok=True)
    for name,url in [('NotoSansCJKsc-Regular.otf','https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'),
                     ('LICENSE.txt','https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE')]:
        path=directory/name
        if not path.exists():path.write_bytes(get(url).content)
        print('font asset',name,path.stat().st_size,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("garment", "motion", "font", "all"))
    args = parser.parse_args()
    if args.kind in ("garment", "all"):
        tripvvt()
    if args.kind in ("motion", "all"):
        hmdb()
    if args.kind=='font':font()
