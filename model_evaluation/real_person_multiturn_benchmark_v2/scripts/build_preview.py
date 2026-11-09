#!/usr/bin/env python3
"""Build a reviewable snapshot from all currently completed v2 chains."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT.parent / "real_person_multiturn_benchmark_v1"
FIELDS = ("person_relevance", "instruction_feasibility", "chain_coherence", "task_diversity", "technical_quality")

def digest(path: Path) -> str:
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()

def completed() -> list[tuple[dict, dict]]:
    chains = [json.loads(line) for line in (V1 / "benchmark.jsonl").read_text().splitlines() if line.strip()]
    state = json.loads((ROOT / "metadata/build_state.json").read_text())
    records = state.get("records", {})
    return [(chain, records[chain["chain_id"]]) for chain in chains
            if chain["chain_id"] in records and not records[chain["chain_id"]].get("problems")]

def build() -> Path:
    items = completed(); count = len(items)
    if not count: raise RuntimeError("no completed v2 chains")
    package = ROOT / "review" / f"real_person_multiturn_benchmark_v2_preview{count}_review"
    if package.exists() or package.with_suffix(".zip").exists(): raise FileExistsError(package)
    (package / "media").mkdir(parents=True)
    cards, rows = [], []
    for index, (chain, record) in enumerate(items, 1):
        source = V1 / chain["source"]["media"]; destination = package / "media" / source.name
        shutil.copy2(source, destination)
        cards.append({"index": index, "chain_id": chain["chain_id"], "chain_type": chain["chain_type"],
                      "source_caption": chain["source"]["official_source_caption"],
                      "source_dataset": f"IVEBench {chain['source']['subset']} #{chain['source']['official_id']}",
                      "video": f"media/{source.name}", "scene": chain["scene"], "turns": record["result"]["turns"]})
        rows.append({"review_id": chain["chain_id"], "chain_id": chain["chain_id"],
                     **{field: "1" for field in FIELDS}, "notes": ""})
    with (package / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"title": f"Real-person multi-turn benchmark v2 preview ({count}/30)",
               "version": package.name, "fields": list(FIELDS), "cards": cards, "defaults": rows}
    template = (V1 / "scripts/review_template.html").read_text()
    (package / "index.html").write_text(template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")))
    (package / "README.md").write_text(
        f"# v2 instruction preview ({count}/30)\n\nThis is a temporary review snapshot of the currently completed chains. "
        "Open `index.html` in Chrome or Edge. All labels default to 1; use 0 for failure and 2 for uncertainty.\n")
    (package / "package_summary.json").write_text(json.dumps({"status": "preview_pending_browser_validation",
        "cards": count, "media_files": count, "offline": True, "final_release": False}, indent=2) + "\n")
    print(package)
    return package

def finalize(package: Path) -> None:
    report = json.loads((package / "browser_validation.json").read_text())
    if report.get("status") != "passed": raise RuntimeError("browser validation did not pass")
    files = [path for path in sorted(package.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    (package / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(package).as_posix()}\n" for path in files))
    archive = package.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for path in sorted(package.rglob("*")):
            if path.is_file(): output.write(path, path.relative_to(package.parent).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None: raise RuntimeError("ZIP CRC failure")
        if {name.split("/")[0] for name in zipped.namelist()} != {package.name}: raise RuntimeError("multiple ZIP roots")
    receipt = {"status": "complete_preview", "archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": digest(archive), "cards": report["cards"], "remote_upload_status": "not_uploaded"}
    archive.with_suffix(".sha256.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    if len(sys.argv) == 1: build()
    elif len(sys.argv) == 3 and sys.argv[1] == "--finalize": finalize(Path(sys.argv[2]).resolve())
    else: raise SystemExit("usage: build_preview.py [--finalize PACKAGE]")
