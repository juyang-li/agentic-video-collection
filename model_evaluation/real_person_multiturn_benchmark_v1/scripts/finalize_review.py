#!/usr/bin/env python3
"""Checksum and archive the validated offline review package."""

import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "review/real_person_multiturn_benchmark_v1_review"
ARCHIVE = PACKAGE.with_suffix(".zip")
FIELDS = ("person_relevance", "instruction_feasibility", "chain_coherence", "task_diversity", "technical_quality")

def sha256(path: Path) -> str:
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()

def main() -> None:
    if ARCHIVE.exists(): raise FileExistsError(ARCHIVE)
    with (PACKAGE / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream: rows=list(csv.DictReader(stream))
    if len(rows)!=30 or any(row[f]!="1" for row in rows for f in FIELDS): raise ValueError("invalid review defaults/count")
    browser=json.loads((PACKAGE/"browser_validation.json").read_text())
    if browser.get("status")!="passed" or browser.get("external_requests")!=0: raise ValueError("browser validation missing")
    files=[path for path in sorted(PACKAGE.rglob("*")) if path.is_file() and path.name!="SHA256SUMS"]
    (PACKAGE/"SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(PACKAGE).as_posix()}\n" for path in files))
    with zipfile.ZipFile(ARCHIVE,"x",compression=zipfile.ZIP_STORED,allowZip64=True) as output:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file(): output.write(path,path.relative_to(PACKAGE.parent).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None: raise ValueError("ZIP CRC failure")
        if {name.split('/')[0] for name in archive.namelist()}!={PACKAGE.name}: raise ValueError("ZIP has multiple roots")
        videos=sum(name.endswith('.mp4') for name in archive.namelist())
    if videos!=30: raise ValueError(f"expected 30 videos, got {videos}")
    receipt={"status":"complete","archive":str(ARCHIVE),"bytes":ARCHIVE.stat().st_size,"sha256":sha256(ARCHIVE),
             "cards":30,"videos":30,"zip_crc_verified":True,"single_top_level_directory":True,
             "remote_path":"/apps/bypy/lance_review/"+ARCHIVE.name,"remote_upload_status":"not_uploaded"}
    ARCHIVE.with_suffix(".sha256.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
