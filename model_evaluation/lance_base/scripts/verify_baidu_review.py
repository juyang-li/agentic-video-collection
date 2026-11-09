#!/usr/bin/env python3
"""Read-only Baidu listing verification; produce a credential-free ZIP receipt."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import requests

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "lance_general_edit_finetune/1_specialist_filtering/scripts"))
from upload_baidu_xpan import checked_json, load_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    receipt_path = archive.with_suffix(".sha256.json")
    upload = json.loads(archive.with_suffix(".baidu_upload.json").read_text())
    receipt = json.loads(receipt_path.read_text())
    assert upload["status"] == "complete"
    assert upload["local"] == str(archive)
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == receipt["sha256"]
    assert receipt["bytes"] == archive.stat().st_size == upload["size"]
    remote = receipt["remote_path"]
    assert upload["remote"] == remote and remote.startswith("/apps/bypy/lance_review/")
    token = load_token(Path.home() / ".bypy/bypy.json")
    matches, sidecars = [], []
    session = requests.Session()
    try:
        for offset in range(0, 10000, 1000):
            response = session.get("https://pan.baidu.com/rest/2.0/xpan/file", params={
                "method": "list", "access_token": token, "dir": str(Path(remote).parent),
                "start": offset, "limit": 1000}, timeout=(20, 60))
            entries = checked_json(response, "verify uploaded review listing").get("list", [])
            matches.extend(entry for entry in entries if entry.get("path") == remote)
            sidecars.extend(entry for entry in entries if entry.get("path") == str(Path(remote).with_suffix(".sha256.json")))
            if len(entries) < 1000:
                break
        else:
            raise RuntimeError("listing exceeded pagination bound")
    except requests.RequestException as error:
        raise RuntimeError("listing network failure: " + type(error).__name__) from None
    assert len(matches) == 1, "uploaded archive missing or ambiguous in fresh listing"
    entry = matches[0]
    assert int(entry["size"]) == receipt["bytes"]
    assert entry["fs_id"] == upload["remote_result"]["fs_id"]
    # Do not imply that a local SHA256 is an independent remote download check.
    receipt.pop("local_file", None)
    receipt.update(upload_status="complete", remote_listing_verified=True,
                   remote_bytes=int(entry["size"]), remote_fs_id=entry["fs_id"],
                   verified_at_utc=datetime.now(timezone.utc).isoformat(),
                   sha256_scope="local ZIP; upload completion and remote listing size separately verified",
                   sidecar_remote_path=str(Path(remote).with_suffix(".sha256.json")),
                   sidecar_exists=bool(sidecars))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
