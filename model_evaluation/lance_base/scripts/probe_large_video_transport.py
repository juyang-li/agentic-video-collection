#!/usr/bin/env python3
"""Test exact public source-video URL transport for an oversized Spatial request."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lance_tools"))
import judge_spatial_openrouter as judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="Sp_Landscape_change_07")
    parser.add_argument("--resolved-url", action="store_true", help="use verified final public CDN URL to avoid redirects")
    parser.add_argument("--vertex-inline", action="store_true", help="separate probe of Google's Vertex route; never merge with Studio scores")
    args = parser.parse_args()
    tasks = json.loads((ROOT / "evaluator_workspace/judge_tasks.json").read_text())
    task = next(t for t in tasks if t["sample_id"] == args.sample)
    relative = Path(task["source_path"]).relative_to(ROOT / "data/official_source")
    source_url = f"https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench/resolve/{task['dataset_revision']}/{quote(str(relative))}"
    metadata, rules = judge.protocol()
    key = os.environ["OPEN_ROUTER_QWEN"]
    suffix = "_vertex_inline" if args.vertex_inline else ("_resolved" if args.resolved_url else "")
    directory = ROOT / "gemini38_low_openrouter_v1/transport_probe" / (args.sample + suffix)
    if (directory / "receipt.json").exists():
        raise RuntimeError("probe already recorded; inspect receipt rather than repeat blindly")
    with requests.Session() as session:
        session.trust_env = False
        download = session.get(source_url, timeout=(20, 180))
        download.raise_for_status()
        if hashlib.sha256(download.content).hexdigest() != task["source_sha256"]:
            raise ValueError("public URL bytes do not match official source")
        if args.resolved_url:
            source_url = download.url
        payload = judge.payload_for(task, rules)
        if args.vertex_inline:
            payload["provider"] = {**payload["provider"], "only": ["google-vertex/global"], "order": ["google-vertex/global"]}
            metadata = deepcopy(metadata)
            metadata.update(protocol="omni_edit_spatial_gemini38_low_vertex_probe", expected_response_provider="Google")
            metadata["settings"]["provider"] = payload["provider"]
            judge.PROVIDER = "Google"
        else:
            payload["messages"][0]["content"][2]["video_url"]["url"] = source_url
        record = {"task": task, "source_url": source_url, "source_url_sha256_verified": task["source_sha256"],
                  "generated_sha256": judge.sha256_file(Path(task["generated_video"])),
                  "scoring_protocol": metadata, "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": "request_started_outcome_unknown"}
        judge.write_json(directory / "receipt.json", record)
        response = session.post(judge.ENDPOINT, headers={"Authorization": "Bearer " + key},
                                json=payload, timeout=(20, 300))
        body = json.loads(json.dumps(response.json()).replace(key, "[REDACTED]"))
        record.update(http_status=response.status_code, response=body, status="response_received")
        if response.ok:
            try:
                record["score"] = judge.parse_score(body)
            except (ValueError, KeyError, TypeError) as error:
                record["parse_error"] = str(error)
        judge.write_json(directory / "receipt.json", record)
        print(json.dumps({k: record.get(k) for k in ("http_status", "score", "parse_error", "response")}, indent=2))


if __name__ == "__main__":
    main()
