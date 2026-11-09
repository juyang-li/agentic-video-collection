#!/usr/bin/env python3
"""Resume a large gated Hugging Face file with verified parallel byte ranges."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers


CONTENT_RANGE = re.compile(r"bytes \d+-\d+/(\d+)")


def remote_size(url: str, headers: dict[str, str], timeout: float) -> int:
    request_headers = dict(headers)
    request_headers["Range"] = "bytes=0-0"
    response = requests.get(url, headers=request_headers, timeout=timeout)
    if response.status_code != 206:
        raise RuntimeError(f"range probe returned HTTP {response.status_code}")
    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
    if not match:
        raise RuntimeError("range probe did not return a valid Content-Range")
    return int(match.group(1))


def ranges(start: int, stop: int, chunk_size: int) -> list[tuple[int, int]]:
    result = []
    while start < stop:
        end = min(stop - 1, start + chunk_size - 1)
        result.append((start, end))
        start = end + 1
    return result


def download_range(
    url: str,
    headers: dict[str, str],
    output: Path,
    start: int,
    end: int,
    timeout: float,
    attempts: int,
) -> tuple[int, int, Path]:
    expected = end - start + 1
    if output.is_file() and output.stat().st_size == expected:
        return start, expected, output
    temporary = output.with_suffix(output.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request_headers = dict(headers)
            request_headers["Range"] = f"bytes={start}-{end}"
            with requests.get(
                url, headers=request_headers, stream=True, timeout=timeout
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}")
                written = 0
                with temporary.open("wb") as file:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            file.write(block)
                            written += len(block)
                if written != expected:
                    raise RuntimeError(f"short range: expected={expected} got={written}")
            temporary.replace(output)
            return start, expected, output
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(min(20, 2**attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument(
        "--prefix",
        type=Path,
        help="Existing contiguous prefix from a sequential .part download.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()

    if args.output.exists():
        print(json.dumps({"status": "preserved", "output": str(args.output)}))
        return
    if args.prefix is not None and (
        not args.prefix.is_file() or args.prefix.stat().st_size <= 0
    ):
        raise FileNotFoundError(f"prefix must be a non-empty file: {args.prefix}")
    if args.workers < 1 or args.chunk_mib < 8:
        raise ValueError("workers must be >=1 and chunk-mib must be >=8")

    headers = build_hf_headers(token=True)
    url = hf_hub_url(
        args.repo_id,
        args.filename,
        repo_type="dataset",
        revision=args.revision,
    )
    total = remote_size(url, headers, args.timeout)
    prefix_size = args.prefix.stat().st_size if args.prefix is not None else 0
    if prefix_size >= total:
        if prefix_size != total:
            raise ValueError(f"prefix exceeds remote size: {prefix_size}>{total}")
        if args.prefix is None:
            raise AssertionError("zero-length remote file is unsupported")
        args.prefix.replace(args.output)
        print(json.dumps({"status": "complete_from_prefix", "bytes": total}))
        return

    args.parts_dir.mkdir(parents=True, exist_ok=True)
    jobs = ranges(prefix_size, total, args.chunk_mib * 1024 * 1024)
    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_range,
                url,
                headers,
                args.parts_dir / f"{start:012d}-{end:012d}.range",
                start,
                end,
                args.timeout,
                args.attempts,
            ): (start, end)
            for start, end in jobs
        }
        for future in as_completed(futures):
            start, size, _ = future.result()
            completed += 1
            print(
                f"range_complete={completed}/{len(jobs)} start={start} bytes={size}",
                flush=True,
            )

    assembling = args.output.with_suffix(args.output.suffix + ".assembling")
    with assembling.open("wb") as destination:
        if args.prefix is not None:
            with args.prefix.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    destination.write(block)
        for start, end in jobs:
            part = args.parts_dir / f"{start:012d}-{end:012d}.range"
            with part.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    destination.write(block)
        destination.flush()
        os.fsync(destination.fileno())
    if assembling.stat().st_size != total:
        raise RuntimeError(
            f"assembled size mismatch: expected={total} got={assembling.stat().st_size}"
        )
    assembling.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "bytes": total,
                "prefix_bytes": prefix_size,
                "ranges": len(jobs),
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
