#!/usr/bin/env python3
"""Download one large Hugging Face file with resumable parallel HTTP ranges."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()

    if args.workers < 1 or args.chunk_mib < 1:
        raise ValueError("workers and chunk-mib must be positive")
    url = hf_hub_url(
        args.repo_id,
        args.filename,
        repo_type="dataset",
        revision=args.revision,
    )
    metadata = get_hf_file_metadata(url)
    size = int(metadata.size or 0)
    if size <= 0:
        raise ValueError("remote file size is unavailable")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    state_path = output.with_suffix(output.suffix + ".ranges.json")
    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = [
        (index, start, min(start + chunk_size, size) - 1)
        for index, start in enumerate(range(0, size, chunk_size))
    ]

    completed: set[int] = set()
    if state_path.is_file() and partial.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("size") == size and state.get("chunk_size") == chunk_size:
            completed = {int(value) for value in state.get("completed", [])}
    file_descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(file_descriptor, size)
    lock = threading.Lock()
    started = time.monotonic()

    def save_state() -> None:
        atomic_json(
            state_path,
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "filename": args.filename,
                "size": size,
                "chunk_size": chunk_size,
                "completed": sorted(completed),
            },
        )

    def fetch(item: tuple[int, int, int]) -> int:
        index, start, end = item
        expected = end - start + 1
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = requests.get(
                    metadata.location,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=args.timeout,
                )
                if response.status_code != 206 or len(response.content) != expected:
                    raise RuntimeError(
                        f"range {index} status={response.status_code} "
                        f"bytes={len(response.content)}/{expected}"
                    )
                os.pwrite(file_descriptor, response.content, start)
                return index
            except Exception:
                if attempt == args.max_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 20))
        raise AssertionError("unreachable")

    pending = [item for item in ranges if item[0] not in completed]
    print(
        f"file={args.filename} size={size} chunks={len(ranges)} "
        f"resumed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch, item) for item in pending]
            for future in as_completed(futures):
                completed.add(future.result())
                with lock:
                    save_state()
                if len(completed) % 8 == 0 or len(completed) == len(ranges):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    downloaded = sum(
                        ranges[index][2] - ranges[index][1] + 1 for index in completed
                    )
                    print(
                        f"completed={len(completed)}/{len(ranges)} "
                        f"effective_mib_s={downloaded / elapsed / 1048576:.2f}",
                        flush=True,
                    )
    finally:
        os.close(file_descriptor)

    if len(completed) != len(ranges):
        raise RuntimeError("download ended with incomplete ranges")
    partial.replace(output)
    state_path.unlink(missing_ok=True)
    print(f"done output={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
