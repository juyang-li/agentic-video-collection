#!/usr/bin/env python3
"""Resumable Baidu Xpan block upload using the access token already configured by bypy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import requests


PRECREATE_URL = "https://pan.baidu.com/rest/2.0/xpan/file"
UPLOAD_URL = "https://d.pcs.baidu.com/rest/2.0/pcs/superfile2"


def load_token(config: Path) -> str:
    data = json.loads(config.read_text(encoding="utf-8"))
    token = data.get("access_token")
    if not token:
        raise ValueError(f"no access_token in {config}")
    return str(token)


def block_hashes(path: Path, block_size: int) -> list[str]:
    hashes: list[str] = []
    with path.open("rb") as file:
        while block := file.read(block_size):
            hashes.append(hashlib.md5(block).hexdigest())
    return hashes


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def checked_json(response: requests.Response, action: str) -> dict[str, object]:
    try:
        response.raise_for_status()
    except requests.RequestException:
        # requests includes the complete query string in its exception, which
        # would expose access_token. Keep all propagated errors credential-free.
        raise RuntimeError(f"{action} HTTP failure: status={response.status_code}") from None
    data = response.json()
    error = int(data.get("errno", data.get("error_code", 0)) or 0)
    if error:
        raise RuntimeError(f"{action} failed: {json.dumps(data, ensure_ascii=False)}")
    return data


def safe_post(
    session: requests.Session, action: str, url: str, **kwargs: object
) -> requests.Response:
    try:
        return session.post(url, **kwargs)
    except requests.RequestException as error:
        raise RuntimeError(
            f"{action} network failure: {type(error).__name__}"
        ) from None


def retry_call(
    action: str, retries: int, callback: Callable[[], dict[str, object]]
) -> dict[str, object]:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            return callback()
        except RuntimeError as error:
            last_error = str(error)
            if attempt == retries:
                raise RuntimeError(
                    f"{action} failed after {retries} attempts: {last_error}"
                ) from None
            time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def precreate(
    session: requests.Session,
    token: str,
    remote: str,
    size: int,
    hashes: list[str],
) -> dict[str, object]:
    response = safe_post(
        session,
        "precreate",
        PRECREATE_URL,
        params={"method": "precreate", "access_token": token},
        data={
            "path": remote,
            "size": str(size),
            "isdir": "0",
            "autoinit": "1",
            "rtype": "3",
            "block_list": json.dumps(hashes),
        },
        timeout=(20, 120),
    )
    return checked_json(response, "precreate")


def upload_block(
    session: requests.Session,
    token: str,
    remote: str,
    uploadid: str,
    index: int,
    block: bytes,
) -> dict[str, object]:
    response = safe_post(
        session,
        f"upload block {index}",
        UPLOAD_URL,
        params={
            "method": "upload",
            "access_token": token,
            "type": "tmpfile",
            "path": remote,
            "uploadid": uploadid,
            "partseq": str(index),
        },
        files={"file": ("block", block, "application/octet-stream")},
        timeout=(20, 300),
    )
    return checked_json(response, f"upload block {index}")


def create_file(
    session: requests.Session,
    token: str,
    remote: str,
    size: int,
    uploadid: str,
    hashes: list[str],
) -> dict[str, object]:
    response = safe_post(
        session,
        "create",
        PRECREATE_URL,
        params={"method": "create", "access_token": token},
        data={
            "path": remote,
            "size": str(size),
            "isdir": "0",
            "rtype": "3",
            "uploadid": uploadid,
            "block_list": json.dumps(hashes),
        },
        timeout=(20, 180),
    )
    return checked_json(response, "create")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--bypy-config", type=Path, default=Path.home() / ".bypy/bypy.json")
    parser.add_argument("--block-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    local = args.local.resolve()
    if not local.is_file():
        raise FileNotFoundError(local)
    if not args.remote.startswith("/apps/bypy/"):
        raise ValueError("remote must stay inside /apps/bypy/")

    size = local.stat().st_size
    mtime_ns = local.stat().st_mtime_ns
    hashes = block_hashes(local, args.block_size)
    token = load_token(args.bypy_config)
    session = requests.Session()
    session.headers["User-Agent"] = "lance-review-uploader/1.0"

    state: dict[str, object] = {}
    if args.state.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
    matches = (
        state.get("local") == str(local)
        and state.get("remote") == args.remote
        and state.get("size") == size
        and state.get("mtime_ns") == mtime_ns
        and state.get("block_size") == args.block_size
        and state.get("block_md5s") == hashes
        and state.get("uploadid")
    )

    if not matches:
        result = retry_call(
            "precreate",
            args.retries,
            lambda: precreate(session, token, args.remote, size, hashes),
        )
        needed = [int(value) for value in result.get("block_list", range(len(hashes)))]
        state = {
            "local": str(local),
            "remote": args.remote,
            "size": size,
            "mtime_ns": mtime_ns,
            "block_size": args.block_size,
            "block_md5s": hashes,
            "uploadid": result["uploadid"],
            "needed": needed,
            "completed": [],
            "status": "uploading",
        }
        write_state(args.state, state)

    uploadid = str(state["uploadid"])
    needed = [int(value) for value in state.get("needed", range(len(hashes)))]
    completed = {int(value) for value in state.get("completed", [])}
    with local.open("rb") as file:
        for position, index in enumerate(needed, 1):
            if index in completed:
                continue
            file.seek(index * args.block_size)
            block = file.read(args.block_size)
            expected = hashes[index]
            for attempt in range(1, args.retries + 1):
                try:
                    result = upload_block(session, token, args.remote, uploadid, index, block)
                    returned = str(result.get("md5", ""))
                    if returned and returned != expected:
                        raise RuntimeError(
                            f"block {index} md5 mismatch: expected={expected} returned={returned}"
                        )
                    completed.add(index)
                    state["completed"] = sorted(completed)
                    write_state(args.state, state)
                    print(
                        f"blocks={len(completed)}/{len(needed)} index={index} "
                        f"bytes={min((index + 1) * args.block_size, size)}/{size}",
                        flush=True,
                    )
                    break
                except Exception:
                    if attempt == args.retries:
                        raise
                    time.sleep(min(2**attempt, 10))

    result = retry_call(
        "create",
        args.retries,
        lambda: create_file(session, token, args.remote, size, uploadid, hashes),
    )
    if int(result.get("size", size)) != size:
        raise RuntimeError(f"remote size mismatch: local={size} response={result.get('size')}")
    state["status"] = "complete"
    state["remote_result"] = {
        key: result.get(key) for key in ("fs_id", "path", "server_filename", "size")
    }
    write_state(args.state, state)
    print(json.dumps(state["remote_result"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
