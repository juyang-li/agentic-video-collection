#!/usr/bin/env python3
"""Exercise the review package in the repository's bundled Chromium."""

import json
import os
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "review/real_person_multiturn_benchmark_v1_review"
CHROME = ROOT.parent / "ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"
CHROME_LIBS = ROOT.parent / "ivebench_lance/runtime/debs/extracted/usr/lib/x86_64-linux-gnu"

def main() -> None:
    requests = []
    with sync_playwright() as p:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(CHROME_LIBS) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        browser = p.chromium.launch(executable_path=str(CHROME), headless=True, args=["--no-sandbox"], env=environment)
        page = browser.new_page(); page.on("request", lambda request: requests.append(request.url))
        page.goto((PACKAGE / "index.html").resolve().as_uri(), wait_until="load")
        if page.locator("#position").inner_text() != "1 / 30": raise RuntimeError("wrong card count")
        if page.locator("video").evaluate("v=>v.readyState") < 1: page.locator("video").evaluate("v=>new Promise((ok,bad)=>{v.onloadedmetadata=ok;v.onerror=bad})")
        page.select_option('[data-f="person_relevance"]', "2")
        if page.evaluate("localStorage.length") < 1: raise RuntimeError("localStorage did not persist")
        page.click("#next")
        if page.locator("#position").inner_text() != "2 / 30": raise RuntimeError("navigation failed")
        browser.close()
    external = [url for url in requests if not url.startswith("file:")]
    if external: raise RuntimeError(f"offline package made network requests: {external}")
    report = {"status": "passed", "cards": 30, "video_metadata_loaded": True, "navigation": True,
              "local_storage": True, "external_requests": 0}
    (PACKAGE / "browser_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
