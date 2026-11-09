#!/usr/bin/env python3
"""Offline Chromium smoke test for a v2 preview package."""

import json
import os
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CHROME = ROOT.parent / "ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"
LIBS = ROOT.parent / "ivebench_lance/runtime/debs/extracted/usr/lib/x86_64-linux-gnu"

def main() -> None:
    package = Path(sys.argv[1]).resolve(); expected = int(sys.argv[2]); external = []
    environment = os.environ.copy(); environment["LD_LIBRARY_PATH"] = str(LIBS) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(CHROME), headless=True, env=environment)
        page = browser.new_page(); page.on("request", lambda request: external.append(request.url) if request.url.startswith(("http:", "https:")) else None)
        page.goto((package / "index.html").as_uri(), wait_until="load")
        assert page.locator("#position").inner_text() == f"1 / {expected}"
        page.locator("video").evaluate("v=>new Promise((ok,bad)=>{if(v.readyState>=1)ok();else{v.onloadedmetadata=ok;v.onerror=bad}})")
        page.click("#next"); assert page.locator("#position").inner_text() == f"2 / {expected}"
        assert not external; browser.close()
    report = {"status": "passed", "cards": expected, "video_metadata_loaded": True, "navigation": True, "external_requests": 0}
    (package / "browser_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
