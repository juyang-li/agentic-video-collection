#!/usr/bin/env python3
"""Offline Chromium acceptance test for the 30-card review package."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


CHROME = Path(__file__).resolve().parents[2] / "ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"
CHROME_LIBS = Path(__file__).resolve().parents[2] / "ivebench_lance/runtime/debs/extracted/usr/lib/x86_64-linux-gnu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    errors, external, playback = [], [], []
    with sync_playwright() as playwright:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(CHROME_LIBS) + (
            ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        browser = playwright.chromium.launch(executable_path=str(CHROME), headless=True,
                                             timeout=60000, env=environment)
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url)
                if request.url.startswith(("http:", "https:")) else None)
        page.goto((package / "index.html").as_uri())
        page.wait_for_selector("article")
        assert page.locator("article").count() == 30
        assert page.locator("article[data-section=multiturn]").count() == 30
        assert page.locator(".review").count() == 60 == len(reviews)
        assert page.locator("#summary").inner_text() == "审核项 60 · 非通过 0"
        for index in (0, 14, 29):
            card = page.locator("article").nth(index)
            card.scroll_into_view_if_needed()
            page.wait_for_function(
                "i => [...document.querySelectorAll('article')[i].querySelectorAll('video')].every(v=>v.src)",
                arg=index,
            )
            videos = card.locator("video")
            sample = min(3, videos.count())
            for video_index in range(sample):
                videos.nth(video_index).evaluate("v=>v.play()")
            page.wait_for_function(
                "x => [...document.querySelectorAll('article')[x.i].querySelectorAll('video')].slice(0,x.n).every(v=>!v.paused&&v.currentTime>.15&&v.videoWidth>0&&!v.error)",
                arg={"i": index, "n": sample},
            )
            playback.append({"eval_id": card.get_attribute("data-id"),
                             "videos": videos.evaluate_all("vs=>vs.slice(0,3).map(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}))")})
        first = page.locator(".review").first
        first.locator('input[data-field="overall_decision"][value="0"]').check()
        first.locator("textarea").fill("浏览器验收，逗号, 引号\"，换行\n第二行")
        assert page.locator("#summary").inner_text() == "审核项 60 · 非通过 1"
        with page.expect_download() as event:
            page.locator("#exportJson").click()
        exported_json = Path(event.value.path()).read_text()
        assert len(json.loads(exported_json)["labels"]) == 60
        page.set_input_files("#importFile", {"name": "labels.json", "mimeType": "application/json",
                                             "buffer": exported_json.encode()})
        with page.expect_download() as event:
            page.locator("#exportCsv").click()
        exported_csv = Path(event.value.path()).read_bytes()
        page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector("article")
        page.set_input_files("#importFile", {"name": "labels.csv", "mimeType": "text/csv",
                                             "buffer": exported_csv})
        assert page.locator("#summary").inner_text() == "审核项 60 · 非通过 1"
        page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector("article")
        assert page.locator("#summary").inner_text() == "审核项 60 · 非通过 0"
        assert not errors and not external
        report = {
            "status": "passed", "browser": browser.version, "transport": "file:// (offline)",
            "cards": 30, "review_units": 60, "playback_samples": playback,
            "javascript_errors": errors, "external_requests": external,
            "checks": ["30 multi-turn cards", "60 default-pass review rows",
                       "sampled source/base/gt video playback", "JSON/CSV export-import",
                       "no network requests"],
        }
        write_path = package / "browser_validation.json"
        write_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()

