#!/usr/bin/env python3
"""Offline Chromium acceptance test for the dataset audit package."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


CHROME = Path(__file__).resolve().parents[3] / "model_evaluation/ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"
CHROME_LIBS = Path(__file__).resolve().parents[3] / "model_evaluation/ivebench_lance/runtime/debs/extracted/usr/lib/x86_64-linux-gnu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        audits = list(csv.DictReader(stream))
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
        assert page.locator("article").count() == 26
        assert page.locator('article[data-origin="ue_synthetic"]').count() == 20
        assert page.locator('article[data-origin="coinve"]').count() == 6
        assert page.locator(".review").count() == 26 == len(audits)
        assert page.locator("#summary").inner_text() == "样本 26 · 非全通过 0"
        for index in (0, 9, 19, 20, 25):
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
            playback.append({
                "sample_id": card.get_attribute("data-id"),
                "videos": videos.evaluate_all(
                    "vs=>vs.slice(0,3).map(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}))"),
            })
        first = page.locator(".review").first
        first.locator('input[data-field="technical_quality"][value="2"]').check()
        first.locator("textarea").fill("浏览器验收，逗号, 引号\"，换行\n第二行")
        assert page.locator("#summary").inner_text() == "样本 26 · 非全通过 1"
        with page.expect_download() as event:
            page.locator("#exportJson").click()
        exported_json = Path(event.value.path()).read_text()
        assert len(json.loads(exported_json)["labels"]) == 26
        page.set_input_files("#importFile", {"name": "labels.json", "mimeType": "application/json",
                                             "buffer": exported_json.encode()})
        with page.expect_download() as event:
            page.locator("#exportCsv").click()
        exported_csv = Path(event.value.path()).read_bytes()
        page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector("article")
        page.set_input_files("#importFile", {"name": "labels.csv", "mimeType": "text/csv",
                                             "buffer": exported_csv})
        assert page.locator("#summary").inner_text() == "样本 26 · 非全通过 1"
        page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector("article")
        assert page.locator("#summary").inner_text() == "样本 26 · 非全通过 0"
        assert not errors and not external
        report = {
            "status": "passed", "browser": browser.version, "transport": "file:// (offline)",
            "cards": 26, "audit_rows": 26, "playback_samples": playback,
            "javascript_errors": errors, "external_requests": external,
            "checks": ["20 synthetic and 6 CoinVE cards", "26 default-pass audit rows",
                       "synthetic-chain and CoinVE-pair playback", "JSON/CSV export-import",
                       "no network requests"],
        }
        (package / "browser_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()

