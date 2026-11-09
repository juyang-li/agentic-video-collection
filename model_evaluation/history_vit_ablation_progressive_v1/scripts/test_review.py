#!/usr/bin/env python3
"""Offline Chromium acceptance test for the no-camera review package."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


MODEL_EVAL = Path(__file__).resolve().parents[2]
CHROME = MODEL_EVAL / "ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"
CHROME_LIBS = MODEL_EVAL / "ivebench_lance/runtime/debs/extracted/usr/lib/x86_64-linux-gnu"
FIELDS = (
    "edit_applied",
    "history_preserved",
    "normal_equivalent",
    "identity_background_stable",
    "temporal_consistency",
    "visual_quality",
    "overall_pass",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    if len(reviews) != 48 or len({row["review_id"] for row in reviews}) != 48:
        raise RuntimeError("expected 48 unique review units")
    if any(row[field] != "1" for row in reviews for field in FIELDS):
        raise RuntimeError("review defaults must all be 1")

    errors: list[str] = []
    external: list[str] = []
    playback: list[dict] = []
    with sync_playwright() as playwright:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(CHROME_LIBS) + (
            ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        browser = playwright.chromium.launch(
            executable_path=str(CHROME), headless=True, timeout=60000, env=environment)
        page = browser.new_page(viewport={"width": 1800, "height": 1000}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url)
                if request.url.startswith(("http:", "https:")) else None)
        page.goto((package / "index.html").as_uri())
        page.wait_for_selector("article")
        assert page.locator("article").count() == 6
        assert page.locator("article[data-section=multiturn]").count() == 6
        assert page.locator(".review").count() == 48 == len(reviews)
        assert page.locator("#summary").inner_text() == "审核项 48 · 非通过 0"

        for index in (0, 2, 5):
            card = page.locator("article").nth(index)
            card.scroll_into_view_if_needed()
            page.wait_for_function(
                "i => [...document.querySelectorAll('article')[i].querySelectorAll('video')].every(v=>v.src)",
                arg=index,
            )
            videos = card.locator("video")
            assert videos.count() == 21
            sample_indexes = [0, 8, 9, 13, 15, 16, 20]
            for video_index in sample_indexes:
                videos.nth(video_index).evaluate("v=>v.play()")
            page.wait_for_function(
                "x => x.js.every(j=>{const v=document.querySelectorAll('article')[x.i].querySelectorAll('video')[j];return !v.paused&&v.currentTime>.15&&v.videoWidth>0&&!v.error})",
                arg={"i": index, "js": sample_indexes},
            )
            playback.append({
                "card_id": card.get_attribute("data-id"),
                "videos": videos.evaluate_all(
                    "(vs,js)=>js.map(i=>({index:i,width:vs[i].videoWidth,height:vs[i].videoHeight,duration:vs[i].duration}))",
                    sample_indexes,
                ),
            })
            for video_index in sample_indexes:
                videos.nth(video_index).evaluate("v=>v.pause()")

        first = page.locator(".review").first
        first.locator('input[data-field="overall_pass"][value="0"]').check()
        first.locator("textarea").fill("浏览器验收，逗号, 引号\"，换行\n第二行")
        assert page.locator("#summary").inner_text() == "审核项 48 · 非通过 1"
        with page.expect_download() as event:
            page.locator("#exportJson").click()
        exported_json = Path(event.value.path()).read_text(encoding="utf-8")
        assert len(json.loads(exported_json)["labels"]) == 48
        page.set_input_files(
            "#importFile",
            {"name": "labels.json", "mimeType": "application/json", "buffer": exported_json.encode()},
        )
        with page.expect_download() as event:
            page.locator("#exportCsv").click()
        exported_csv = Path(event.value.path()).read_bytes()
        page.evaluate("localStorage.clear()")
        page.reload()
        page.wait_for_selector("article")
        page.set_input_files(
            "#importFile",
            {"name": "labels.csv", "mimeType": "text/csv", "buffer": exported_csv},
        )
        assert page.locator("#summary").inner_text() == "审核项 48 · 非通过 1"
        page.evaluate("localStorage.clear()")
        page.reload()
        page.wait_for_selector("article")
        assert page.locator("#summary").inner_text() == "审核项 48 · 非通过 0"
        assert not errors and not external
        report = {
            "status": "passed",
            "browser": browser.version,
            "transport": "file:// (offline)",
            "cards": 6,
            "review_units": 48,
            "videos_per_card": 21,
            "playback_samples": playback,
            "javascript_errors": errors,
            "external_requests": external,
            "checks": [
                "6 no-camera checkpoint cards",
                "48 default-pass ablation reviews",
                "sampled source/direct/normal/four-ablation playback for both models",
                "JSON/CSV export-import",
                "no network requests",
            ],
        }
        (package / "browser_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()
