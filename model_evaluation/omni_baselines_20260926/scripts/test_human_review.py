#!/usr/bin/env python3
"""Offline Chromium smoke test for the aligned human review UI."""

import argparse
import csv
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = Path(__file__).resolve().parents[2] / "ivebench_lance/runtime/google-chrome-stable/opt/google/chrome/chrome"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors, external, playback = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(CHROME), headless=True, timeout=60000)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: external.append(r.url) if r.url.startswith(("http:", "https:")) else None)
        page.goto((package / "index.html").as_uri())
        page.wait_for_selector("article")
        count = len(rows)
        assert count in (24, 48)
        assert page.locator("article").count() == count
        assert page.locator("#summary").inner_text() == f"已标注 0/{count}"
        embedded = json.loads(page.locator("#reviewData").text_content())
        assert [x["eval_id"] for x in embedded["items"]] == [x["eval_id"] for x in rows]
        for i in sorted({0, count // 2, count - 1}):
            card = page.locator("article").nth(i)
            card.scroll_into_view_if_needed()
            page.wait_for_function("i => [...document.querySelectorAll('article')[i].querySelectorAll('video')].every(v=>v.src)", arg=i)
            card.locator(".sync").click()
            page.wait_for_function("i => [...document.querySelectorAll('article')[i].querySelectorAll('video')].every(v=>!v.paused&&v.currentTime>.2&&v.videoWidth>0&&!v.error)", arg=i)
            values = card.locator("video").evaluate_all("vs=>vs.map(v=>({time:v.currentTime,width:v.videoWidth,height:v.videoHeight,duration:v.duration}))")
            assert max(v["time"] for v in values) - min(v["time"] for v in values) < .35
            card.locator(".sync").click()
            playback.append({"eval_id": rows[i]["eval_id"], "videos": values})
        first = page.locator("article").first
        first.scroll_into_view_if_needed()
        first.locator('input[data-field="instruction_correctness"][value="1"]').check()
        first.locator("textarea").fill("浏览器验收，逗号, 引号\"，换行\n第二行")
        assert page.locator("#summary").inner_text() == f"已标注 1/{count}"
        with page.expect_download() as event:
            page.locator("#exportJson").click()
        exported_json = Path(event.value.path()).read_text()
        assert len(json.loads(exported_json)["labels"]) == count
        page.set_input_files("#importFile", {"name": "labels.json", "mimeType": "application/json", "buffer": exported_json.encode()})
        assert page.locator("#summary").inner_text() == f"已标注 1/{count}"
        with page.expect_download() as event:
            page.locator("#exportCsv").click()
        exported_csv = Path(event.value.path()).read_bytes()
        page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector("article")
        page.set_input_files("#importFile", {"name": "labels.csv", "mimeType": "text/csv", "buffer": exported_csv})
        assert page.locator("#summary").inner_text() == f"已标注 1/{count}"
        page.evaluate("localStorage.clear()")
        page.reload()
        page.wait_for_selector("article")
        assert page.locator("#summary").inner_text() == f"已标注 0/{count}"
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors and not external
        report = {"status": "passed", "browser": browser.version, "transport": "file:// (offline)",
                  "sample_count": count, "playback_samples": playback, "javascript_errors": errors,
                  "external_requests": external,
                  "checks": [f"{count} aligned rows and initially blank labels", "embedded data matches manifest order",
                             "first/middle/last synchronized playback", "JSON and CSV export/import, localStorage reset",
                             "mobile layout has no horizontal overflow"]}
        (package / "browser_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()
