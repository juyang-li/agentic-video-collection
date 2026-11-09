#!/usr/bin/env python3
"""Real-Chromium acceptance test for an offline IVEBench review package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright


def progress(stage: str) -> None:
    print(json.dumps({"browser_acceptance_stage": stage}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "review_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_samples = len(rows)
    if expected_samples not in (43, 91, 94):
        raise ValueError(f"unexpected review sample count: {expected_samples}")
    errors: list[str] = []
    external: list[str] = []
    playback = []
    checks = []
    with sync_playwright() as playwright, tempfile.TemporaryDirectory(prefix="ivebench-review-") as temporary:
        progress("launch_browser")
        # Playwright's bundled Chromium/headless shell may omit H.264 support.
        # Prefer the project-local Chrome runtime used to verify the H.264
        # review deliverables; retain the Chromium channel as a fallback for
        # environments where that runtime has not been provisioned.
        chrome = (
            Path(__file__).resolve().parents[1]
            / "runtime/google-chrome-stable/opt/google/chrome/chrome"
        )
        launch_options = {"headless": True, "timeout": 60_000}
        if chrome.exists():
            launch_options["executable_path"] = str(chrome)
        else:
            launch_options["channel"] = "chromium"
        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context(viewport={"width": 1680, "height": 1050}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url) if request.url.startswith(("http:", "https:")) else None)
        progress("load_offline_page")
        page.goto((package / "index.html").as_uri(), wait_until="domcontentloaded")
        page.wait_for_selector(".card")
        if page.locator(".card").count() != expected_samples:
            raise AssertionError("not all continuous-scroll cards rendered")
        embedded = json.loads(page.locator("#surveyData").text_content())
        if [item["sample_id"] for item in embedded["items"]] != [row["sample_id"] for row in rows]:
            raise AssertionError("embedded data differs from review manifest")
        if page.locator("details .metric-table tr").count() != expected_samples * 12:
            raise AssertionError("local metric tables are incomplete")
        checks.append(f"{expected_samples} continuous-scroll cards contain the complete score and reason records")
        progress("page_structure_passed")

        for index in (0, expected_samples // 2, expected_samples - 1):
            row = rows[index]
            progress(f"playback_{row['sample_id']}")
            card = page.locator("#" + row["sample_id"])
            card.scroll_into_view_if_needed()
            page.wait_for_function("id=>document.getElementById(id).querySelectorAll('video').length===3", arg=row["sample_id"])
            card.get_by_role("button", name="同步播放 / 暂停").click()
            try:
                page.wait_for_function(
                    "id=>[...document.getElementById(id).querySelectorAll('video')].every(v=>!v.paused&&v.currentTime>.25&&v.videoWidth>0&&!v.error)",
                    arg=row["sample_id"],
                )
            except Exception as error:
                diagnostic = card.locator("video").evaluate_all(
                    "vs=>vs.map(v=>({src:v.getAttribute('src'),paused:v.paused,currentTime:v.currentTime,"
                    "duration:v.duration,readyState:v.readyState,networkState:v.networkState,"
                    "videoWidth:v.videoWidth,videoHeight:v.videoHeight,error:v.error&&v.error.code}))"
                )
                raise RuntimeError(f"three-way playback did not start for {row['sample_id']}: {diagnostic}") from error
            state = card.locator("video").evaluate_all(
                "vs=>vs.map(v=>({time:v.currentTime,width:v.videoWidth,height:v.videoHeight,duration:v.duration}))"
            )
            if max(item["time"] for item in state) - min(item["time"] for item in state) >= .35:
                raise AssertionError("three-way playback is not synchronized")
            card.get_by_role("button", name="同步播放 / 暂停").click()
            playback.append({"sample_id": row["sample_id"], "videos": state})
        checks.append("first/middle/last three-way videos decode and play synchronously")
        progress("playback_passed")

        first = rows[0]
        page.locator("#search").fill(first["sample_id"])
        if page.locator(".card").count() != 1:
            raise AssertionError("search filter failed")
        card = page.locator(".card").first
        card.get_by_role("button", name="1 接受").click()
        notes = '审核备注：逗号, 引号"，换行\n第二行🙂'
        card.locator("textarea").fill(notes)
        with page.expect_download() as download:
            page.locator("#exportCsv").click()
        export = Path(temporary) / "review.csv"
        download.value.save_as(export)
        context.clear_cookies(); page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector(".card")
        page.locator("#importFile").set_input_files(export)
        page.wait_for_function("id=>document.querySelector('#'+id+' textarea').value.length>0", arg=first["sample_id"])
        if page.locator("#" + first["sample_id"] + " textarea").input_value() != notes:
            raise AssertionError("human review CSV roundtrip failed")
        checks.append("human decision and multiline notes survive CSV export/import")
        progress("csv_roundtrip_passed")
        page.locator("#search").fill("")
        page.locator("#subcategory").select_option(first["subcategory"])
        expected = sum(row["subcategory"] == first["subcategory"] for row in rows)
        if page.locator(".card").count() != expected:
            raise AssertionError("subcategory filter failed")
        page.locator("#subcategory").select_option("")
        page.locator("#sort").select_option("delta_desc")
        shown = page.locator(".card").first.get_attribute("id")
        wanted = max(rows, key=lambda row: float(row["delta_sample_total"]))["sample_id"]
        if shown != wanted:
            raise AssertionError("delta sorting failed")
        checks.append("search, subcategory filtering, and paired-delta sorting passed")
        progress("filters_and_sort_passed")
        page.screenshot(path=str(package / "review_preview.png"), full_page=False)
        page.set_viewport_size({"width": 390, "height": 844})
        if not page.evaluate("document.documentElement.scrollWidth <= innerWidth"):
            raise AssertionError("mobile layout overflows horizontally")
        if errors or external:
            raise RuntimeError({"javascript_errors": errors, "external_requests": external})
        progress("offline_and_mobile_passed")
        report = {
            "status": "passed", "browser": browser.version, "transport": "file:// (offline)",
            "sample_count": expected_samples, "video_columns_per_sample": 3, "checks": checks,
            "playback_samples": playback, "javascript_errors": errors, "external_requests": external,
        }
        (package / "browser_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
        progress("complete")


if __name__ == "__main__":
    main()
