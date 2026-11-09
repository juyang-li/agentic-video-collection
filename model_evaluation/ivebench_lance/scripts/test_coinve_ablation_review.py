#!/usr/bin/env python3
"""Real-Chrome acceptance test for the offline CoinVE ablation review."""

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
    expected = len(rows)
    if expected < 1:
        raise ValueError("empty review")
    errors: list[str] = []
    external: list[str] = []
    playback = []
    checks = []
    with sync_playwright() as playwright, tempfile.TemporaryDirectory(prefix="coinve-review-") as temporary:
        progress("launch_browser")
        chrome = Path(__file__).resolve().parents[1] / "runtime/google-chrome-stable/opt/google/chrome/chrome"
        options = {"headless": True, "timeout": 60_000}
        if chrome.exists():
            options["executable_path"] = str(chrome)
        else:
            options["channel"] = "chromium"
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(viewport={"width": 1880, "height": 1100}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url) if request.url.startswith(("http:", "https:")) else None)
        progress("load_offline_page")
        page.goto((package / "index.html").as_uri(), wait_until="domcontentloaded")
        page.wait_for_selector(".card")
        if page.locator(".card").count() != expected:
            raise AssertionError("not all review cards rendered")
        if page.locator("tr.pair-row").count() != expected * 6:
            raise AssertionError("pairwise consensus tables are incomplete")
        embedded = json.loads(page.locator("#surveyData").text_content())
        if [item["sample_id"] for item in embedded["items"]] != [row["sample_id"] for row in rows]:
            raise AssertionError("embedded records differ from review manifest")
        if page.locator(".aggregate").evaluate("x=>x.open"):
            raise AssertionError("aggregate conclusions must be collapsed by default")
        header_position = page.locator("header").evaluate("x=>getComputedStyle(x).position")
        if header_position in ("sticky", "fixed"):
            raise AssertionError("header must scroll away with the document")
        page.set_viewport_size({"width": 1200, "height": 900})
        widths = page.locator(".card").first.locator(".video-wrap").evaluate_all("xs=>xs.slice(0,2).map(x=>x.getBoundingClientRect().width)")
        if max(widths) / min(widths) > 1.05:
            raise AssertionError("Source must use the same display width as candidate videos")
        page.set_viewport_size({"width": 1880, "height": 1100})
        checks.append(f"{expected} cards contain five media columns and six complete pairwise records")
        checks.append("header scrolls away, aggregate conclusions start collapsed, and Source is not enlarged")
        progress("page_structure_passed")

        for index in sorted(set((0, expected // 2, expected - 1))):
            row = rows[index]
            progress("playback_" + row["sample_id"])
            card = page.locator("#" + row["sample_id"])
            card.scroll_into_view_if_needed()
            page.wait_for_function("id=>document.getElementById(id).querySelectorAll('video').length===5", arg=row["sample_id"])
            card.get_by_role("button", name="五路同步播放 / 暂停").click()
            try:
                page.wait_for_function(
                    "id=>[...document.getElementById(id).querySelectorAll('video')].every(v=>!v.paused&&v.currentTime>.2&&v.videoWidth>0&&!v.error)",
                    arg=row["sample_id"],
                )
            except Exception as error:
                diagnostic = card.locator("video").evaluate_all(
                    "vs=>vs.map(v=>({src:v.getAttribute('src'),paused:v.paused,currentTime:v.currentTime,duration:v.duration,readyState:v.readyState,videoWidth:v.videoWidth,videoHeight:v.videoHeight,error:v.error&&v.error.code}))"
                )
                raise RuntimeError(f"five-way playback failed for {row['sample_id']}: {diagnostic}") from error
            state = card.locator("video").evaluate_all(
                "vs=>vs.map(v=>({time:v.currentTime,width:v.videoWidth,height:v.videoHeight,duration:v.duration}))"
            )
            if max(value["time"] for value in state) - min(value["time"] for value in state) >= .4:
                raise AssertionError("five-way playback is not synchronized")
            card.get_by_role("button", name="五路同步播放 / 暂停").click()
            playback.append({"sample_id": row["sample_id"], "videos": state})
        checks.append("first/middle/last five-way videos decode and play synchronously")
        progress("playback_passed")

        first = rows[0]
        page.locator("#search").fill(first["sample_id"])
        if page.locator(".card").count() != 1:
            raise AssertionError("search failed")
        card = page.locator(".card").first
        card.get_by_role("button", name="0 不通过").first.click()
        card.get_by_role("button", name="CoinVE-4K").click()
        notes = '审核备注：逗号, 引号"，换行\n第二行🙂'
        card.locator("textarea").fill(notes)
        with page.expect_download() as download:
            page.locator("#exportCsv").click()
        exported = Path(temporary) / "review.csv"
        download.value.save_as(exported)
        page.evaluate("localStorage.clear()")
        page.reload(); page.wait_for_selector(".card")
        page.locator("#importFile").set_input_files(exported)
        page.wait_for_function("id=>document.querySelector('#'+id+' textarea').value.length>0", arg=first["sample_id"])
        card = page.locator("#" + first["sample_id"])
        if card.locator("textarea").input_value() != notes:
            raise AssertionError("CSV notes roundtrip failed")
        if card.locator(".audit button.active").first.text_content() != "0 不通过":
            raise AssertionError("CSV tri-state roundtrip failed")
        if card.locator(".best button.active").text_content() != "CoinVE-4K":
            raise AssertionError("CSV best-model roundtrip failed")
        checks.append("default tri-state fields and multiline CSV export/import passed")
        progress("csv_roundtrip_passed")

        page.locator("#search").fill("")
        page.locator("#stage").select_option(first["stage"])
        if page.locator(".card").count() != sum(row["stage"] == first["stage"] for row in rows):
            raise AssertionError("stage filter failed")
        page.locator("#stage").select_option("")
        page.locator("#subcategory").select_option(first["subcategory"])
        if page.locator(".card").count() != sum(row["subcategory"] == first["subcategory"] for row in rows):
            raise AssertionError("subcategory filter failed")
        page.locator("#subcategory").select_option("")
        page.locator("#consistency").select_option("inconsistent")
        if page.locator(".card").count() < 1:
            raise AssertionError("consistency filter failed")
        page.locator("#consistency").select_option("")
        page.locator("#sort").select_option("inconsistent")
        checks.append("search, subset, subcategory, consistency filters and sorting passed")
        progress("filters_passed")
        page.screenshot(path=str(package / "review_preview.png"), full_page=False)
        page.set_viewport_size({"width": 390, "height": 844})
        if not page.evaluate("document.documentElement.scrollWidth <= innerWidth"):
            raise AssertionError("mobile layout overflows horizontally")
        if errors or external:
            raise RuntimeError({"javascript_errors": errors, "external_requests": external})
        report = {
            "status": "passed", "browser": browser.version, "transport": "file:// (offline)",
            "sample_count": expected, "video_columns_per_sample": 5,
            "pairwise_comparisons_per_sample": 6, "checks": checks,
            "playback_samples": playback, "javascript_errors": errors, "external_requests": external,
        }
        (package / "browser_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
        progress("complete")


if __name__ == "__main__":
    main()
