#!/usr/bin/env python3
"""Real Chromium acceptance of the extracted three-way Omni comparison package."""

import argparse
import csv
import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 240:
        raise ValueError("browser acceptance requires the full 240")
    errors, external, checks, playback = [], [], [], []
    with sync_playwright() as playwright, tempfile.TemporaryDirectory(prefix="lance-omni-compare-") as temporary:
        print("browser: launching Chromium", flush=True)
        browser = playwright.chromium.launch(channel="chromium", headless=True, timeout=60000)
        context = browser.new_context(viewport={"width": 1680, "height": 1100}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(30000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url) if request.url.startswith(("http:", "https:")) else None)
        print("browser: opening offline review", flush=True)
        page.goto((package / "index.html").as_uri(), wait_until="domcontentloaded")
        page.wait_for_selector(".card video")
        assert page.locator(".card").count() == 240
        assert page.get_by_role("button", name="下一条", exact=False).count() == 0
        assert page.locator("#progress").inner_text() == "人工已标注 0 / 240"
        assert page.locator("video").count() <= 18
        embedded = json.loads(page.locator("#surveyData").text_content())
        assert [item["sample_id"] for item in embedded["items"]] == [row["sample_id"] for row in rows]
        rendered = page.locator(".card").evaluate_all(
            "cards=>cards.map(card=>({id:card.id,instruction:card.querySelector('.instruction').textContent,"
            "baseReason:card.querySelector('.base-reason').textContent,"
            "finetuneReason:card.querySelector('.finetune-reason').textContent,"
            "scoreCount:card.querySelectorAll('.score').length}))"
        )
        expected = [
            {"id": row["sample_id"], "instruction": row["instruction"],
             "baseReason": row["base_reason"], "finetuneReason": row["finetune_reason"],
             "scoreCount": 15}
            for row in rows
        ]
        assert rendered == expected
        checks.append("240 continuous-scroll cards contain both complete scores/reasons and five deltas")
        print("browser: 240 cards and paired judge data passed", flush=True)
        for index in (0, 120, 239):
            row = rows[index]
            card = page.locator("#" + row["sample_id"])
            card.scroll_into_view_if_needed()
            card.get_by_role("button", name="同步播放 / 暂停", exact=True).click()
            page.wait_for_function("id=>{const v=[...document.getElementById(id).querySelectorAll('video')];return v.length===3&&v.every(x=>!x.paused&&x.currentTime>.3&&x.videoWidth>0&&!x.error)}", arg=row["sample_id"])
            state = card.locator("video").evaluate_all("vs=>vs.map(v=>({time:v.currentTime,width:v.videoWidth,height:v.videoHeight,duration:v.duration}))")
            assert max(item["time"] for item in state) - min(item["time"] for item in state) < .3
            card.get_by_role("button", name="同步播放 / 暂停", exact=True).click()
            page.wait_for_function("id=>[...document.getElementById(id).querySelectorAll('video')].every(v=>v.paused)", arg=row["sample_id"])
            card.locator("video").nth(2).evaluate("v=>v.currentTime=1")
            page.wait_for_function("id=>[...document.getElementById(id).querySelectorAll('video')].every(v=>Math.abs(v.currentTime-1)<.09)", arg=row["sample_id"])
            playback.append({"sample_id": row["sample_id"], "videos": state})
            assert page.locator("video").count() <= 18
        assert page.locator(".card").first.locator("video").count() == 0
        checks.append("first/middle/last three-way playback, pause, seek synchronization, and lazy unloading passed")
        print("browser: first/middle/last synchronized playback passed", flush=True)
        first = rows[0]
        page.locator("#search").fill(first["sample_id"])
        assert page.locator(".card").count() == 1
        card = page.locator(".card").first
        card.get_by_role("button", name="1 接受", exact=True).click()
        notes = '对比审核：逗号, 引号"，换行\n第二行🙂'
        card.locator("textarea").fill(notes)
        with page.expect_download() as download:
            page.locator("#exportCsv").click()
        csv_path = Path(temporary) / "review.csv"
        download.value.save_as(csv_path)
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == 240 and exported[0]["human_decision"] == "accepted" and exported[0]["human_notes"] == notes
        card.get_by_role("button", name="0 拒绝", exact=True).click()
        page.locator("#importFile").set_input_files(csv_path)
        page.wait_for_function("document.getElementById('saveState').textContent.startsWith('已按 sample_id 导入')")
        assert card.locator("textarea").input_value() == notes
        page.reload(); page.wait_for_selector(".card")
        assert page.locator(".card").first.locator("textarea").input_value() == notes
        checks.append("human decision/notes CSV roundtrip and browser persistence passed")
        print("browser: annotation export/import and persistence passed", flush=True)
        page.locator("#category").select_option(first["category"])
        assert page.locator(".card").count() == sum(row["category"] == first["category"] for row in rows)
        page.locator("#category").select_option("")
        page.locator("#search").fill("")
        page.locator("#sort").select_option("delta_low")
        displayed = float(page.locator(".card").first.locator(".delta-row .score").first.locator("b").text_content())
        assert displayed == round(min(float(row["delta_overall_score"]) for row in rows), 2)
        page.locator("#sort").select_option("delta_high")
        displayed = float(page.locator(".card").first.locator(".delta-row .score").first.locator("b").text_content().replace("+", ""))
        assert displayed == round(max(float(row["delta_overall_score"]) for row in rows), 2)
        checks.append("category filter and paired-difference sorting passed")
        print("browser: filtering and score-difference sorting passed", flush=True)
        context.clear_cookies(); page.evaluate("localStorage.clear()"); page.reload(); page.wait_for_selector(".card video")
        assert page.locator("#progress").inner_text() == "人工已标注 0 / 240"
        page.screenshot(path=str(package / "review_preview.png"), full_page=False)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        if errors or external:
            raise RuntimeError({"javascript_errors": errors, "external_requests": external})
        report = {"status": "passed", "browser": browser.version, "transport": "file:// (offline)",
                  "sample_count": 240, "video_columns_per_sample": 3, "checks": checks,
                  "playback_samples": playback, "javascript_errors": errors, "external_requests": external}
        (package / "browser_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        browser.close()


if __name__ == "__main__":
    main()
