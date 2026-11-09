#!/usr/bin/env python3
"""Real Chromium acceptance of the extracted, file://, scroll review package."""

import argparse
import csv
import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with (package / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(rows)
    errors, external_requests = [], []
    checks = []
    with sync_playwright() as playwright, tempfile.TemporaryDirectory(prefix="lance-scroll-test-") as temporary:
        browser = playwright.chromium.launch(channel="chromium", headless=True, timeout=60000)
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external_requests.append(request.url) if request.url.startswith(("http:", "https:")) else None)
        page.goto((package / "index.html").as_uri())
        page.wait_for_selector(".card video")
        assert page.locator(".card").count() == expected
        assert page.locator("#progress").inner_text() == f"人工已标注 0 / {expected}"
        assert page.get_by_role("button", name="下一条", exact=False).count() == 0
        assert page.locator("video").count() <= 12
        checks.append("all cards visible in continuous scroll; no pagination; lazy media; initially unreviewed")
        embedded = page.locator("#surveyData").text_content()
        assert [i["sample_id"] for i in json.loads(embedded)["items"]] == [r["sample_id"] for r in rows]
        for row in rows:
            card = page.locator("#" + row["sample_id"])
            assert card.locator(".reason").text_content() == row["reason"]
            assert card.locator(".instruction").text_content() == row["instruction"]
        checks.append("all IDs, instructions and full Gemini reasons agree with manifest")

        playback = []
        indices = sorted({0, expected // 2, expected - 1})
        for index in indices:
            card = page.locator("#" + rows[index]["sample_id"])
            card.scroll_into_view_if_needed()
            button = card.get_by_role("button", name="同步播放 / 暂停", exact=True)
            button.click()
            page.wait_for_function("id => [...document.getElementById(id).querySelectorAll('video')].length===2 && [...document.getElementById(id).querySelectorAll('video')].every(v=>!v.paused && v.currentTime>.3 && v.videoWidth>0 && !v.error)", arg=rows[index]["sample_id"])
            times = card.locator("video").evaluate_all("vs=>vs.map(v=>({time:v.currentTime,width:v.videoWidth,height:v.videoHeight,duration:v.duration}))")
            assert abs(times[0]["time"] - times[1]["time"]) < .3
            button.click()
            page.wait_for_function("id => [...document.getElementById(id).querySelectorAll('video')].every(v=>v.paused)", arg=rows[index]["sample_id"])
            card.locator("video").first.evaluate("v=>v.currentTime=1")
            page.wait_for_function("id => [...document.getElementById(id).querySelectorAll('video')].every(v=>Math.abs(v.currentTime-1)<.09)", arg=rows[index]["sample_id"])
            card.locator("video").nth(1).evaluate("v=>v.currentTime=2")
            page.wait_for_function("id => [...document.getElementById(id).querySelectorAll('video')].every(v=>Math.abs(v.currentTime-2)<.09)", arg=rows[index]["sample_id"])
            assert page.locator("video").count() <= 12
            playback.append({"sample_id": rows[index]["sample_id"], "videos": times})
        checks.append("first, middle, last pair: H264 playback, synced pause, seek from either side, bounded loaded video count")
        if expected > 5:
            assert page.locator(".card").first.locator("video").count() == 0
            checks.append("offscreen first-pair media unloaded after scrolling to last")

        first_id = rows[0]["sample_id"]
        page.locator("#search").fill(first_id)
        assert page.locator(".card").count() == 1
        first = page.locator(".card").first
        first.get_by_role("button", name="1 接受", exact=True).click()
        notes = '审阅测试：逗号, 引号"，换行\n第二行🙂'
        first.locator("textarea").fill(notes)
        assert page.locator("#progress").inner_text() == f"人工已标注 1 / {expected}"
        with page.expect_download() as download:
            page.locator("#exportCsv").click()
        csv_path = Path(temporary) / "review.csv"
        download.value.save_as(csv_path)
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == expected and exported[0]["human_notes"] == notes
        assert exported[0]["human_decision"] == "accepted"
        assert all(not row["human_decision"] for row in exported[1:])
        first.get_by_role("button", name="0 拒绝", exact=True).click()
        first.locator("textarea").fill("changed")
        page.locator("#importFile").set_input_files(csv_path)
        page.wait_for_function("document.getElementById('saveState').textContent.startsWith('已按 sample_id 导入')")
        assert first.locator("textarea").input_value() == notes
        assert first.locator('[data-decision="accepted"]').get_attribute("class") == "selected"
        with page.expect_download() as download:
            page.locator("#exportJson").click()
        json_path = Path(temporary) / "review.json"
        download.value.save_as(json_path)
        exported_json = json.loads(json_path.read_text())
        assert exported_json["rows"][0]["human_notes"] == notes
        first.get_by_role("button", name="清除结论", exact=True).click()
        page.locator("#importFile").set_input_files(json_path)
        page.wait_for_function("document.getElementById('progress').textContent.startsWith('人工已标注 1')")
        page.reload()
        page.wait_for_selector(".card")
        assert page.locator(".card").first.locator("textarea").input_value() == notes
        assert page.locator("#progress").inner_text() == f"人工已标注 1 / {expected}"
        checks.append("CSV/JSON roundtrip by sample_id; Unicode, commas, quotes, newlines; reload persistence; untouched rows unreviewed")

        page.locator("#category").select_option(rows[0]["category"])
        assert page.locator(".card").count() == sum(r["category"] == rows[0]["category"] for r in rows)
        page.locator("#category").select_option("")
        page.locator("#sort").select_option("low")
        first_score = float(page.locator(".card").first.locator(".overall b").text_content().split(" /")[0])
        assert first_score == min(float(r["overall_score"]) for r in rows)
        page.locator("#search").fill("nonexistent-id-xxx")
        assert page.locator(".card").count() == 0
        page.locator("#search").fill("")
        page.locator("#sort").select_option("index")
        assert page.locator(".card").count() == expected
        checks.append("search/category filters, empty state, overall-score sorting, restore complete list")

        # Exercise keyboard with focus outside native form controls.
        first = page.locator(".card").first
        first.scroll_into_view_if_needed()
        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("2")
        assert first.locator('[data-decision="uncertain"]').get_attribute("class") == "selected"
        page.keyboard.press("Space")
        page.wait_for_function("[...document.querySelector('.card').querySelectorAll('video')].every(v=>!v.paused && v.currentTime>.2)")
        page.keyboard.press("Space")
        page.wait_for_function("[...document.querySelector('.card').querySelectorAll('video')].every(v=>v.paused)")
        checks.append("keyboard 2 verdict and Space synced play/pause")
        context.clear_cookies()
        page.evaluate("localStorage.clear()")
        page.reload()
        page.wait_for_selector(".card video")
        assert page.locator("#progress").inner_text() == f"人工已标注 0 / {expected}"
        page.screenshot(path=str(package / "review_preview.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        checks.append("mobile width has no horizontal page overflow")
        assert not errors, errors
        assert not external_requests, external_requests
        report = {"status": "passed", "browser": browser.version, "transport": "file:// (offline)",
                  "sample_count": expected, "checks": checks, "playback_samples": playback,
                  "javascript_errors": errors, "external_requests": external_requests}
        (package / "browser_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        browser.close()


if __name__ == "__main__":
    main()
