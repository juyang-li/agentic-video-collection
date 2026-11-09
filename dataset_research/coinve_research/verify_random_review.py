#!/usr/bin/env python3
"""Exercise the real offline CoinVE reviewer, including media and label roundtrips."""
import argparse
import csv
import json
import hashlib
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--chrome", type=Path, help="Portable Chrome executable")
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--play-all", action="store_true", help="Exercise playback on every pair")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    rows = list(csv.DictReader((bundle / "review_manifest.csv").open(encoding="utf-8-sig")))
    errors, external, checks = [], [], []
    with sync_playwright() as p, tempfile.TemporaryDirectory(prefix="coinve-review-test-") as temp:
        launch = dict(headless=True, timeout=30000)
        if args.chrome:
            launch.update(executable_path=str(args.chrome.resolve()), args=["--no-sandbox"])
        else:
            launch["channel"] = "chromium"
        browser = p.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: external.append(r.url) if r.url.startswith(("http:", "https:")) else None)
        page.goto((bundle / "index.html").as_uri())
        page.wait_for_selector("#sourceVideo")
        embedded = json.loads(page.locator("#surveyData").text_content())
        assert len(rows) == len(embedded["items"]) == args.expected_count
        assert page.locator("#progress").get_attribute("value") == "0"
        assert page.locator("[data-decision].active").count() == 0
        for i, row in enumerate(rows):
            assert page.locator("#sampleId").text_content() == row["sample_id"]
            instructions = json.loads(row["instructions_json"])
            assert page.locator("#instructions li > span:first-child").all_text_contents() == instructions
            if "task_types_json" in row:
                assert embedded["items"][i]["task_labels"] == json.loads(row["task_types_json"])
                assert page.locator("#taskGroup").text_content() == "展示组：" + row["branch"]
            if "comparison_group" in row:
                assert page.locator('#ratioGroup').text_content()==row['branch']
                assert embedded['items'][i]['comparison_group']==row['comparison_group']
                r=embedded['items'][i]['ratio_info']
                assert abs(r['ratio']-max(r['frame_ratio'],r['duration_ratio']))<1e-9
                assert f"{r['ratio']:.6f}×" in page.locator('#ratioDescription').text_content()
                assert len(instructions) in (2,3) and not any('styl' in op.lower() for op in embedded['items'][i]['operations'])
            assert page.locator("video").count() == 2
            for role in ("source", "target"):
                assert page.locator(f"#{role}Video").get_attribute("src") == row[f"{role}_media"]
                assert page.locator(f"#{role}Full").get_attribute("href") == row[f"{role}_media"]
            if i < len(rows)-1:
                page.locator("#next").click()
        assert page.locator("#progress").get_attribute("value") == "0"
        checks.append(f"{len(rows)} fixed IDs, all subinstructions, correct source/target links; skip without marking; only two video elements")
        if 'comparison_group' in rows[0]:
            starts=[(i,r['comparison_group']) for i,r in enumerate(rows) if i==0 or r['comparison_group']!=rows[i-1]['comparison_group']]
            for start,group in starts:
                page.select_option('#groupJump',str(start))
                assert page.locator('#sampleId').text_content()==rows[start]['sample_id']
                assert embedded['items'][start]['comparison_group']==group
            checks.append('all ratio-group jumps and explicit ratio descriptions match manifest')

        def jump(index):
            page.locator("#jumpIndex").fill(str(index+1))
            page.locator("#jump").click()
            page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.readyState>=2 && v.videoWidth>0)")

        selected = set(range(len(rows))) if args.play_all else {0, len(rows)//2, len(rows)-1}
        for count in {int(r["instruction_count"]) for r in rows}:
            selected.add(next(i for i,r in enumerate(rows) if int(r["instruction_count"]) == count))
        selected.add(next((i for i,r in enumerate(rows) if abs(float(r["source_duration"])-float(r["target_duration"]))>.05), 0))
        playback = []
        for index in sorted(selected):
            jump(index)
            page.locator("#play").click()
            page.wait_for_function("[...document.querySelectorAll('video')].every(v=>!v.paused && v.currentTime>.25 && !v.error)")
            times = page.locator("video").evaluate_all("vs=>vs.map(v=>({time:v.currentTime,duration:v.duration,width:v.videoWidth,height:v.videoHeight}))")
            assert abs(times[0]["time"]-times[1]["time"]) < .25
            page.locator("#play").click()
            page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.paused)")
            end = min(t["duration"] for t in times)
            t1, t2 = min(.8, end/4), min(1.5, end/2)
            page.locator("#sourceVideo").evaluate("(v,t)=>v.currentTime=t", t1)
            page.wait_for_function("t=>[...document.querySelectorAll('video')].every(v=>Math.abs(v.currentTime-t)<.08)", arg=t1)
            page.wait_for_timeout(100)
            page.locator("#targetVideo").evaluate("(v,t)=>v.currentTime=t", t2)
            page.wait_for_function("t=>[...document.querySelectorAll('video')].every(v=>Math.abs(v.currentTime-t)<.08)", arg=t2)
            assert not page.locator("#mediaError").text_content()
            playback.append({"sample_id": rows[index]["sample_id"], "videos": times})
        checks.append("first/middle/last, every present task count and duration-mismatch pairs: real H264 playback, synchronized pause and seek from both sides")

        jump(0)
        page.locator('[data-decision="1"]').click()
        notes = '测试备注：逗号, 引号"与换行\n第二行🙂'
        page.locator("#notes").fill(notes)
        assert page.locator("#progress").get_attribute("value") == "1"
        with page.expect_download() as event:
            page.locator("#exportCsv").click()
        csv_path = Path(temp)/"review.csv"
        event.value.save_as(csv_path)
        exported = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        assert len(exported) == len(rows) and exported[0]["notes"] == notes
        assert exported[0]["human_decision"] == "1" and exported[0]["reviewed"] == "1"
        assert all(r["human_decision"] == "1" and r["reviewed"] == "0" for r in exported[1:])
        page.locator('[data-decision="0"]').click()
        page.locator("#notes").fill("changed")
        page.locator("#importFile").set_input_files(csv_path)
        page.wait_for_function("document.getElementById('saveState').textContent.startsWith('已按 sample_id 导入')")
        assert page.locator("#notes").input_value() == notes
        assert page.locator('[data-decision="1"]').get_attribute("class") == "active"
        with page.expect_download() as event:
            page.locator("#exportJson").click()
        json_path = Path(temp)/"review.json"
        event.value.save_as(json_path)
        page.locator("#clear").click()
        assert page.locator("#progress").get_attribute("value") == "0"
        page.locator("#importFile").set_input_files(json_path)
        page.wait_for_function("document.getElementById('progress').value===1")
        page.reload()
        page.wait_for_selector("#sampleId")
        assert page.locator("#notes").input_value() == notes
        assert page.locator("#progress").get_attribute("value") == "1"
        checks.append("CSV/JSON imports preserve numeric verdict, reviewed flag and notes including Unicode/commas/quotes/newlines; reload persistence; defaults not counted")

        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("2")
        assert page.locator('[data-decision="2"]').get_attribute("class") == "active"
        page.keyboard.press("ArrowRight")
        assert page.locator("#sampleId").text_content() == rows[1]["sample_id"]
        page.keyboard.press("ArrowLeft")
        assert page.locator("#sampleId").text_content() == rows[0]["sample_id"]
        page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.readyState>=2)")
        page.keyboard.press("Space")
        page.wait_for_function("[...document.querySelectorAll('video')].every(v=>!v.paused && v.currentTime>.2)")
        page.keyboard.press("Space")
        page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.paused)")
        checks.append("keyboard verdict, navigation and synchronized playback")
        context.close()

        context = browser.new_context(viewport={"width": 1440, "height": 1050})
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: external.append(r.url) if r.url.startswith(("http:", "https:")) else None)
        page.goto((bundle / "index.html").as_uri())
        page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.readyState>=2)")
        page.locator("#play").click()
        page.wait_for_function("document.getElementById('sourceVideo').currentTime>.25")
        page.locator("#play").click()
        page.screenshot(path=str(bundle / "review_preview.png"), full_page=True)
        assert page.locator("#progress").get_attribute("value") == "0"
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        checks.append("fresh browser starts unreviewed; desktop screenshot and mobile width fit; no external webpage requests")
        assert not errors, errors
        assert not external, external
        report = {"status":"passed", "sample_count":len(rows), "browser":browser.version, "transport":"file://",
                  "checks":checks, "playback_samples":playback, "javascript_errors":errors, "external_requests":external,
                  "index_sha256":hashlib.sha256((bundle/'index.html').read_bytes()).hexdigest()}
        (bundle / "browser_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps(report,ensure_ascii=False),flush=True)
        browser.close()


if __name__ == "__main__":
    main()
