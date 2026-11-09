#!/usr/bin/env python3
"""Exercise every review card and both audit round trips in offline Chromium."""
import csv
import io
import hashlib
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
NAME='ref_edit_clothing25_motion25_review_20261030'
PACKAGE=ROOT/'review'/NAME
EVAL=ROOT.parents[1]/'model_evaluation/ivebench_lance/runtime'
CHROME=EVAL/'google-chrome-stable/opt/google/chrome/chrome'
LIBS=EVAL/'debs/extracted/usr/lib/x86_64-linux-gnu'


def ready(page):
    page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.error) && [...document.querySelectorAll('#media img')].every(i=>i.complete&&i.naturalWidth>0)",timeout=30000)
    page.evaluate('document.fonts.ready')
    assert page.evaluate("document.fonts.check('15px ReviewCJK','换衣动作参考审核')")


def main():
    external=[];errors=[];download_dir=ROOT/'metadata/browser_checks';download_dir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(LIBS)+( ':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(CHROME),headless=True,env=env)
        context=browser.new_context(viewport={'width':1500,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((PACKAGE/'index.html').as_uri(),wait_until='load');ready(page)
        assert page.locator('#position').inner_text()=='1 / 50'
        checked=0;video_instances=0;image_instances=0
        for i in range(50):
            ready(page)
            assert page.locator('#position').inner_text()==f'{i+1} / 50'
            assert page.locator('#media video').count()==2
            expected_images=1 if i<25 else 0
            assert page.locator('#media img').count()==expected_images
            if i>=25:
                assert '无结果 GT' in page.locator('#meta').inner_text()
                assert page.locator('select[data-field="instruction_correctness"]').is_disabled()
                assert page.locator('select[data-field="content_preservation"]').is_disabled()
            page.click('#playAll')
            page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.currentTime>0)",timeout=15000)
            page.click('#pauseAll')
            assert page.evaluate("[...document.querySelectorAll('video')].every(v=>v.paused)")
            if i in [0,25]:
                page.evaluate('window.scrollTo(0,0)')
                page.screenshot(path=str(download_dir/f'card_{i+1:02d}.png'),full_page=True)
            checked+=1;video_instances+=2;image_instances+=expected_images
            if i<49:page.click('#next')
        page.select_option('#filter','garment');ready(page);assert page.locator('#position').inner_text()=='1 / 25'
        page.select_option('#filter','motion');ready(page);assert page.locator('#position').inner_text()=='1 / 25'
        assert page.locator('#media img').count()==0
        page.select_option('#filter','all');ready(page)
        notes='审核往返测试，逗号, 双引号"\n第二行'
        page.select_option('[data-field="reference_quality"]','0')
        page.fill('[data-field="notes"]',notes);page.locator('[data-field="notes"]').blur();page.check('#reviewed')
        with page.expect_download() as dl:page.click('#exportJson')
        json_path=download_dir/'roundtrip.json';dl.value.save_as(str(json_path))
        data=json.loads(json_path.read_text());assert len(data['reviews'])==50
        assert data['reviews'][0]['reference_quality']=='0' and data['reviews'][0]['notes']==notes
        assert data['reviews'][0]['reviewed'] is True and sum(bool(r['reviewed']) for r in data['reviews'])==1
        with page.expect_download() as dl:page.click('#exportCsv')
        csv_path=download_dir/'roundtrip.csv';dl.value.save_as(str(csv_path))
        rows=list(csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8-sig'))));assert len(rows)==50 and rows[0]['notes']==notes
        page.select_option('[data-field="reference_quality"]','2');page.set_input_files('#importJson',str(json_path))
        page.wait_for_function("document.querySelector('[data-field=reference_quality]').value==='0'")
        page.select_option('[data-field="reference_quality"]','2');page.set_input_files('#importCsv',str(csv_path))
        page.wait_for_function("document.querySelector('[data-field=reference_quality]').value==='0'")
        assert page.locator('[data-field="notes"]').input_value()==notes
        page.reload();ready(page);assert page.locator('[data-field="reference_quality"]').input_value()=='0'
        page.evaluate('localStorage.clear()');page.reload();ready(page)
        assert page.locator('[data-field="reference_quality"]').input_value()=='1' and not page.locator('#reviewed').is_checked()
        assert not external and not errors,(external,errors)
        browser.close()
    report={'status':'passed','cards_checked':checked,'video_instances_played':video_instances,
            'garment_images_loaded':image_instances,'external_requests':len(external),'javascript_errors':errors,
            'csv_roundtrip':True,'json_roundtrip':True,'quoted_multiline_notes_roundtrip':True,
            'local_storage_reload':True,'category_filter':True,'no_gt_metrics_disabled':True,
            'default_labels_reset_passed':True,'all_video_pause_controls':True}
    report['embedded_chinese_font_loaded']=True
    report['index_sha256']=hashlib.sha256((PACKAGE/'index.html').read_bytes()).hexdigest()
    (PACKAGE/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
