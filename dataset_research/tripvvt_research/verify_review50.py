#!/usr/bin/env python3
"""Validate every TripVVT review triplet and numeric audit round trips offline."""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
NAME='tripvvt_reference50_20261108'
BUNDLE=ROOT/NAME/NAME
RUNTIME=ROOT.parents[1]/'model_evaluation/ivebench_lance/runtime'


def ready(page):
    page.wait_for_function("[...document.querySelectorAll('#media video')].every(v=>v.readyState>=2&&!v.error)&&[...document.querySelectorAll('#media img')].every(i=>i.complete&&i.naturalWidth>0)")
    page.evaluate('document.fonts.ready')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--expected-count',type=int,default=50)
    args=parser.parse_args();bundle=args.bundle.resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    cards=manifest['cards'];external=[];errors=[];played=0;loaded=0;checks=[]
    count=args.expected_count;captured=set()
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(RUNTIME/'debs/extracted/usr/lib/x86_64-linux-gnu')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    with sync_playwright() as p, tempfile.TemporaryDirectory(prefix='tripvvt-audit-') as temporary:
        browser=p.chromium.launch(executable_path=str(RUNTIME/'google-chrome-stable/opt/google/chrome/chrome'),headless=True,env=env,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1500,'height':1200},accept_downloads=True)
        page=context.new_page()
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((bundle/'index.html').as_uri());ready(page)
        assert f'已审核 0 / {count}' in page.locator('#progress').inner_text()
        assert len(cards)==count and len({c['provenance']['target_member'] for c in cards})==count
        for i,card in enumerate(cards):
            ready(page)
            assert page.locator('#position').inner_text()==f'{i+1} / {count}'
            assert page.locator('#heading').inner_text().startswith(card['id'])
            assert page.locator('#instructionEn').inner_text()==card['instruction_en']
            assert page.locator('#instructionZh').inner_text()==card['instruction_zh']
            assert page.locator('#media video').count()==2 and page.locator('#media img').count()==1
            assert page.locator('#media img').get_attribute('src')==card['panels'][1]['path']
            assert page.locator('#media video').nth(0).get_attribute('src')==card['panels'][0]['path']
            assert page.locator('#media video').nth(1).get_attribute('src')==card['panels'][2]['path']
            assert page.locator('#media a').count()==3
            assert not page.locator('#reviewed').is_checked()
            if 'low_change_rank' in card:
                assert card['low_change_rank']==i+1
                assert not page.locator('#similarityDetails').evaluate('x=>x.open')
                assert json.loads(page.locator('#similarityValues').text_content())==card['similarity']
                assert page.locator('#instructionZh').bounding_box()['y']<page.locator('#media').bounding_box()['y']
                assert page.locator('#officialCaption').text_content()==card['provenance']['official_caption']
            for field in manifest['fields']:
                assert page.locator(f'[data-field="{field}"]').input_value()=='1'
            page.click('#playAll')
            page.wait_for_function("[...document.querySelectorAll('#media video')].every(v=>!v.paused&&v.currentTime>.12)")
            times=page.locator('#media video').evaluate_all('vs=>vs.map(v=>v.currentTime)')
            assert abs(times[0]-times[1])<.25
            page.click('#pauseAll')
            page.wait_for_function("[...document.querySelectorAll('#media video')].every(v=>v.paused)")
            if card['category'] not in captured:
                for index,target_time in [(0,.7),(1,1.3)]:
                    page.locator('#media video').nth(index).evaluate('(v,t)=>v.currentTime=t',target_time)
                    page.wait_for_function("t=>[...document.querySelectorAll('#media video')].every(v=>Math.abs(v.currentTime-t)<.1)",arg=target_time)
                    page.wait_for_timeout(100)
                page.screenshot(path=str(bundle/f'preview_{card["category"]}.png'),full_page=True)
                captured.add(card['category'])
            played+=2;loaded+=1
            if i<count-1:page.click('#next')
        assert f'已审核 0 / {count}' in page.locator('#progress').inner_text()
        checks.append(f'all {count} triplets: correct source/reference/target and instructions, {played} real synchronized plays, {loaded} reference images, defaults remain unreviewed')
        for category in dict.fromkeys(c['category'] for c in cards):
            category_count=sum(c['category']==category for c in cards)
            page.select_option('#filter',category);ready(page)
            assert page.locator('#position').inner_text()==f'1 / {category_count}'
            assert all(c['category']==category for c in page.evaluate('visible'))
            shown=page.evaluate('visible')
            if manifest.get('selection_policy')=='roi_mae_ascending':
                values=[(c['similarity']['roi_mae_median'],c['id']) for c in shown]
                assert values==sorted(values)
            else:
                ids=[c['id'] for c in shown];assert ids==sorted(ids)
        page.select_option('#filter','all');ready(page)
        checks.append('category filters and ordering match manifest, synchronized pause and seeking from both sides')
        notes='参考不匹配，逗号, 引号"\n第二行测试🙂'
        page.select_option('[data-field="reference_match"]','0')
        page.select_option('[data-field="edit_visible"]','2')
        page.fill('[data-field="notes"]',notes);page.locator('[data-field="notes"]').blur();page.check('#reviewed')
        with page.expect_download() as event:page.click('#exportJson')
        jp=Path(temporary)/'review.json';event.value.save_as(jp)
        data=json.loads(jp.read_text());assert len(data['reviews'])==count
        assert data['reviews'][0]['reference_match']=='0' and data['reviews'][0]['edit_visible']=='2'
        assert data['reviews'][0]['notes']==notes and sum(bool(r['reviewed']) for r in data['reviews'])==1
        with page.expect_download() as event:page.click('#exportCsv')
        cp=Path(temporary)/'review.csv';event.value.save_as(cp)
        rows=list(csv.DictReader(cp.open(encoding='utf-8-sig')))
        assert len(rows)==count and rows[0]['notes']==notes and rows[0]['reference_match']=='0'
        for control,path in [('#importJson',jp),('#importCsv',cp)]:
            page.select_option('[data-field="reference_match"]','1');page.set_input_files(control,path)
            page.wait_for_function("document.querySelector('[data-field=reference_match]').value==='0'")
            assert page.locator('[data-field="notes"]').input_value()==notes
            assert page.locator('[data-field="edit_visible"]').input_value()=='2'
            assert page.locator('#reviewed').is_checked()
        page.reload();ready(page)
        assert page.locator('[data-field="reference_match"]').input_value()=='0'
        page.evaluate('localStorage.clear()');page.reload();ready(page)
        assert not page.locator('#reviewed').is_checked() and page.locator('[data-field="reference_match"]').input_value()=='1'
        page.evaluate('document.activeElement.blur()');page.keyboard.press('ArrowRight');ready(page)
        assert page.locator('#position').inner_text()==f'2 / {count}'
        page.keyboard.press('ArrowLeft');ready(page);assert page.locator('#position').inner_text()==f'1 / {count}'
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not external and not errors,(external,errors)
        checks.append('CSV/JSON 0/1/2 and reviewed round trips, quoted multiline Unicode notes, reload persistence, clean defaults, keyboard navigation, mobile fit, zero external requests')
        report=dict(status='passed',cards_checked=count,video_instances_played=played,reference_images_loaded=loaded,
            external_requests=len(external),javascript_errors=errors,checks=checks,browser=browser.version,
            index_sha256=hashlib.sha256((bundle/'index.html').read_bytes()).hexdigest())
        (bundle/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True);browser.close()


if __name__=='__main__':main()
