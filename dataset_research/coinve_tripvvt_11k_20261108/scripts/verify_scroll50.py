#!/usr/bin/env python3
"""Offline scroll-layout and complete prompt/media validation for all 50 pairs."""
import hashlib
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
NAME='coinve40_tripvvt10_scroll50_20261108'
BUNDLE=ROOT/'reviews'/NAME/NAME
RUNTIME=ROOT.parents[1]/'model_evaluation/ivebench_lance/runtime'


def main():
    rows=[json.loads(l) for l in (BUNDLE/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==50 and sum(r['dataset']=='coinve' for r in rows)==40 and sum(r['dataset']=='tripvvt' for r in rows)==10
    for r in rows:
        for role in ('source','target','reference'):
            if not r.get(role):continue
            for key in (role,role+'_original'):
                path=BUNDLE/r[key];assert path.is_file() and not Path(r[key]).is_absolute()
                expected=r[key+'_sha256'] if key.endswith('_original') else r[role+'_sha256']
                with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(RUNTIME/'debs/extracted/usr/lib/x86_64-linux-gnu')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    errors=[];external=[];captures=set();played=0;refs=0;positions=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(RUNTIME/'google-chrome-stable/opt/google/chrome/chrome'),headless=True,env=env,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1500,'height':1100})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.goto((BUNDLE/'index.html').as_uri());page.evaluate('document.fonts.ready')
        assert page.locator('section').count()==50
        assert page.locator('input,select,textarea').count()==0
        assert page.locator('button').count()==100
        assert page.locator('section').evaluate_all("cards=>cards.every(c=>getComputedStyle(c).display!=='none'&&c.getBoundingClientRect().height>0)")
        for i,r in enumerate(rows):
            card=page.locator('section').nth(i);card.scroll_into_view_if_needed();positions.append(card.evaluate('el=>el.offsetTop'))
            assert r['sample_id'] in card.locator('h2').text_content()
            assert card.locator('.prompt').text_content()==r['prompt']
            assert card.locator('.prompt').evaluate("el=>getComputedStyle(el).whiteSpace==='pre-wrap'&&el.scrollHeight<=el.clientHeight+1")
            assert card.locator('video').count()==2
            if r['dataset']=='tripvvt':
                assert card.locator('.caption pre').text_content()==r['caption']==r['original_metadata']['text']
                assert 'reference image' in card.locator('.prompt').text_content()
                assert card.locator('.caption').bounding_box()['y']<card.locator('.panels').bounding_box()['y']
                assert card.locator('img').get_attribute('src')==r['reference']
                card.locator('img').evaluate('im=>im.decode()');refs+=1
            for index,role in enumerate(('source','target')):assert card.locator('video').nth(index).get_attribute('src')==r[role]
            card.locator('.play').click()
            card.locator('video').evaluate_all("vs=>Promise.all(vs.map(v=>new Promise((resolve,reject)=>{let timer=setTimeout(()=>reject('play timeout '+v.currentSrc),45000);function check(){if(v.error){clearTimeout(timer);reject('media error')}else if(!v.paused&&v.currentTime>.15){clearTimeout(timer);resolve()}else setTimeout(check,60)}check()})))")
            times=card.locator('video').evaluate_all('vs=>vs.map(v=>v.currentTime)');assert abs(times[0]-times[1])<.5
            card.locator('.pause').click();assert card.locator('video').evaluate_all('vs=>vs.every(v=>v.paused)');played+=2
            card.locator('video').nth(0).evaluate('(v,end)=>{v.currentTime=end;v.dispatchEvent(new Event("timeupdate"))}',r['common_end_seconds'])
            assert card.locator('video').evaluate_all('vs=>vs.every(v=>v.paused)')
            if r['dataset'] not in captures:
                card.screenshot(path=str(BUNDLE/f"preview_{r['dataset']}.png"));captures.add(r['dataset'])
            print('scroll card passed',i+1,r['sample_id'],flush=True)
        assert positions==sorted(positions) and len(set(positions))==50 and played==100 and refs==10
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert page.locator('section').count()==50 and page.locator('input,select,textarea').count()==0
        assert not errors and not external,(errors,external)
        report=dict(status='passed',pairs=50,coinve=40,tripvvt=10,video_instances_played=played,reference_images_loaded=refs,
            all_cards_continuously_visible=True,scroll_positions_increasing=True,rating_controls=0,pagination_controls=0,
            full_prompts_verified=50,full_trip_captions_verified=10,all_media_hashes_verified=True,mobile_no_horizontal_overflow=True,
            javascript_errors=errors,external_requests=external,browser=browser.version,
            index_sha256=hashlib.sha256((BUNDLE/'index.html').read_bytes()).hexdigest())
        (BUNDLE/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report),flush=True);browser.close()

if __name__=='__main__':main()
