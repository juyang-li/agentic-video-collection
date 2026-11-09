"""Check continuous scroll, zero score controls, and representative offline playback."""
from pathlib import Path
from collections import Counter
import hashlib,json,os
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT/'review/recent_motion_datasets_scroll50_20261030'
RUNTIME=ROOT.parents[1]/'model_evaluation/ivebench_lance/runtime'


def main():
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(RUNTIME/'debs/extracted/usr/lib/x86_64-linux-gnu')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    external=[];errors=[];shots=ROOT/'metadata/browser';shots.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(RUNTIME/'google-chrome-stable/opt/google/chrome/chrome'),headless=True,env=env)
        page=browser.new_page(viewport={'width':1500,'height':1100});page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None);page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((PACKAGE/'index.html').as_uri(),wait_until='load');page.evaluate('document.fonts.ready')
        assert page.locator('article').count()==50
        assert page.locator('input,select,textarea,form').count()==0
        assert page.evaluate("[...document.querySelectorAll('article')].every((x,i,a)=>x.offsetHeight>0 && (i===0 || x.offsetTop>a[i-1].offsetTop))")
        assert page.evaluate('document.documentElement.scrollHeight')>20000
        ids=page.locator('article').evaluate_all('(xs)=>xs.map(x=>x.id)')
        videos=0
        for i,ident in enumerate(ids):
            card=page.locator('#'+ident);card.scroll_into_view_if_needed();card.locator('video').first.scroll_into_view_if_needed()
            page.wait_for_function("id=>[...document.getElementById(id).querySelectorAll('video')].every(v=>v.readyState>=1&&!v.error)",arg=ident,timeout=30000)
            for img in card.locator('img').all():
                img.scroll_into_view_if_needed();page.wait_for_function('(id)=>[...document.getElementById(id).querySelectorAll("img")].every(x=>x.complete&&x.naturalWidth>0)',arg=ident)
            videos+=card.locator('video').count()
        played=[]
        for group in ['xdance','realis','movebench','wan']:
            card=page.locator(f'article[data-group="{group}"]').first;card.scroll_into_view_if_needed();v=card.locator('video').first;v.scroll_into_view_if_needed()
            v.evaluate('v=>v.play()');page.wait_for_function('id=>document.querySelector(`#${id} video`).currentTime>0',arg=card.get_attribute('id'),timeout=15000);v.evaluate('v=>v.pause()');played.append(group)
            card.screenshot(path=str(shots/f'{group}.png'))
        page.locator('header a[href="#wan"]').click();page.wait_for_timeout(100)
        assert page.locator('#wan').evaluate('e=>Math.abs(e.getBoundingClientRect().top)<200')
        page.locator('header a[href="#xdance"]').click();page.wait_for_timeout(100)
        assert page.locator('#xdance').evaluate('e=>Math.abs(e.getBoundingClientRect().top)<200')
        assert not errors and not external,(errors,external)
        browser.close()
    report={'status':'passed','cards':50,'all_cards_simultaneously_present':True,'all_cards_scrollable':True,
        'video_metadata_loaded':videos,'rating_controls':0,'representative_playback':played,'external_requests':0,
        'javascript_errors':errors,'section_anchor_navigation':True,'html_sha256':hashlib.sha256((PACKAGE/'index.html').read_bytes()).hexdigest()}
    (PACKAGE/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
