from pathlib import Path
import hashlib,json,os,sys
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'review/datamake_v1_balanced80_scroll_20261030'
RUNTIME=ROOT.parents[1]/'model_evaluation/ivebench_lance/runtime'

def main():
    m=json.loads((PACKAGE/'manifest.json').read_text());external=[];errors=[]
    for r in m['media']:assert (PACKAGE/r['path']).is_file()
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(RUNTIME/'debs/extracted/usr/lib/x86_64-linux-gnu')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    shots=ROOT/'metadata/browser';shots.mkdir(parents=True,exist_ok=True)
    video_count=image_count=0;played=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(RUNTIME/'google-chrome-stable/opt/google/chrome/chrome'),headless=True,env=env)
        page=browser.new_page(viewport={'width':1550,'height':1100})
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((PACKAGE/'index.html').as_uri(),wait_until='load');page.evaluate('document.fonts.ready')
        assert page.locator('article').count()==80
        assert page.locator('input,select,textarea,form').count()==0
        assert page.evaluate('[...document.querySelectorAll("article")].every((x,i,a)=>x.offsetHeight>0&&(i===0||x.offsetTop>a[i-1].offsetTop))')
        if '--quick' in sys.argv:
            for c in m['cards']:
                card=page.locator('#'+c['review_id'])
                assert card.locator('.prompts').first.locator('p').nth(0).text_content()==c['prompt']['zh']
                assert card.locator('.prompts').first.locator('p').nth(1).text_content()==c['prompt']['en']
            v=page.locator('video.primary').first;v.scroll_into_view_if_needed();v.evaluate('v=>v.play()')
            page.wait_for_function('document.querySelector("video.primary").currentTime>0',timeout=15000)
            v.evaluate('v=>v.pause()');page.locator('article').first.screenshot(path=str(shots/'first_card.png'))
            assert not errors and not external
            browser.close()
            report=dict(status='passed',scope='basic page and prompt verification; full browser media sweep stopped at user request',
                        cards=80,full_bilingual_prompts_verified=80,rating_controls=0,continuous_scroll=True,
                        representative_video_playback=True,external_requests=0,javascript_errors=[],
                        html_sha256=hashlib.sha256((PACKAGE/'index.html').read_bytes()).hexdigest())
            (PACKAGE/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(report),flush=True);return
        for i,c in enumerate(m['cards']):
            card=page.locator('#'+c['review_id']);card.scroll_into_view_if_needed()
            assert card.locator('.prompts').first.locator('p').nth(0).text_content()==c['prompt']['zh']
            assert card.locator('.prompts').first.locator('p').nth(1).text_content()==c['prompt']['en']
            for v in card.locator('video.primary').all():
                v.scroll_into_view_if_needed();v.evaluate('v=>{if(!v.src)v.src=v.dataset.src}')
                v.evaluate('v=>new Promise((resolve,reject)=>{if(v.readyState>=1)return resolve();v.addEventListener("loadedmetadata",resolve,{once:true});v.addEventListener("error",()=>reject(Error("media error")),{once:true})})')
                assert v.evaluate('v=>v.duration>9.9&&!v.error');video_count+=1
            # Current-round references only: history is separately expanded below.
            for im in card.locator(':scope > .refs img').all():
                im.scroll_into_view_if_needed();page.wait_for_function('(src)=>[...document.images].some(i=>i.getAttribute("src")===src&&i.complete&&i.naturalWidth>0)',arg=im.get_attribute('src'));image_count+=1
            if (i+1)%10==0:print('browser cards',i+1,flush=True)
        for g in range(8):
            card=page.locator(f'article[data-group="{g}"]').first;card.scroll_into_view_if_needed()
            card.locator('.play').click()
            page.wait_for_function('id=>[...document.getElementById(id).querySelectorAll("video.primary")].every(v=>v.currentTime>0)',arg=card.get_attribute('id'),timeout=20000)
            card.locator('.pause').click();played.append(g)
            card.screenshot(path=str(shots/f'group{g+1}.png'))
        for ident in ['review_3_09','review_4_09']:
            card=page.locator('#'+ident);card.scroll_into_view_if_needed();card.locator('details.history > summary').click()
            assert card.locator('details.history').get_attribute('open') is not None
            assert card.locator('.history .turn').count()>0
            v=card.locator('details.history video').first;v.scroll_into_view_if_needed();v.evaluate('v=>{if(!v.src)v.src=v.dataset.src;return v.play()}')
            page.wait_for_function('(src)=>[...document.querySelectorAll("video")].some(v=>v.getAttribute("data-src")===src&&v.currentTime>0)',arg=v.get_attribute('data-src'),timeout=15000)
            v.evaluate('v=>v.pause()');card.locator('details.history > summary').click()
        page.locator('header a[href="#group7"]').click();page.wait_for_timeout(100)
        assert page.locator('#group7').evaluate('e=>Math.abs(e.getBoundingClientRect().top)<200')
        page.locator('header a[href="report.html"]').click()
        assert '847' in page.locator('pre').text_content()
        assert 'Prompt' in page.title()
        assert not external and not errors,(external,errors)
        browser.close()
    report=dict(status='passed',cards=80,primary_videos_loaded=video_count,current_round_images_loaded=image_count,
                full_bilingual_prompts_verified=80,rating_controls=0,continuous_scroll=True,
                representative_group_playback=played,multiturn_context_playback=True,
                external_requests=0,javascript_errors=errors,html_sha256=hashlib.sha256((PACKAGE/'index.html').read_bytes()).hexdigest())
    (PACKAGE/'browser_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
