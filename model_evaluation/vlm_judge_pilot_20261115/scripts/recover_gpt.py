"""Serial recovery honoring OpenRouter's in-flight budget settlement window."""
import os,time
from pilot import ROOT,load,score,save

def wait(seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:time.sleep(min(15,deadline-time.monotonic()))

def main():
    if os.environ.get('VLM_GPT_OPENROUTER')!='1':raise ValueError('Select the authorized OpenRouter route explicitly')
    selected=load(ROOT/'metadata/stability_selection.json')['candidate_ids']
    jobs=[(c,m,r) for c in load(ROOT/'metadata/candidates.json') for r in ([0,1,2] if c['id'] in selected else [0]) for m in ['edit_achievement','content_preservation']]
    print('Waiting120s for prior in-flight reservations to settle',flush=True);wait(120)
    for c,m,r in jobs:
        dest=ROOT/'results'/f'gpt_{c["id"]}_{m}_r{r}.json'
        if dest.exists() and load(dest)['status']=='ok':continue
        for trial in range(3):
            score(c,m,'gpt',r);record=load(dest)
            if record['status']=='ok':break
            err=record.get('response',{}).get('error',{})
            metadata=err.get('metadata',{})
            if metadata.get('reason')!='in_flight_budget_exhausted':
                save(ROOT/'metadata/gpt_recovery_state.json',{'status':'needs_attention','candidate':c['id'],'metric':m,'repeat':r,'response':err});return
            delay=int(metadata.get('headers',{}).get('Retry-After','120'))
            print('Provider asks settlement wait',delay,'s',flush=True);wait(delay)
        else:
            save(ROOT/'metadata/gpt_recovery_state.json',{'status':'settlement_not_resolved','candidate':c['id'],'metric':m,'repeat':r});return
    save(ROOT/'metadata/gpt_recovery_state.json',{'status':'complete'})

if __name__=='__main__':main()
