"""Read-only progress summary; derived H3 copies count as one model output."""
import json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT.parent/'a849_tripvvt_pilot32_20261115'
EXT=ROOT.parent/'a849_v4_extensions16_20261115'

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def complete(root,model,ids):
    return sum(read(root/'outputs'/model/f'{sid}.json',{}).get('status')=='complete'for sid in ids)

def main():
    ids=[f'tripvvt_{n:02}'for n in range(1,33)]
    ext=[json.loads(line)['id']for line in (EXT/'metadata/inputs.jsonl').read_text().splitlines()if line.strip()]
    selected=read(PILOT/'metadata/selection_decision.json',{}).get('selected',[])
    pilot={m:complete(PILOT,m,ids)for m in ['a849','h3_pdd8_native','h3_pdd8']}
    additions={m:complete(EXT,m,ext)for m in ['ltx_base','a849','vace','h3_pdd8_native','h3_pdd8']}
    other={m:complete(PILOT,m,selected)for m in ['ltx_base','vace']}
    total=pilot['a849']+pilot['h3_pdd8_native']+sum(additions[m]for m in ['ltx_base','a849','vace','h3_pdd8_native'])+sum(other.values())
    state=read(ROOT/'metadata/finalize_state.json',{})
    result=dict(beijing_time=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(time.time()+28800)),job_id=202896,allocation_cancellation_authorized=False,
        pilot32=dict(expected_primary_outputs=64,models=pilot,selected=selected),
        extension16=dict(expected_outputs=64,models=additions,input_gate=read(EXT/'metadata/generation_gate.json',{}).get('status'),supervisor=read(EXT/'metadata/supervisor.json',{}).get('status')),
        selected8_other_models=dict(expected_outputs=16,models=other),new_inference=dict(complete=total,expected=144,excluded_pilot_pairs_kept=48),
        final116=dict(state=state.get('status'),manifest_assembled=(ROOT/'metadata/inputs.jsonl').exists(),direct_reviews=read(ROOT/'metadata/visual_observations.json',{}).get('reviewed_outputs'),valid_scores=read(ROOT/'metadata/group_metrics.json',{}).get('valid_scores'),expected_outputs=464,expected_scores=2784),
        deliveries={name:{k:read(root/'metadata/baidu_delivery.json',{}).get(k)for k in ['status','remote']}for name,root in [('pilot32',PILOT),('final116',ROOT)]},
        failures={name:read(root/f'metadata/{file}.json')for name,root,file in [('extension',EXT,'supervisor_failure'),('final',ROOT,'finalize_failure')]if(root/f'metadata/{file}.json').exists()})
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
