"""Persistent login completion; quality selection and direct reviews are explicit gates."""
import subprocess,time,shutil,traceback
from common import *

PILOT=ROOT.parent/'a849_tripvvt_pilot32_20261115'
EXT=ROOT.parent/'a849_v4_extensions16_20261115'
PY=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'
VB=ROOT.parent/'a849_external_vbench_v2_96/runtime/vbench/bin/python'

def main():
    assert not os.environ.get('SLURM_JOB_ID')
    env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def state(status,**kw):atomic(ROOT/'metadata/finalize_state.json',dict(status=status,pid=os.getpid(),updated_unix=time.time(),**kw))
    def run(root,name,*args,python=PY):
        with (ROOT/'logs/finalize_commands.log').open('a',buffering=1)as log:subprocess.run([str(python),str(root/'scripts'/name),*args],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    def upload(root):
        receipt=root/'metadata/baidu_delivery.json'
        if receipt.exists()and read(receipt).get('status')=='complete':return
        package=read(root/'metadata/package.json');uploader=ROOT.parent/'minimax_h3_ref2va90_local_20261112/scripts/upload_baidu_xpan.py'
        credentials=ROOT.parents[1]/'dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json'
        upload_env=dict(env)
        for key in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:upload_env.pop(key,None)
        remote='/apps/bypy/lance_review/'+Path(package['path']).name
        with (root/'logs/baidu_upload.log').open('a',buffering=1)as log:subprocess.run([str(PY),str(uploader),'--local',package['path'],'--remote',remote,'--state',str(receipt),'--bypy-config',str(credentials)],stdout=log,stderr=subprocess.STDOUT,check=True,env=upload_env)
        assert read(receipt)['status']=='complete'
    state('waiting_direct_pilot_selection')
    while not (PILOT/'metadata/selection_decision.json').exists():time.sleep(30)
    decision=read(PILOT/'metadata/selection_decision.json')
    run(PILOT,'review.py','--package');upload(PILOT)
    if decision['status']=='shortfall':
        state('quality_shortfall',selected=decision['selected'],shortfall=decision['shortfall'],no_automatic_candidate_33=True);return
    assert decision['status']=='complete'and len(decision['selected'])==8
    targets=read(PILOT/'metadata/generation_targets.json')
    for model in ['ltx_base','vace']:
        if model in targets:assert targets[model]==sorted(decision['selected'])
        targets[model]=sorted(decision['selected'])
    atomic(PILOT/'metadata/generation_targets.json',targets)
    cmd=['srun','--jobid=202896','--partition=testqueue','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','bash',str(PILOT/'scripts/selected_other_models.sh')]
    with (PILOT/'logs/selected_other_models.log').open('a',buffering=1)as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    state('generating_selected_base_vace',gpu_launcher_pid=proc.pid,selected_ids=decision['selected'])
    while proc.poll()is None:time.sleep(30)
    assert proc.returncode==0,'Selected LTX/VACE GPU step failed'
    while not (EXT/'metadata/gpu_generation.json').exists()or not (EXT/'metadata/h3_derivation.json').exists()or read(EXT/'metadata/h3_derivation.json')['status']!='complete':time.sleep(30)
    run(ROOT,'assemble.py');run(ROOT,'overlap.py');run(ROOT,'gate_inputs.py');run(ROOT,'reuse_scores.py');run(ROOT,'score.py','raw',python=VB)
    state('scoring_and_waiting_direct_reviews')
    cmd=['srun','--jobid=202896','--partition=testqueue','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','bash',str(ROOT/'scripts/score_gpu.sh')]
    with (ROOT/'logs/score_gpu.log').open('a',buffering=1)as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    while True:
        run(ROOT,'visual.py','sync');run(ROOT,'aggregate.py')
        visual=read(ROOT/'metadata/visual_observations.json');metric=read(ROOT/'metadata/group_metrics.json')
        if proc.poll()is not None:assert proc.returncode==0,'GPU metric step failed'
        if proc.poll()==0 and visual['reviewed_outputs']==464 and metric['valid_scores']==2784:break
        state('scoring_and_waiting_direct_reviews',reviewed_outputs=visual['reviewed_outputs'],scores=metric['valid_scores'],gpu_returncode=proc.poll())
        time.sleep(60)
    run(ROOT,'validate.py');run(ROOT,'review.py','--package');upload(ROOT)
    state('complete',tasks=116,outputs=464,scores=2784,uploaded_pilot=True,uploaded_v4=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'metadata/finalize_failure.json',dict(status='failed',error=traceback.format_exc(),failed_unix=time.time()));raise
