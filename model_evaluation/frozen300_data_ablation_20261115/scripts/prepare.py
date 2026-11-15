from common import *
import concurrent.futures as cf,subprocess,shutil
import av

def one(item):
 gt=GTS[item['id']];assert gt['uid']==item['uid'] and sha(PKG/gt['target']['path'])==gt['target']['sha256']
 result=[]
 for model in MODELS:
  p=SOURCES[model]/f'outputs/{item["id"]}.mp4';r=read(p.with_suffix('.json'));c=r['contract']
  assert r['status']=='complete' and r['id']==item['id'] and r['uid']==item['uid']
  assert c['input_manifest_sha256']==INPUT_SHA and r['original_prompt']==item['prompt']['en']
  assert r['source_sha256']==item['source']['sha256'] and r['ordered_reference_sha256s']==[x['sha256'] for x in item['references']]
  assert c['steps']==8 and c['seed']==42 and c['noise_seed']==10042 and c['cfg']==1 and c['stg']==0 and c['ref_cfg']==1
  assert c['lora']['sha256']==ADAPTERS[model]['sha256'] and r['num_frames']==item['num_frames'] and r['fps']==12 and r['canvas_wh']==item['canvas_wh']
  assert r['comparison_interval_seconds']==[item['timestamps_seconds'][0],item['timestamps_seconds'][-1]]
  assert sha(p)==r['output_sha256']
  n=0
  with av.open(str(p)) as container:
   assert float(container.streams.video[0].average_rate)==12
   container.streams.video[0].codec_context.thread_count=2
   for f in container.decode(video=0):
    assert [f.width,f.height]==item['canvas_wh'] and abs(float(f.pts*f.time_base)-n/12)<1e-6;n+=1
  assert n==item['num_frames']
  result.append(dict(id=item['id'],uid=item['uid'],model=model,dataset=item['dataset'],group=item['group'],path=str(p),output_sha256=r['output_sha256'],gt_sha256=gt['target']['sha256'],num_frames=n,canvas_wh=item['canvas_wh'],record_sha256=sha(p.with_suffix('.json'))))
 atomic(ROOT/f'metadata/pairs/{item["id"]}.json',{'status':'passed','outputs':result})
 print('PAIR_VERIFIED',item['id'],flush=True);return result

def main():
 global GTS,INPUT_SHA,ADAPTERS
 assert not os.environ.get('SLURM_JOB_ID')
 for d in ['metadata','logs','scores','runtime','report']:(ROOT/d).mkdir(exist_ok=True)
 items=rows(PKG/'inputs.jsonl');GTS={r['id']:r for r in rows(PKG/'ground_truth.jsonl')};INPUT_SHA=sha(PKG/'inputs.jsonl')
 assert len(items)==len({i['id'] for i in items})==len(GTS)==300
 ADAPTERS={m:read(SOURCES[m]/'metadata/adapter.json') for m in MODELS}
 for m in MODELS:
  assert sha(SOURCES[m]/'metadata/inputs.jsonl')==INPUT_SHA
  assert {p.stem for p in (SOURCES[m]/'outputs').glob('*.mp4')}==set(GTS)
  for i in range(4):
   loaded=read(SOURCES[m]/f'metadata/adapter_loaded_{i}.json');assert loaded['sha256']==ADAPTERS[m]['sha256'] and loaded['all_values_verified'] and loaded['strength']==1
 gtproto=read(GTOLD/'metadata/protocol.json');vbid=read(VBOLD/'metadata/implementation.json')
 assert subprocess.check_output(['git','-C',str(VBENV/'runtime/VBench'),'rev-parse','HEAD'],text=True).strip()==vbid['vbench_revision']
 for n,h in vbid['official_sources_sha256'].items():assert sha(VBENV/'runtime/VBench'/n)==h
 for n,h in vbid['weights_sha256'].items():assert sha(VBENV/'runtime/vbench_models'/n)==h
 gtenv=read(GTOLD/'metadata/environment.json')
 for n,h in gtenv['lpips_weights'].items():assert sha(n)==h
 assert sha(Path('/scratch/prj0000000275/jcy/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))==gtenv['alexnet_sha256']
 impl={'input_manifest_sha256':INPUT_SHA,'gt_manifest_sha256':sha(PKG/'ground_truth.jsonl'),'gt_numeric_protocol':{k:v for k,v in gtproto.items() if k not in ['source_sha256','models','execution','delivery','hf_revision']},'parent_gt_protocol_sha256':sha(GTOLD/'metadata/protocol.json'),'parent_vbench_identity_sha256':sha(VBOLD/'metadata/implementation.json'),'vbench_identity':vbid,'gt_environment':gtenv,'adapters':ADAPTERS,'static_filter':'not_requested','cpu_node_authorized_by_user':True,'release_job_on_terminal':JOB}
 old=ROOT/'metadata/implementation.json'
 if old.exists():assert read(old)==impl
 else:atomic(old,impl)
 shutil.copyfile(PKG/'inputs.jsonl',ROOT/'metadata/inputs.jsonl')
 with cf.ThreadPoolExecutor(max_workers=4) as pool:groups=list(pool.map(one,items))
 media=[r for g in groups for r in g];assert len(media)==600
 (ROOT/'metadata/media.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in media))
 # Existing baseline and A849 values remain in their parent experiments, never rewritten.
 count=0
 for model in ['ltx_distilled_v7','a849_v7']:
  for item in items:
   r=read(GTOLD/f'scores/{model}/{item["id"]}.json');gt=GTS[item['id']]
   assert r['status']=='complete' and r['uid']==item['uid'] and r['gt_sha256']==gt['target']['sha256'] and r['protocol_sha256']==sha(GTOLD/'metadata/protocol.json')
   native=EVAL/(f'ltx25_distilled_v7_300/outputs/{item["id"]}.mp4' if model=='ltx_distilled_v7' else f'frozen300_20261113/outputs/a849/{item["id"]}.mp4')
   assert sha(native)==r['output_sha256']
   for dim in VBDIMS:
    p=VBOLD/'metadata/vbench'/('raw' if dim=='temporal_flickering' else 'quality')/(f'{r["output_sha256"]}.json' if dim=='temporal_flickering' else f'{dim}--{r["output_sha256"]}.json')
    v=read(p);assert v['status']=='complete' and v['implementation_sha256']==sha(VBOLD/'metadata/implementation.json') and v['output_sha256']==r['output_sha256']
    assert math.isfinite(v['score']) and abs(v['score']-(v['raw_score']/100 if dim=='imaging_quality' else v['raw_score']))<1e-10
   count+=1
 atomic(ROOT/'metadata/reuse_audit.json',{'status':'passed','models':['ltx_distilled_v7','a849_v7'],'videos':count,'parent_gt_protocol_sha256':sha(GTOLD/'metadata/protocol.json'),'parent_vbench_identity_sha256':sha(VBOLD/'metadata/implementation.json'),'note':'Same frozen inputs/GT, original native media SHA and numeric protocol; A849 dev40 is separate reference'})
 atomic(ROOT/'metadata/preparation.json',{'status':'complete','tasks':300,'outputs':600,'full_native_decodes':600,'frozen_frame_pairs':sum(r['num_frames'] for r in media),'static_filter':'not_requested'})
 print('PREPARATION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  atomic(ROOT/'metadata/prepare_failure.json',{'error':traceback.format_exc()});raise
