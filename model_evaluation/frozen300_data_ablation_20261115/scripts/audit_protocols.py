"""Read-only sidecar audit of all three main inference configurations."""
from common import *
import subprocess

def main():
 baseline=EVAL/'ltx25_distilled_v7_300/outputs';common_contract=None;software={}
 for case in load_cases():
  p=Path(case['path']).with_suffix('.json');assert sha(p)==case['record_sha256']
  r=read(p);base=read(baseline/f'{case["id"]}.json')
  a=r['contract'];b=base['contract']
  fields=set(a)-{'experiment','lora','implementation_sha256'}
  assert fields==set(b)-{'experiment','lora','implementation_sha256'}
  assert all(a[k]==b[k] for k in fields),(case['model'],case['id'])
  assert b['lora'] is None and a['lora']['sha256']==read(ROOT/'metadata/implementation.json')['adapters'][case['model']]['sha256']
  audit=r['protocol_audit'];assert audit['actual_updates']==8 and audit['step_indices']==list(range(8))
  assert all(audit[k] for k in ['condition_tokens_unchanged','finite_each_step','target_last','no_gt'])
  software[case['model']]=r['software'];software['ltx_distilled_v7']=base['software']
  fixed={k:a[k] for k in fields-{'num_frames','canvas_wh'}}
  if common_contract is None:common_contract=fixed
  else:assert fixed==common_contract
 code="import importlib.metadata as m,json\nout={}\nfor n in ['torch','torchvision','numpy','Pillow','av','lpips','scikit-image','pyiqa']:\n try:out[n]=m.version(n)\n except m.PackageNotFoundError:pass\nprint(json.dumps(out))"
 envs={}
 for name,py in [('gt',GTPY),('vbench',VBPY)]:
  envs[name]=json.loads(subprocess.check_output([str(py),'-c',code],text=True))
 expected=read(ROOT/'metadata/implementation.json')['gt_environment']['versions']
 assert all(envs['gt'][k]==v for k,v in expected.items())
 atomic(ROOT/'metadata/inference_protocol_audit.json',dict(status='passed',new_output_sidecars=600,main_models=['ltx_distilled_v7']+MODELS,common_contract=common_contract,inference_software=software,evaluation_software=envs,all_new_condition_protection_passed=True))
 print('INFERENCE_PROTOCOL_AUDIT_PASSED_600')

if __name__=='__main__':main()
