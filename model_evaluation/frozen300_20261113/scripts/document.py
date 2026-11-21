"""Record exact model interfaces and local V7 code dependencies without bundling code/weights."""
import subprocess
from build import *

def main():
 p=ROOT/'package';p.mkdir(exist_ok=True)
 v=read(ROOT.with_name('ltx25_vace_a100_20261111')/'runtime/vace_ready.json')
 h=ROOT.with_name('minimax_h3_ref2va90_local_20261112')
 hm=read(h/'metadata/model_download_manifest.json');ltx=read(NATIVE/'runtime/models/native_model_revision.json')
 checkpoint=NATIVE/'experiments/training/mixed720p_sequential_visual_dropout/job_197731/checkpoints/lora_weights_step_00849.safetensors'
 contract=read(checkpoint.with_suffix('.contract.json'))
 native_code=['project_layout.py','native_reference_inference.py','native_reference_strategy.py','reference_positions.py','native_cached_dataset.py','compact_text_cache.py','native_image_cache_contract.py','prepare_datamake_media.py','coinve_cache_contract.py','encoding_helpers.py','encode_native_dataset.py','contact_sheet_contract.py']
 code=[]
 for name in native_code:
  f=NATIVE/'scripts/common'/name
  if not f.exists():continue
  tracked=subprocess.run(['git','-C',str(NATIVE),'ls-files','--error-unmatch',str(f)],capture_output=True).returncode==0
  code.append({'file':str(f),'sha256':sha(f),'tracked_in_git':tracked})
 vpath=Path(v['vace']['path'])
 vfiles=[x.relative_to(vpath).as_posix() for x in sorted(vpath.rglob('*')) if x.is_file() and '.cache' not in x.parts]
 lock={'h3':{'repo':hm['repo'],'revision':hm['revision'],'files':[x['rfilename'] for x in hm['files']],'code_repo':'https://github.com/sgl-project/sglang','code_commit':(h/'metadata/sglang_commit.txt').read_text().strip(),'local_patch':{'file':str(h/'metadata/sglang_build_portability.patch'),'sha256':sha(h/'metadata/sglang_build_portability.patch')},'requirements':{'file':str(h/'metadata/requirements.lock.txt'),'sha256':sha(h/'metadata/requirements.lock.txt')}},'vace':{**v['vace'],'files':vfiles,'packages':v['packages'],'pipeline':v['pipeline'],'annotators':{k:v[k] for k in ['dino','pose','human_parsing','sam2','pose_code']}},'ltx':ltx,'v7':{'native_root':str(NATIVE),'datamake_head':subprocess.check_output(['git','-C',str(NATIVE),'rev-parse','HEAD'],text=True).strip(),'upstream_repo':'https://github.com/Lightricks/LTX-2','upstream_commit':subprocess.check_output(['git','-C',str(NATIVE/'runtime/repos/LTX-2'),'rev-parse','HEAD'],text=True).strip(),'modules':code,'training_config':{'file':str(NATIVE/'configs/training/mixed720p_sequential_visual_dropout/train.yaml'),'sha256':sha(NATIVE/'configs/training/mixed720p_sequential_visual_dropout/train.yaml')},'inference_settings':read(NATIVE/'configs/inference/stronger_guidance/inference.json')},'a849':{'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'contract_file':str(checkpoint.with_suffix('.contract.json'))}}
 audit=read(ROOT.with_name('benchmark_full10s_20261112')/'metadata/technical_audit_vace_partial.json')
 vals=[x['peak_allocated_bytes'] for x in audit['checks']]
 lock['vace']['measured_h200']={'scope':'26 historical full10s samples; independent process/card','num_frames':121,'peak_allocated_bytes_max':max(vals),'peak_allocated_gib_max':max(vals)/1024**3,'measurement':'PyTorch peak allocated; not board total'}
 write(p/'model_interfaces.json',lock)
 shutil.copyfile(ROOT/'HANDOFF.md',p/'README.md')
 shutil.copyfile(ROOT/'result_schema.json',p/'result_schema.json')
 print(json.dumps({'upstream_commit':lock['v7']['upstream_commit'],'modules':code,'a849':lock['a849']},indent=2))
if __name__=='__main__':main()
