from common import *
import csv,math
import subprocess

def main():
 c=read(ROOT/'metadata/completion.json');assert c['status']=='complete' and c['valid_scores']==7200 and c['errors']==0
 use_static=static_required();assert c['static_filter_required']==use_static
 assert read(ROOT/'metadata/pairing_audit.json')['status']=='passed'
 rs=list(csv.DictReader((ROOT/'report/per_video_metrics.csv').open()));items=rows(ROOT/'metadata/inputs.jsonl')
 identity=read(ROOT/'metadata/implementation.json')
 assert sha(PKG/'inputs.jsonl')==sha(ROOT/'metadata/inputs.jsonl')==identity['input_manifest_sha256']
 assert subprocess.check_output(['git','-C',str(OLD/'runtime/VBench'),'rev-parse','HEAD'],text=True).strip()==identity['vbench_revision']
 for name,h in identity['official_sources_sha256'].items():assert sha(OLD/'runtime/VBench'/name)==h
 for name,h in identity['weights_sha256'].items():assert sha(OLD/'runtime/vbench_models'/name)==h
 assert len(rs)==len({(r['id'],r['model'],r['dimension']) for r in rs})==7200
 assert {(r['id'],r['model'],r['dimension']) for r in rs}=={(i['id'],m,d) for i in items for m in MODELS for d in DIMS}
 frozen={i['id']:i for i in items}
 for r in rs:
  assert r['status']=='complete' and math.isfinite(float(r['score']))
  assert abs(float(r['score'])-(float(r['raw_score'])/100 if r['dimension']=='imaging_quality' else float(r['raw_score'])))<1e-10
  assert all(r[k]==frozen[r['id']][k] for k in ['uid','dataset','group'])
  h=r['output_sha256'];dim=r['dimension']
  cache=ROOT/'metadata/vbench'/('raw' if dim=='temporal_flickering' else 'quality')/(f'{h}.json' if dim=='temporal_flickering' else f'{dim}--{h}.json')
  cached=read(cache);assert cached['status']=='complete' and cached['output_sha256']==h and cached['dimension']==dim
  assert cached['score']==float(r['score']) and cached['raw_score']==float(r['raw_score'])
 score_values={(r['id'],r['model'],r['dimension']):float(r['score']) for r in rs}
 scopes={('overall','all'):[i['id'] for i in items]}
 for field,level in [('dataset','dataset'),('group','task')]:
  for label in dict.fromkeys(i[field] for i in items):scopes[level,label]=[i['id'] for i in items if i[field]==label]
 summary=list(csv.DictReader((ROOT/'report/summary.csv').open()))
 expected_summary={(level,label,m,d) for level,label in scopes for m in MODELS for d in DIMS}
 assert len(summary)==len(expected_summary) and {(r['level'],r['group'],r['model'],r['dimension']) for r in summary}==expected_summary
 for r in summary:
  ids=scopes[r['level'],r['group']];values=[score_values[sid,r['model'],r['dimension']] for sid in ids]
  assert int(r['n'])==int(r['expected'])==int(r['paired_n'])==len(ids)
  expected_mean=math.fsum(values)/len(values)
  assert abs(float(r['mean'])-expected_mean)<1e-12 and abs(float(r['paired_mean'])-expected_mean)<1e-12
 flags={};actual={'n':None}
 if use_static:
  rowsflags=list(csv.DictReader((ROOT/'report/static_flags.csv').open()));assert len(rowsflags)==1500
  for row in rowsflags:assert row['is_static'] in ['True','False'];flags[row['id'],row['model']]=row['is_static']=='True'
  assert set(flags)=={(i['id'],m) for i in items for m in ['source']+MODELS}
  scores={(r['id'],r['model']):float(r['score']) for r in rs if r['dimension']=='temporal_flickering'}
  static_groups=read(ROOT/'report/temporal_flickering_static_paired.json');assert set(static_groups)=={f'{level}/{label}' for level,label in scopes}
  for (level,label),ids in scopes.items():
   eligible=[sid for sid in ids if all(flags[sid,m] for m in ['source']+MODELS)]
   s=static_groups[f'{level}/{label}']
   assert s['filter_complete'] and s['n']==len(eligible) and s['task_ids']==s['eligible_ids']==eligible
   assert s['status']==('complete' if eligible else 'not_applicable')
   for m in MODELS:
    if eligible:assert abs(s['means'][m]-math.fsum(scores[sid,m] for sid in eligible)/len(eligible))<1e-12
    else:assert s['means'][m] is None
  actual=static_groups['overall/all']
 else:
  assert read(ROOT/'metadata/user_override.json')['request']=='不用静态筛选'
  omitted=read(ROOT/'report/static_filter_status.json')
  assert omitted['status']==c['static_status']=='not_requested' and not omitted['partial_cache_used_in_report'] and not omitted['static_paired_means_reported']
 media=rows(ROOT/'metadata/scoring_media.jsonl');assert len(media)==1200
 sources=rows(ROOT/'metadata/source_media.jsonl');assert len(sources)==300
 output_hashes={(r['id'],r['model']):r['output_sha256'] for r in media}
 for r in rs:assert r['output_sha256']==output_hashes[r['id'],r['model']]
 for r in media+(sources if use_static else []):
  assert sha(r['path'])==r['output_sha256']
  item=frozen[r['id']]
  wh=r.get('canvas_wh',item['canvas_wh']);n=item['num_frames']
  if use_static:
   static=read(ROOT/f'metadata/vbench/static/{r["output_sha256"]}.json')
   assert static['media_sha256']==r['output_sha256'] and static['is_static']==flags[r['id'],r['model']]
   assert static['parameters']=={'thres':3.0*min(wh)/256.0,'count_num':round(2*n/16.0)}
  if r['model']=='h3_pdd8':
   assert sha(r['native_path'])==r['native_sha256']
   assert r['pixel_preservation'] and len(r['pixel_hashes'])==r['num_frames']
   assert r['generated_frame_indices']==[2*i for i in range(frozen[r['id']]['num_frames'])]
   assert r['selected_pts']==[i/12 for i in range(r['num_frames'])]
 for item in items:assert sha(PKG/item['source']['path'])==item['source']['sha256']
 assert read(ROOT/'metadata/gpu_smoke.json')['status']=='passed'
 atomic(ROOT/'metadata/final_validation.json',{'status':'passed','tasks':300,'models':4,'score_keys':7200,'static_filter_required':use_static,'static_status':c['static_status'],'source_and_output_filter_task_keys':1500 if use_static else 0,'h3_lossless_derivations':300,'summary_rows_verified':len(summary),'static_scopes_verified':len(scopes) if use_static else 0,'independent_means_verified':True,'static_common_n':actual['n'],'human_reviewed':False,'kind':'automatic metric and technical validation only'})
 print('Full VBench metric validation passed')
if __name__=='__main__':main()
