"""Pure input equivalence, identity decision and paired-static policy."""
import hashlib,json,math
MODELS=('ltx_base','a849','vace')
DIMS=('subject_consistency','background_consistency','aesthetic_quality','imaging_quality','temporal_flickering','motion_smoothness')
LABELS=('Subject Consistency','Background Consistency','Aesthetic Quality','Imaging Quality','Temporal Flickering','Motion Smoothness')
def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def generation_input(item,model,condition=None):
 result=dict(source_sha256=item['source']['sha256'],references=[{k:r.get(k) for k in ('sha256','type','view')} for r in item['references']],prompt_en=item['prompt']['en'],canvas_wh=item['canvas_wh'],fps=item['fps'],num_frames=item['num_frames'],timestamps_seconds=item['timestamps_seconds'])
 if model=='vace':
  if condition is None: raise ValueError('VACE condition required')
  result.update(mask_sha256=condition['mask_sha256'],adapted_prompt=condition['adapted_prompt'],source_mode='official_repaint_gray128')
 return result
def identity_decision(reviews):
 if len(reviews)!=4:raise ValueError('Need all four tasks')
 failed=0
 for row in reviews:
  if set(row['models'])!=set(MODELS):raise ValueError('Need all three models')
  vals=[row['models'][m]['clear_identity_transfer'] for m in MODELS]
  if any(type(v) is not bool for v in vals):raise ValueError('Missing/uncertain result cannot count as identity failure')
  failed+=all(v is False for v in vals)
 return {'drop_identity':failed>=3,'all_three_failed_tasks':failed,'identity_count':0 if failed>=3 else 4,'hair_color_count':16 if failed>=3 else 12}
def common_static(ids,flags):
 return [sid for sid in ids if all(flags.get((sid,m)) is True for m in ('source',*MODELS))]
def finite_score(value):
 return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
