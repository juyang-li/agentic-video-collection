#!/usr/bin/env python3
"""Resumable CPU-only CoinVE/TripVVT delivery. Never changes historical artifacts."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
PROJECT = RESEARCH.parent
COIN = RESEARCH/'coinve_research'
TRIP = RESEARCH/'tripvvt_research'
ART = PROJECT/'lance_general_edit_finetune/3_lance_lora_finetuning/artifacts'
SEED = 20261108
FFMPEG = '/scratch/prj0000000275/jcy/.conda/envs/vivid/bin/ffmpeg'
FFPROBE = str(Path(FFMPEG).with_name('ffprobe'))
sys.path.insert(0, str(COIN))
import download_random_pairs as coin_download
from sample_random_pairs import shard_and_member, EXPECTED_METADATA_SHA256
spec = importlib.util.spec_from_file_location('trip_similarity', TRIP/'score_pair_similarity.py')
similarity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similarity)
CATEGORIES = ('upper_body','lower_body','dresses')
SPLITS = ('validation','test','train')
COUNTS = {'coinve':{'train':8000,'validation':400,'test':400},
          'tripvvt':{'train':2000,'validation':100,'test':100}}
TAR_LOCKS=defaultdict(threading.Lock)
TAR_REMOTES={}
TAR_INDICES={}


class MissingAssetError(Exception):
    """A pinned Hugging Face member explicitly returned EntryNotFound."""
    def __init__(self,role,member):
        self.role=role;self.member=member
        super().__init__('official pairing asset missing: '+role)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_name(path.name+'.part');p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');p.replace(path)


def read(path): return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_rows(path, rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.part')
    with temp.open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp.replace(p)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csv_write(path, rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def order(*parts): return hashlib.sha256((str(SEED)+'\0'+'\0'.join(map(str,parts))).encode()).hexdigest()


def coin_group(path):
    # Remove segment index and frame span, preserving the complete original source ID.
    m=re.match(r'(.+)_\d+_\d+to\d+(?:_[0-9a-f]+)?\.mp4$',Path(path).name)
    if m:return 'coinve:'+m[1]
    m=re.match(r'(magictime_\d+)(?:_\d+)*\.mp4$',Path(path).name)
    if m:return 'coinve:'+m[1]
    raise ValueError('unknown CoinVE source naming: '+str(path))


def quotas(dataset,split):
    if dataset=='tripvvt':
        return dict(zip(CATEGORIES,(667,667,666) if split=='train' else (33,33,34)))
    n=COUNTS[dataset][split]
    return {f'{tasks}:{op}':int(n*fraction*weight) for tasks,fraction in [(2,.7),(3,.3)]
            for op,weight in [('add',.25),('remove',.30),('replace',.45)]}


def stratum(r):return f"{r['instruction_count']}:{r['anchor_operation']}" if r['dataset']=='coinve' else r['category']


def history(population):
    """Freeze historical training and held-out group constraints and reusable originals."""
    locks=defaultdict(set); assets={}; inputs=[]
    pool=ART/'coinve_ablation_v1/coinve_4k_pool.csv'
    rows=csv_rows(pool);inputs.append(dict(path=str(pool),sha256=sha(pool),kind='historical_training_split'))
    for r in rows:
        g=coin_group(r['source_dataset_path']);locks[g].add('train' if r['split']=='train' else 'heldout')
        for role in ('source','target'):
            p=Path(r[role+'_local'])
            if p.is_file():assets['coinve:'+r[role+'_dataset_path']]=str(p)
    # Survey originals are reusable, but a survey by itself is not a frozen evaluation split.
    for base in (COIN,TRIP):
        for p in base.glob('*/sample_manifest.csv'):
            for r in csv_rows(p):
                for role in ('source','target','reference'):
                    key = r.get(role+'_path') if base==COIN else r.get({'source':'input','target':'output','reference':'cloth_image'}[role])
                    local=r.get(role+'_original')
                    if key and local and (p.parent/local).is_file():assets[('coinve:' if base==COIN else 'tripvvt:')+key]=str((p.parent/local).resolve())
    calibration=TRIP/'tripvvt_similarity_calibration_20261108/pair_similarity.json'
    scores=read(calibration)
    trip_pop=trip_population()
    for r in scores['records']:
        if r['sample_id'].startswith('tripvvt-'):
            item=trip_pop[int(r['sample_id'].split('-')[1])]
        else:
            item=next(v for v in trip_pop if Path(v['input']).name==Path(r['source']).name)
        for role,key in [('source','input'),('target','output'),('reference','cloth_image')]:
            if Path(r[role]).is_file():assets['tripvvt:'+item[key]]=r[role]
    # Inspect persisted evaluation inputs. Numeric sample IDs and dataset paths both map to original sources.
    eval_inputs=[];eval_hash_locks=defaultdict(set)
    for base in (PROJECT/'model_evaluation',ART):
        for directory,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in {'runtime','vendor','.git','node_modules','outputs','outputs_raw','media','records','attempts','inference_checkpoints','checkpoints','logs','technical','__pycache__'}]
            for name in files:
                if Path(name).suffix not in ('.json','.jsonl','.csv','.yaml'):continue
                p=Path(directory)/name
                if not any(x in str(p).lower() for x in ('evaluation','eval','validation','test','frozen','omni','benchmark')):continue
                if not any(x in name.lower() for x in ('manifest','task','selection','sample','split','validation','test','config','pool','inputs','evaluation_only','reference_controls')):continue
                if p.stat().st_size>20_000_000:continue
                text=p.read_text(errors='replace')
                ids=set(map(int,re.findall(r'coinve-(\d{6})',text)))
                paths=re.findall(r'src_videos/src_video_\d+/[^\s"\\,]+\.mp4',text)
                groups={coin_group(x) for x in paths}
                groups.update(coin_group(population[i]['source_video_path']) for i in ids if i<len(population))
                if 'tripvvt' in text.lower():
                    groups.update('tripvvt:'+x for x in re.findall(r'groundtruth/\d+\.mp4',text))
                # A source pool / training config is not a held-out list. Split pool handled above.
                if name in ('coinve_4k_pool.csv','mixed_2k2k_pool.csv','candidate_manifest.csv'):continue
                if groups:
                    for g in groups:locks[g].add('heldout')
                    for value in re.findall(r'"(?:source|target)(?:_original)?_sha256"\s*:\s*"([a-f0-9]{64})"',text):
                        eval_hash_locks[value].add('heldout')
                    eval_inputs.append(dict(path=str(p),sha256=sha(p),groups=len(groups)))
    inputs+=eval_inputs
    return dict(group_locks={g:sorted(v) for g,v in locks.items()},assets=assets,inputs=inputs,
                evaluation_hash_locks={k:sorted(v) for k,v in eval_hash_locks.items()},
                score_cache=str(calibration),score_script_sha256=sha(TRIP/'score_pair_similarity.py'))


def refresh_history():
    initial=ROOT/'metadata/history_initial.json'
    if not initial.exists():shutil.copyfile(ROOT/'metadata/history.json',initial)
    value=history(jsonl(COIN/'metadata/metadata_coinve200k.jsonl'));save(ROOT/'metadata/history.json',value)
    sources=read(ROOT/'metadata/sources.json');sources['history_inputs']=value['inputs'];sources['history_sha256']=sha(ROOT/'metadata/history.json')
    save(ROOT/'metadata/sources.json',sources)
    print('history refreshed',len(value['inputs']),'inputs',len(value['group_locks']),'source groups',flush=True)


def group_allowed(r,split):
    labels=HISTORY['group_locks'].get(r['group_id'],[])
    if len(labels)>1:return False
    return not (('heldout' in labels and split=='train') or ('train' in labels and split!='train'))


def trip_population():
    out=[]
    for split in ('train','test'):
        p=RESEARCH/f'ref_edit_review_20261030/metadata/tripvvt_{split}.jsonl'
        for index,r in enumerate(jsonl(p)):out.append(dict(r,official_split=split,official_row=index,metadata_index=len(out)))
    return out


def candidates():
    path=COIN/'metadata/metadata_coinve200k.jsonl'
    assert sha(path)==EXPECTED_METADATA_SHA256
    frozen=ROOT/'metadata/candidates.jsonl'
    if frozen.exists():
        sources=read(ROOT/'metadata/sources.json');assert sha(frozen)==sources['candidate_manifest_sha256']
        for split in ('train','test'):
            assert sha(RESEARCH/f'ref_edit_review_20261030/metadata/tripvvt_{split}.jsonl')==sources['metadata_sha256']['tripvvt_'+split]
        print('reusing verified frozen candidates',flush=True);return
    pop=jsonl(path)
    h=read(ROOT/'metadata/history.json') if (ROOT/'metadata/history.json').exists() else history(pop)
    save(ROOT/'metadata/history.json',h)
    hashes={'coinve':sha(path)}
    expected={'train':'1f1d1368f182836d697c0a0daa99169b45b61c2bf1a4ce7cef57b629d8538b83','test':'590d5b1ed925641f616238cfab9ad69318c1032f29d0c26af34916a2921db62b'}
    for s,v in expected.items():
        p=RESEARCH/f'ref_edit_review_20261030/metadata/tripvvt_{s}.jsonl';assert sha(p)==v;hashes['tripvvt_'+s]=v
    out=[]
    for i,r in enumerate(pop):
        ops=r['instruction_operation'];n=len(ops)
        if n not in (2,3) or any('styl' in o.lower() for o in ops):continue
        assert len(r['instruction'])==n==len(r['instruction_object'])
        anchor=ops[0].lower()
        if anchor not in ('add','remove','replace'):continue
        g=coin_group(r['source_video_path']);lock=h['group_locks'].get(g,[])
        if len(lock)>1:continue # Historical contamination cannot be repaired by assigning a new split.
        bucket=int(order('split',g)[:8],16)/2**32
        split=('train' if lock==['train'] else ('validation' if bucket<.5 else 'test') if lock==['heldout'] else
               'validation' if bucket<.10 else 'test' if bucket<.20 else 'train')
        row=dict(sample_id=f'coinve-{i:06d}',dataset='coinve',metadata_index=i,group_id=g,split=split,
                 instruction_count=n,anchor_operation=anchor,instructions=r['instruction'],operations=ops,
                 prompt='\n'.join(f'{j}. {s}' for j,s in enumerate(r['instruction'],1)),
                 reference_member=None,repo_id=coin_download.REPO_ID,revision=coin_download.REVISION,
                 original_metadata=r,source_member=r['source_video_path'],target_member=r['edited_video_path'])
        out.append(row)
    # A global used-target set makes the first 1K/category mutually target-disjoint.
    used=set();tp=trip_population();trip=[]
    queues={cat:iter(sorted((r for r in tp if r['cloth_type']==cat),key=lambda r:order('trip',r['metadata_index']))) for cat in CATEGORIES}
    ranks=Counter(); exhausted=set()
    while len(exhausted)<len(CATEGORIES):
        for category in CATEGORIES:
            if category in exhausted:continue
            r=next((v for v in queues[category] if v['output'] not in used),None)
            if r is None:
                exhausted.add(category);continue
            used.add(r['output']);ranks[category]+=1;rank=ranks[category]
            region={'upper_body':'upper-body garment','lower_body':'lower-body garment','dresses':'garment/outfit'}[category]
            trip.append(dict(sample_id=f"tripvvt-{r['metadata_index']:05d}",dataset='tripvvt',category=category,
                candidate_rank=rank,group_id='tripvvt:'+r['output'],split=None,metadata_index=r['metadata_index'],
                source_member=r['input'],target_member=r['output'],reference_member=r['cloth_image'],
                caption=r['text'],prompt=f"Replace the target person's {region} with the garment shown in the reference image. Preserve identity, motion, other clothing, other people, background, and camera.",
                instruction_origin='local reference-conditioned template; official caption retained separately',
                repo_id='TripVVT/TripVVT-10K',revision='596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3',original_metadata=r))
    # Prefer reuse within each CoinVE stratum; all ties have a stable seeded order.
    out.sort(key=lambda r:(r['split'],stratum(r),not all('coinve:'+r[k+'_member'] in h['assets'] for k in ('source','target')),order(r['sample_id'])))
    out+=trip
    target=ROOT/'metadata/candidates.jsonl'
    if target.exists():assert jsonl(target)==out,'frozen candidate generation differs'
    else:write_rows(target,out)
    save(ROOT/'metadata/sources.json',dict(seed=SEED,metadata_sha256=hashes,candidate_manifest_sha256=sha(target),
        source_revisions={'coinve':coin_download.REVISION,'tripvvt':'596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3'},
        cpu_only=True,visual_model_calls=0,history_inputs=h['inputs'],
        candidate_counts=dict(Counter((r['dataset']+':'+str(r.get('split'))+':'+stratum(r)) for r in out))))
    print('frozen candidates',len(out),'reusable media',len(h['assets']),flush=True)


def asset_path(r,role):
    h=HISTORY['assets'].get(r['dataset']+':'+r[role+'_member'])
    if h and Path(h).is_file():return Path(h)
    return ROOT/'media'/r['dataset']/(r['sample_id']+'_'+role+('.png' if role=='reference' else '.mp4'))


def roles(r):return ('source','target','reference') if r['reference_member'] else ('source','target')


def check_space():
    free=shutil.disk_usage(ROOT).free
    if free<12_000_000_000:raise RuntimeError('shared storage below 12 GB reserve; stop without deleting history')


def historical_hashes(workers=4):
    """Content aliases must respect history even when their dataset paths differ."""
    cache_path=ROOT/'metadata/historical_media_hashes.json'
    cache=read(cache_path) if cache_path.exists() else {}
    jobs=[]
    for row in csv_rows(ART/'coinve_ablation_v1/coinve_4k_pool.csv'):
        for role in ('source','target'):
            path=Path(row[role+'_local'])
            if path.is_file():jobs.append((path,'train' if row['split']=='train' else 'heldout'))
    def one(job):
        path,label=job;key=str(path);stat=path.stat();fingerprint=[stat.st_size,stat.st_mtime_ns]
        old=cache.get(key)
        value=old if old and old['fingerprint']==fingerprint else dict(fingerprint=fingerprint,sha256=sha(path))
        return key,value,label
    hashes=defaultdict(set,{k:set(v) for k,v in HISTORY.get('evaluation_hash_locks',{}).items()})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key,value,label in pool.map(one,jobs):cache[key]=value;hashes[value['sha256']].add(label)
    save(cache_path,cache)
    return hashes


def allowed(r, hash_locks, selected_hashes):
    technical_record=read(ROOT/'metadata/technical'/f"{r['sample_id']}.json")
    if technical_record['status']!='ok':return False
    reason=None
    for role in ('source','target'):
        value=technical_record['media'][role]['sha256']
        labels=hash_locks.get(value,set())
        if ('train' in labels and r['split']!='train') or ('heldout' in labels and r['split']=='train'):
            reason='historical_media_hash_conflict'
        if value in selected_hashes and selected_hashes[value]!=r['split']:reason='cross_split_media_hash_conflict'
    if reason:
        save(ROOT/'metadata/eligibility_exclusions'/f"{r['sample_id']}.json",dict(sample_id=r['sample_id'],reason=reason))
        return False
    for role in ('source','target'):selected_hashes[technical_record['media'][role]['sha256']]=r['split']
    return True


def fetch(r):
    check_space()
    if r['dataset']=='coinve':
        for role in roles(r):
            dest=asset_path(r,role)
            if dest.is_file() and dest.stat().st_size:continue
            shard,member=shard_and_member(r[role+'_member'])
            with TAR_LOCKS[shard]:
                if shard not in TAR_REMOTES:
                    TAR_REMOTES[shard]=coin_download.RemoteTar(shard)
                    index_path=ROOT/'metadata/tar_indices'/(Path(shard).stem+'.json')
                    previous=read(index_path) if index_path.exists() else {}
                    assert not previous or previous['revision']==r['revision']
                    TAR_INDICES[shard]={e['name']:coin_download.TarEntry(**e) for e in previous.get('entries',[])}
                remote=TAR_REMOTES[shard];index=TAR_INDICES[shard]
                entry,lookup_bytes=coin_download.locate_member(remote,member,index)
                save(ROOT/'metadata/tar_indices'/(Path(shard).stem+'.json'),dict(revision=r['revision'],size=remote.size,
                    entries=[dict(name=e.name,offset=e.offset,size=e.size) for e in index.values()]))
                payload=remote.read_range(entry.data_offset,entry.data_offset+entry.size-1)
                dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.mp4.part')
                temp.write_bytes(payload);temp.replace(dest)
                print('downloaded',r['sample_id'],role,'lookup_mb',round(lookup_bytes/1e6,1),'bytes',entry.size,flush=True)
        return
    import requests
    session=requests.Session();session.trust_env=False
    token=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token')
    if token.is_file():session.headers['Authorization']='Bearer '+token.read_text().strip()
    try:
        for role in roles(r):
            p=asset_path(r,role)
            if p.is_file() and p.stat().st_size:continue
            p.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(5):
                try:
                    with session.get(f"https://huggingface.co/datasets/{r['repo_id']}/resolve/{r['revision']}/{r[role+'_member']}",stream=True,timeout=(25,120)) as response:
                        if response.status_code==404 and response.headers.get('X-Error-Code')=='EntryNotFound':
                            raise MissingAssetError(role,r[role+'_member'])
                        response.raise_for_status();temp=p.with_suffix(p.suffix+'.part')
                        with temp.open('wb') as f:
                            for block in response.iter_content(1024*1024):f.write(block)
                        length=response.headers.get('Content-Length')
                        if not temp.stat().st_size or (length and temp.stat().st_size!=int(length)):raise RuntimeError('short download')
                        temp.replace(p)
                    break
                except (requests.RequestException,RuntimeError):
                    if attempt==4:raise RuntimeError('download retries exhausted '+r['sample_id']+' '+role) from None
                    time.sleep(2**attempt)
    finally:session.close()


def video_probe(p):
    run=subprocess.run([FFPROBE,'-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,start_time','-of','json',str(p)],capture_output=True,text=True,check=True,timeout=90)
    info=json.loads(run.stdout)['streams'][0];fps=float(Fraction(info['avg_frame_rate']))
    assert fps>0 and math.isfinite(fps)
    # Full decode with frame count and timestamps: detects errors beyond the sampled frames.
    run=subprocess.run([FFMPEG,'-nostdin','-v','error','-xerror','-threads','1','-i',str(p),
        '-map','0:v:0','-an','-progress','pipe:1','-f','null','-'],capture_output=True,text=True,check=True,timeout=300)
    if run.stderr.strip():raise ValueError('full decode diagnostics')
    frames=[int(x) for x in re.findall(r'^frame=(\d+)$',run.stdout,re.M)]
    if not frames or frames[-1]<1:raise ValueError('no decoded frames')
    duration=float(info.get('duration',frames[-1]/fps));start=float(info.get('start_time',0))
    if start<-.001 or abs(start)>.05:raise ValueError('nonzero timeline origin requires explicit handling')
    last=(frames[-1]-1)/fps
    return dict(codec=info['codec_name'],width=info['width'],height=info['height'],fps=fps,
                frames=frames[-1],duration=duration,last_frame_seconds=last,start_time=start,full_decode=True)


def technical(r):
    p=ROOT/'metadata/technical'/f"{r['sample_id']}.json"
    fingerprints={role:dict(path=str(asset_path(r,role).resolve()),bytes=asset_path(r,role).stat().st_size,
                         mtime_ns=asset_path(r,role).stat().st_mtime_ns) for role in roles(r)}
    if p.exists():
        old=read(p)
        if old.get('fingerprints')==fingerprints:return old
    try:
        media={}
        for role in roles(r):
            f=asset_path(r,role);info=dict(path=str(f.resolve()),sha256=sha(f),bytes=f.stat().st_size)
            if role=='reference':
                from PIL import Image
                with Image.open(f) as im:im.verify()
                with Image.open(f) as im:im.load();info.update(width=im.width,height=im.height,readable=True)
            else:info.update(video_probe(f))
            media[role]=info
        # Last decoded frame timestamp is the conservative common usable endpoint.
        end=min(media[role]['last_frame_seconds'] for role in ('source','target'))
        if r['dataset']=='coinve' and end+1e-7<4:raise ValueError('common_interval_under_4_seconds')
        times=[end*f for f in (0,.25,.5,.75,.99)]
        indices={role:[min(media[role]['frames']-1,round(t*media[role]['fps'])) for t in times] for role in ('source','target')}
        for role in ('source','target'):similarity.decode(media[role]['path'],indices[role],FFMPEG)
        out=dict(sample_id=r['sample_id'],status='ok',media=media,common_start_seconds=0,
                 common_end_seconds=end,sample_times_seconds=times,frame_indices=indices,fingerprints=fingerprints)
    except Exception as e:
        out=dict(sample_id=r['sample_id'],status='technical_reject',reason=str(e) if isinstance(e,ValueError) else type(e).__name__,fingerprints=fingerprints)
    save(p,out);return out


def score(r,t):
    p=ROOT/'metadata/scores'/f"{r['sample_id']}.json"
    hashes={role:t['media'][role]['sha256'] for role in ('source','target')}
    script_hash=sha(TRIP/'score_pair_similarity.py')
    if p.exists():
        old=read(p)
        if old.get('media_sha256')==hashes and old.get('script_sha256')==script_hash:return old
    cached=next((s for s in SCORE_CACHE if s['sample_id']==r['sample_id'] or
        (s['category']==r['category'] and Path(s['source']).name==Path(r['source_member']).name
         and Path(s['target']).name==Path(r['target_member']).name)),None)
    if cached and all(sha(cached[role])==hashes[role] for role in ('source','target')):
        result=dict(cached,reused_score=True,reused_score_sample_id=cached['sample_id'],sample_id=r['sample_id'])
    else:
        row=dict(sample_id=r['sample_id'],category=r['category'],**{role:t['media'][role]['path'] for role in ('source','target')})
        for role in ('source','target'):
            row[role+'_frames']=t['media'][role]['frames'];row[role+'_fps']=t['media'][role]['fps']
        result=similarity.score_pair(row,FFMPEG,FFPROBE)
    result.update(media_sha256=hashes,script_sha256=script_hash,algorithm_version=similarity.VERSION)
    save(p,result);return result


def process(r):
    record_path=ROOT/'metadata/technical'/f"{r['sample_id']}.json"
    if record_path.exists():
        cached=read(record_path)
        if (cached.get('reason')=='official_asset_missing' and cached.get('revision')==r['revision']
                and cached.get('repo_id')==r['repo_id']
                and cached.get('missing_member')==r.get(cached.get('missing_role','')+'_member')):
            return cached
    try:
        fetch(r) # Transient network failures propagate and never become exclusions.
    except MissingAssetError as error:
        result=dict(sample_id=r['sample_id'],status='technical_reject',reason='official_asset_missing',
            repo_id=r['repo_id'],revision=r['revision'],missing_role=error.role,missing_member=error.member,
            evidence=dict(http_status=404,huggingface_error_code='EntryNotFound'),
            pairing_complete=False,network_failure=False)
        save(record_path,result);return result
    t=technical(r)
    if t['status']=='ok' and r['dataset']=='tripvvt':score(r,t)
    return t


def run_batch(rows,workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(process,r):r for r in rows}
        failures=[]
        for future in as_completed(futures):
            r=futures[future]
            try:
                result=future.result();print(r['sample_id'],result['status'],flush=True)
            except Exception as e:
                failures.append(dict(sample_id=r['sample_id'],error_type=type(e).__name__,http_status=getattr(e,'http_status',None)))
                print(r['sample_id'],'retry_required',type(e).__name__,flush=True)
    if failures:
        save(ROOT/'metadata/retry_required.json',failures)
        raise RuntimeError('batch incomplete; retry original candidates before any replenishment')


def individual_stage(phase, manifest, workers):
    """Optional isolated stages over an explicit frozen JSONL subset."""
    rows=jsonl(manifest)
    assert rows and len({r['sample_id'] for r in rows})==len(rows)
    def one(r):
        if phase=='download':fetch(r);return dict(sample_id=r['sample_id'],status='downloaded_or_reused')
        if phase=='probe':return technical(r)
        record=read(ROOT/'metadata/technical'/f"{r['sample_id']}.json")
        assert record['status']=='ok'
        return score(r,record) if r['dataset']=='tripvvt' else dict(sample_id=r['sample_id'],status='not_applicable')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result=list(pool.map(one,rows))
    save(ROOT/'metadata'/(phase+'_stage.json'),dict(input_manifest_sha256=sha(manifest),records=result))


def similarity_exclusions(rows):
    excluded=set();report={}
    for cat in CATEGORIES:
        values=[r for r in rows if r['category']==cat and good(r)]
        scores=[read(ROOT/'metadata/scores'/f"{r['sample_id']}.json") for r in values]
        high=[s for s in scores if s['phash_median']<=2 and s['ssim_median']>=.95 and s['rgb_mae_median']<=.02 and s['roi_mae_median']<=.035]
        high.sort(key=lambda s:(s['phash_median'],s['rgb_mae_median'],s['roi_mae_median'],-s['ssim_median'],s['sample_id']))
        remove=high[:len(values)*15//100];excluded.update(s['sample_id'] for s in remove)
        report[cat]=dict(technical_qualified=len(values),high_similarity=len(high),excluded=len(remove),
                         rate=len(remove)/len(values) if values else 0,cap=len(values)*15//100)
    total=sum(v['technical_qualified'] for v in report.values())
    return excluded,dict(categories=report,overall_rate=len(excluded)/total if total else 0,
        interpretation='similarity rule only; not a proven edit failure',excluded_ids=sorted(excluded))


def good(r):
    p=ROOT/'metadata/technical'/f"{r['sample_id']}.json"
    return p.exists() and read(p)['status']=='ok'


def smoke(workers):
    rows=jsonl(ROOT/'metadata/candidates.jsonl');chosen=[]
    for split in SPLITS:
        for key in quotas('coinve',split):chosen.append(next(r for r in rows if r['dataset']=='coinve' and r['split']==split and stratum(r)==key))
    for cat in CATEGORIES:chosen.append(next(r for r in rows if r['dataset']=='tripvvt' and r['category']==cat))
    run_batch(chosen,workers)
    assert all(good(r) for r in chosen),'smoke technical rejection: inspect before scaling'
    save(ROOT/'metadata/smoke.json',dict(status='passed',pairs=len(chosen),ids=[r['sample_id'] for r in chosen],cpu_only=True))


def prepare(workers):
    assert read(ROOT/'metadata/smoke.json')['status']=='passed'
    rows=jsonl(ROOT/'metadata/candidates.jsonl')
    selected=[];hash_locks=historical_hashes(workers);selected_hashes={}
    for split in SPLITS:
        for key,needed in quotas('coinve',split).items():
            values=[r for r in rows if r['dataset']=='coinve' and r['split']==split and stratum(r)==key and group_allowed(r,split)]
            accepted=[];cursor=0
            while len(accepted)<needed:
                batch=values[cursor:cursor+min(32,needed-len(accepted))]
                if not batch:raise RuntimeError('CoinVE stratum exhausted: '+split+key)
                run_batch(batch,workers);cursor+=len(batch)
                accepted.extend(r for r in batch if good(r) and allowed(r,hash_locks,selected_hashes))
            selected+=accepted[:needed]
            print('quota complete',split,key,needed,flush=True)
    trip=[r for r in rows if r['dataset']=='tripvvt'];active=[]
    limit_path=ROOT/'metadata/trip_candidate_limits.json'
    limits=read(limit_path) if limit_path.exists() else {cat:1000 for cat in CATEGORIES}
    while True:
        active=[r for r in trip if r['candidate_rank']<=limits[r['category']]]
        for i in range(0,len(active),32):run_batch(active[i:i+32],workers)
        excluded,report=similarity_exclusions(active);save(ROOT/'metadata/similarity_exclusions.json',report)
        short=[]
        for cat in CATEGORIES:
            available=[r for r in active if r['category']==cat and good(r) and r['sample_id'] not in excluded]
            if len(available)<sum(quotas('tripvvt',s)[cat] for s in SPLITS):short.append(cat)
        if not short:break
        for cat in short:
            if limits[cat]>=max(r['candidate_rank'] for r in trip if r['category']==cat):raise RuntimeError('TripVVT category exhausted '+cat)
            limits[cat]+=100
        save(limit_path,limits)
    for cat in CATEGORIES:
        available=sorted((r for r in active if r['category']==cat and good(r) and r['sample_id'] not in excluded),key=lambda r:order('final',r['sample_id']))
        used=set()
        for split in SPLITS:
            count=quotas('tripvvt',split)[cat]
            eligible=[r for r in available if r['sample_id'] not in used and group_allowed(r,split)]
            if split!='train':eligible.sort(key=lambda r:('heldout' not in HISTORY['group_locks'].get(r['group_id'],[]),order('final',r['sample_id'])))
            chosen=[]
            for r in eligible:
                item=dict(r,split=split)
                if allowed(item,hash_locks,selected_hashes):chosen.append(item);used.add(r['sample_id'])
                if len(chosen)==count:break
            if len(chosen)!=count:
                maximum=max(r['candidate_rank'] for r in trip if r['category']==cat)
                if limits[cat]>=maximum:raise RuntimeError('TripVVT source groups exhausted: '+cat+split)
                limits[cat]=min(maximum,limits[cat]+100);save(limit_path,limits)
                print('expanding candidate pool for historical/source constraints',cat,limits[cat],flush=True)
                return prepare(workers)
            selected+=chosen
    # Every selected record carries original provenance, full instructions and a relative crop manifest.
    final=[]
    for r in selected:
        t=read(ROOT/'metadata/technical'/f"{r['sample_id']}.json")
        row=dict(r,common_start_seconds=0,common_end_seconds=t['common_end_seconds'],
                 source_start_seconds=0,target_start_seconds=0,temporal_policy='same timestamps; no time stretch or tail padding')
        for role in roles(r):
            row[role]=os.path.relpath(t['media'][role]['path'],ROOT)
            row[role+'_sha256']=t['media'][role]['sha256']
        final.append(row)
    write_rows(ROOT/'manifest.jsonl',final);csv_write(ROOT/'manifest.csv',final)
    write_rows(ROOT/'metadata/tripvvt_active_candidates.jsonl',active)
    save(ROOT/'metadata/preparation.json',dict(status='prepared',total=len(final),trip_candidate_limits=limits))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['candidates','refresh_history','download','probe','score','sample','smoke','prepare','handoff','all']);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--input-manifest',type=Path,help='Required explicit JSONL candidate subset for isolated download/probe/score')
    a=p.parse_args();assert a.workers>0
    global HISTORY,SCORE_CACHE
    for phase in (['candidates','prepare','handoff'] if a.phase=='all' else [a.phase]):
        if phase!='candidates':
            HISTORY=read(ROOT/'metadata/history.json');SCORE_CACHE=read(HISTORY['score_cache'])['records']
        print('phase',phase,flush=True)
        if phase in ('download','probe','score'):
            if not a.input_manifest:p.error('--input-manifest is required for isolated stages')
            individual_stage(phase,a.input_manifest,a.workers)
        elif phase=='handoff':
            from write_handoff import generate
            print(generate(),flush=True)
        elif phase=='sample':prepare(a.workers)
        elif phase in ('smoke','prepare'):globals()[phase](a.workers)
        else:globals()[phase]()

if __name__=='__main__':main()
