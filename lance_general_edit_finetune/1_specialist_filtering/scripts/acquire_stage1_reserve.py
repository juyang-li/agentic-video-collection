#!/usr/bin/env python3
"""Acquire frozen reserve media with atomic writes and explicit failure receipts."""
from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from acquire_tar_media import Destination, extract_wanted, safe_output
from run_stage1_metadata import read_csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'dataset_research'))
from remote_zip_extract import Entry, extract_member


def atomic_json(path, value):
    temporary = path.with_suffix('.part.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--archive-workers', type=int, default=2)
    ap.add_argument('--ego-workers', type=int)
    args = ap.parse_args()
    ego_workers = args.ego_workers if args.ego_workers is not None else args.workers
    if min(args.workers,args.archive_workers,ego_workers) < 1:
        raise ValueError('worker counts must be positive')
    out = args.run_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    import requests

    rows = read_csv(out/'manifest.csv')
    archives, ego = defaultdict(dict), {}
    for row in rows:
        for role in ('source', 'target'):
            path = safe_output(row[role+'_local'])
            locator = row[role+'_locator']
            if row['dataset'] == 'egoedit':
                ego[locator] = path
            else:
                archives[(row['repo_id'],row['revision'],row[role+'_archive'])][path] = Destination(path,row['dataset'],locator)
    state_path = out/'acquisition_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'archives':{},'ego':{},'errors':{}}
    manifest_sha = file_sha256(out/'manifest.csv')
    if state.get('manifest_sha256', manifest_sha) != manifest_sha:
        raise ValueError('Acquisition state belongs to another frozen manifest')
    state.update(status='running',manifest=str(out/'manifest.csv'),manifest_sha256=manifest_sha,pid=os.getpid(),
                 archive_workers=args.archive_workers,range_workers_per_archive=args.workers,ego_workers=ego_workers)
    lock = threading.Lock()

    def record(section, key, value):
        with lock:
            state[section][key] = value
            atomic_json(out/'acquisition_state.json', state)

    def acquire_tar(key, destinations):
        repo, revision, name = key
        pending = [d for d in destinations if not d.path.is_file() or not d.path.stat().st_size]
        if not pending:
            previous = state['archives'].get(name,{})
            # Preserve the earlier official hash receipt across resumptions.
            value = dict(previous) if previous.get('status')=='complete' else {'status':'preserved','wanted':len(destinations)}
            value['resumed_no_pending_media'] = True
            record('archives', name, value)
            return
        cache = args.cache_dir/(hashlib.sha256(f'{repo}:{revision}:{name}'.encode()).hexdigest()[:16]+'_'+Path(name).name)
        log = out/('download_'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.log')
        for attempt in range(1,4):
            try:
                meta = get_hf_file_metadata(hf_hub_url(repo,name,repo_type='dataset',revision=revision), token=True)
                record('archives',name,{'status':'downloading','bytes':meta.size,'wanted':len(destinations),'attempt':attempt})
                if not cache.is_file() or cache.stat().st_size != meta.size:
                    with log.open('a') as stream:
                        runner = "import runpy,sys\np=sys.argv.pop(1);sys.argv[0]=p\ntry: runpy.run_path(p,run_name='__main__')\nexcept Exception as e: print('download_error_type='+type(e).__name__,flush=True);sys.exit(1)\n"
                        proc = subprocess.run([sys.executable, '-c', runner, str(ROOT/'scripts/download_hf_ranges.py'),
                            '--repo-id',repo,'--revision',revision,'--filename',name,
                            '--output',str(cache),'--workers',str(args.workers)], stdout=stream, stderr=stream)
                    if proc.returncode:
                        raise RuntimeError(f'download_exit_{proc.returncode}')
                sha = file_sha256(cache)
                if meta.etag and len(meta.etag)==64 and sha!=meta.etag:
                    cache.rename(cache.with_suffix('.bad-sha256'))
                    raise ValueError('archive_sha256_mismatch')
                extracted, missing = extract_wanted(cache, pending)
                record('archives',name,{'status':'complete' if not missing else 'complete_with_missing',
                    'bytes':meta.size,'sha256':sha,'hf_etag':meta.etag,'wanted':len(destinations),
                    'newly_extracted':extracted,'missing_locators':missing})
                print(f'archive={name} extracted={extracted} missing={len(missing)}',flush=True)
                return
            except Exception as error:
                # Never persist signed URLs or authentication headers in status output.
                record('errors', name, {'type':type(error).__name__,'attempt':attempt})
                if attempt==3:
                    record('archives',name,{'status':'acquisition_error','wanted':len(destinations),'error_type':type(error).__name__})
                    return
                time.sleep(5*attempt)

    index = {}
    index_path = ROOT.parents[1]/'dataset_research/egoedit_research/zip_members.csv'
    with index_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['basename'] in ego:
                index[row['basename']] = row
    class HfArchive:
        def __init__(self,name):
            self.name = name
            self.url = hf_hub_url('liguang0115/EgoEdit',name,repo_type='dataset',revision='a9bd3faafe51a581b359acd7c6ad5c432508ff60')
        def read(self, offset, length):
            for attempt in range(5):
                try:
                    headers=build_hf_headers(token=True)
                    headers['Range']=f'bytes={offset}-{offset+length-1}'
                    response=requests.get(self.url,headers=headers,timeout=120)
                    expected=f'bytes {offset}-{offset+length-1}/'
                    if response.status_code !=206 or len(response.content)!=length or not response.headers.get('Content-Range','').startswith(expected):
                        raise RuntimeError('invalid_range_response')
                    return response.content
                except Exception:
                    if attempt==4: raise RuntimeError('range_exhausted') from None
                    time.sleep(min(2**attempt,10))
    remote = {}
    ego_completed = 0
    def acquire_ego(name,path):
        nonlocal ego_completed
        try:
            row = index[name]
            entry = Entry(name=row['name'],method=int(row['method']),flags=int(row['flags']),
                crc32=int(row['crc32'],16),compressed_size=int(row['compressed_size']),
                uncompressed_size=int(row['uncompressed_size']),local_offset=int(row['local_offset']))
            with lock:
                archive = remote.setdefault(row['archive'], HfArchive(row['archive']))
            valid=False
            if path.is_file() and path.stat().st_size==entry.uncompressed_size:
                valid=(binascii.crc32(path.read_bytes()) & 0xffffffff)==entry.crc32
            if not valid:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.download.part')
                extract_member(archive,entry,temporary)
                temporary.replace(path)
            with lock:
                ego_completed+=1
                state['ego'][name]={'status':'complete','bytes':entry.uncompressed_size,'crc32':row['crc32']}
                if ego_completed%50==0:
                    atomic_json(out/'acquisition_state.json',state)
                    print(f'ego_complete={ego_completed}/{len(ego)}',flush=True)
        except Exception as error:
            record('ego',name,{'status':'acquisition_error','error_type':type(error).__name__})

    def tar_loop():
        # Temporary copies stay on node-local storage.
        with ThreadPoolExecutor(max_workers=args.archive_workers) as pool:
            tasks=[pool.submit(acquire_tar,key,list(values.values())) for key,values in sorted(archives.items())]
            for task in as_completed(tasks): task.result()

    atomic_json(out/'acquisition_state.json',state)
    with ThreadPoolExecutor(max_workers=1) as pool:
        tar_future=pool.submit(tar_loop)
        with ThreadPoolExecutor(max_workers=ego_workers) as ego_pool:
            tasks=[ego_pool.submit(acquire_ego,name,path) for name,path in sorted(ego.items())]
            for task in as_completed(tasks): task.result()
        tar_future.result()
    missing=[]
    for row in rows:
        for role in ('source','target'):
            path=safe_output(row[role+'_local'])
            if not path.is_file() or path.stat().st_size==0:
                missing.append({'sample_id':row['sample_id'],'role':role,'local_path':row[role+'_local']})
    state.update(status='complete' if not missing else 'complete_with_unavailable_media',missing_role_files=missing)
    atomic_json(out/'acquisition_state.json',state)
    print(f'acquisition_finished missing_role_files={len(missing)}',flush=True)


if __name__=='__main__':
    main()
