import tempfile
from pathlib import Path
import unittest
import io
import tarfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import prepare as p

class Contracts(unittest.TestCase):
    def test_range_link_refreshes_after_403_without_changing_the_candidate(self):
        remote=object.__new__(p.coin_download.RemoteTar);remote.shard_path='pinned.tar';remote.url='expired';remote._resolved_at=time.monotonic()
        first=SimpleNamespace(status_code=403,close=lambda:None)
        second=SimpleNamespace(status_code=206,content=b'ok',close=lambda:None)
        with patch.object(remote,'refresh',side_effect=lambda:setattr(remote,'url','fresh')) as refresh:
            remote.session=SimpleNamespace(get=unittest.mock.Mock(side_effect=[first,second]))
            self.assertEqual(remote.read_range(0,1),b'ok')
            refresh.assert_called_once()
            self.assertEqual([c.args[0] for c in remote.session.get.call_args_list],['expired','fresh'])
    def test_range_link_refreshes_before_the_45_minute_limit(self):
        remote=object.__new__(p.coin_download.RemoteTar);remote.shard_path='pinned.tar';remote.url='aged';remote._resolved_at=time.monotonic()-3600
        response=SimpleNamespace(status_code=206,content=b'ok',close=lambda:None)
        with patch.object(remote,'refresh',side_effect=lambda:setattr(remote,'url','fresh')) as refresh:
            remote.session=SimpleNamespace(get=unittest.mock.Mock(return_value=response))
            self.assertEqual(remote.read_range(0,1),b'ok');refresh.assert_called_once()
            self.assertEqual(remote.session.get.call_args.args[0],'fresh')
    def test_missing_pinned_asset_is_technical_and_transient_errors_are_not(self):
        row=dict(sample_id='missing',dataset='tripvvt',repo_id='test/repo',revision='fixed',
                 source_member='input/missing.mp4',target_member='target.mp4',reference_member='ref.png')
        with tempfile.TemporaryDirectory() as temp,patch.object(p,'ROOT',Path(temp)):
            with patch.object(p,'fetch',side_effect=p.MissingAssetError('source',row['source_member'])):
                result=p.process(row)
            self.assertEqual(result['status'],'technical_reject');self.assertEqual(result['reason'],'official_asset_missing')
            self.assertFalse(result['network_failure'])
            with patch.object(p,'fetch',side_effect=AssertionError('immutable 404 should be reused')):
                self.assertEqual(p.process(row),result)
            with patch.object(p,'fetch',side_effect=RuntimeError('transient')):
                with self.assertRaises(RuntimeError):p.process(dict(row,revision='different'))
            self.assertEqual(p.read(Path(temp)/'metadata/technical/missing.json')['revision'],'fixed')
    def test_tar_header_walk_skips_video_payloads_and_pax(self):
        headers=[];entries={};offset=0
        for i in range(8):
            name=f'{i:02d}.mp4';info=tarfile.TarInfo(name);info.size=2*1024*1024;info.mtime=1.25
            header=info.tobuf(format=tarfile.PAX_FORMAT)
            headers.append((offset,header));entries[name]=p.coin_download.TarEntry(name,offset+len(header)-512,info.size)
            offset+=len(header)+info.size
        class Remote:
            size=offset
            shard_path='virtual.tar'
            def read_range(self,start,end):
                result=bytearray(end-start+1)
                for pos,header in headers:
                    left=max(start,pos);right=min(end+1,pos+len(header))
                    if left<right:result[left-start:right-start]=header[left-pos:right-pos]
                return bytes(result)
        initial={k:entries[k] for k in ('00.mp4','06.mp4')}
        baseline,old_bytes=p.coin_download.locate_member(Remote(),'04.mp4',dict(initial),walk_small_ranges=False)
        walked,new_bytes=p.coin_download.locate_member(Remote(),'04.mp4',dict(initial))
        self.assertEqual(walked,baseline);self.assertEqual(walked,entries['04.mp4'])
        self.assertLess(new_bytes,old_bytes/10)
    def test_similarity_cache_alias_keeps_new_sample_identity(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(p,'ROOT',Path(temp)):
            source=Path(temp)/'input001.mp4';target=Path(temp)/'real001.mp4'
            source.write_bytes(b'source');target.write_bytes(b'target')
            cached=dict(sample_id='cloth_001',category='upper_body',source=str(source),target=str(target),status='ok',phash_median=1)
            row=dict(sample_id='tripvvt-12345',category='upper_body',source_member='input/input001.mp4',target_member='groundtruth/real001.mp4')
            technical=dict(media={'source':{'sha256':p.sha(source)},'target':{'sha256':p.sha(target)}})
            with patch.object(p,'SCORE_CACHE',[cached],create=True),patch.object(p.similarity,'score_pair',side_effect=AssertionError('should reuse frozen score')):
                result=p.score(row,technical)
            self.assertEqual(result['sample_id'],'tripvvt-12345')
            self.assertEqual(result['reused_score_sample_id'],'cloth_001')
            self.assertTrue(result['reused_score'])
    def test_historical_group_and_content_aliases(self):
        with patch.object(p,'HISTORY',{'group_locks':{'frozen':['heldout'],'seen':['train'],'mixed':['heldout','train']}},create=True):
            self.assertFalse(p.group_allowed({'group_id':'frozen'},'train'))
            self.assertTrue(p.group_allowed({'group_id':'frozen'},'test'))
            self.assertFalse(p.group_allowed({'group_id':'seen'},'validation'))
            self.assertFalse(p.group_allowed({'group_id':'mixed'},'train'))
        with tempfile.TemporaryDirectory() as temp,patch.object(p,'ROOT',Path(temp)):
            p.save(Path(temp)/'metadata/technical/example.json',dict(status='ok',media={'source':{'sha256':'source'},'target':{'sha256':'target'}}))
            row=dict(sample_id='example',split='train')
            self.assertFalse(p.allowed(row,{'source':{'heldout'}},{}))
            self.assertEqual(p.read(Path(temp)/'metadata/technical/example.json')['status'],'ok')
            seen={};self.assertTrue(p.allowed(row,{},seen))
            self.assertFalse(p.allowed(dict(row,split='test'),{},seen))
    def test_tar_index_reuses_validated_headers(self):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w',format=tarfile.USTAR_FORMAT) as archive:
            for name in ('a.mp4','b.mp4','c.mp4'):
                entry=tarfile.TarInfo(name);entry.size=100;archive.addfile(entry,io.BytesIO(bytes(100)))
        class Remote:
            size=len(stream.getvalue())
            calls=0
            def read_range(self,start,end):
                self.calls+=1;return stream.getvalue()[start:end+1]
        remote=Remote();index={}
        first,n=p.coin_download.locate_member(remote,'b.mp4',index)
        self.assertEqual(first.name,'b.mp4');self.assertGreater(n,0)
        second,n=p.coin_download.locate_member(remote,'c.mp4',index)
        self.assertEqual(second.name,'c.mp4');self.assertEqual(n,0);self.assertEqual(remote.calls,1)
    def test_quotas(self):
        for ds in p.COUNTS:
            for split,n in p.COUNTS[ds].items():self.assertEqual(sum(p.quotas(ds,split).values()),n)
        self.assertEqual(p.quotas('coinve','train'),{'2:add':1400,'2:remove':1680,'2:replace':2520,'3:add':600,'3:remove':720,'3:replace':1080})
    def test_original_source_group(self):
        self.assertEqual(p.coin_group('abc_0_0to300.mp4'),p.coin_group('abc_4_200to800_abcd.mp4'))
        self.assertNotEqual(p.coin_group('abc_0_0to300.mp4'),p.coin_group('xyz_0_0to300.mp4'))
        self.assertEqual(p.coin_group('magictime_123_0001_0002.mp4'),p.coin_group('magictime_123.mp4'))
    def test_similarity_cap_and_conjunction(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(p,'ROOT',Path(temp)),patch.object(p,'good',return_value=True):
            rows=[]
            for cat in p.CATEGORIES:
                for i in range(20):
                    sid=f'{cat}_{i:02d}';rows.append(dict(category=cat,sample_id=sid))
                    score=dict(sample_id=sid,phash_median=1,ssim_median=.99,rgb_mae_median=.01+i*.00001,roi_mae_median=.01)
                    if cat=='lower_body' and i>0:score['ssim_median']=.94
                    if cat=='dresses':score['roi_mae_median']=.04
                    p.save(Path(temp)/'metadata/scores'/f'{sid}.json',score)
            ids,report=p.similarity_exclusions(rows)
            self.assertEqual(ids,{'upper_body_00','upper_body_01','upper_body_02','lower_body_00'})
            self.assertEqual(report['categories']['upper_body']['rate'],.15)
            self.assertEqual(report['categories']['lower_body']['excluded'],1)
            self.assertEqual(report['categories']['dresses']['excluded'],0)
    def test_frozen_candidates(self):
        path=p.ROOT/'metadata/candidates.jsonl'
        if not path.exists():self.skipTest('candidate generation pending')
        rows=p.jsonl(path);trip=[r for r in rows if r['dataset']=='tripvvt']
        self.assertEqual(len(trip),len({r['target_member'] for r in trip}))
        for cat in p.CATEGORIES:self.assertGreaterEqual(sum(r['category']==cat for r in trip),1000)
        groups={}
        for r in rows:
            if r['dataset']!='coinve':continue
            self.assertEqual(groups.setdefault(r['group_id'],r['split']),r['split'])
            self.assertIn(r['instruction_count'],(2,3));self.assertFalse(any('styl' in op.lower() for op in r['operations']))
            self.assertEqual(r['original_metadata']['instruction'],r['instructions'])
        for split in p.SPLITS:
            for key,n in p.quotas('coinve',split).items():
                self.assertGreaterEqual(sum(r['dataset']=='coinve' and r['split']==split and p.stratum(r)==key for r in rows),n)

if __name__=='__main__':unittest.main()
