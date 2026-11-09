#!/usr/bin/env python3
"""Build the 50-card offline dataset audit, with truthful pairing semantics."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import io
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile

import cv2
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent
NAME='ref_edit_clothing25_motion25_review_20261030'
PACKAGE=ROOT/'review'/NAME
FIELDS=['person_relevance','instruction_correctness','logical_correctness','content_preservation',
        'temporal_consistency','visual_quality','technical_quality','reference_quality','instruction_feasibility']
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()


def write_html(cards):
    from fontTools import subset
    from fontTools.ttLib import TTFont
    template=(ROOT/'review_template.html').read_text()
    payload=json.dumps({'version':NAME,'cards':cards},ensure_ascii=False).replace('<','\\u003c')
    text=template+payload+''.join(chr(i) for i in range(32,127))
    font=TTFont(ROOT/'assets/font/NotoSansCJKsc-Regular.otf')
    sub=subset.Subsetter();sub.populate(text=text);sub.subset(font)
    buf=io.BytesIO();font.save(buf)
    face="@font-face{font-family:ReviewCJK;src:url(data:font/otf;base64,"+base64.b64encode(buf.getvalue()).decode()+") format('opentype');font-weight:100 900;font-display:block;}"
    (PACKAGE/'index.html').write_text(template.replace('__REVIEW_DATA__',payload).replace('__FONT_FACE__',face))
    shutil.copy2(ROOT/'assets/font/LICENSE.txt',PACKAGE/'FONT_LICENSE.txt')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def media_info(path):
    c=cv2.VideoCapture(str(path));frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS)
    width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=int(c.get(cv2.CAP_PROP_FOURCC));c.release()
    assert frames>0 and fps>0 and width>0 and height>0,path
    return {'frames':frames,'fps':fps,'duration':round(frames/fps,6),'width':width,'height':height,
            'fourcc':''.join(chr((fourcc>>8*i)&255) for i in range(4))}


def faststart(path):
    boxes={};size=path.stat().st_size
    with path.open('rb') as f:
        pos=0
        while pos+8<=size:
            f.seek(pos);n,kind=struct.unpack('>I4s',f.read(8));header=8
            if n==1:n=struct.unpack('>Q',f.read(8))[0];header=16
            if n==0:n=size-pos
            if n<header:raise RuntimeError('invalid MP4 atom')
            boxes.setdefault(kind,pos);pos+=n
    return boxes.get(b'moov',size)<boxes.get(b'mdat',0)


def transcode(job):
    source,dest=job
    if not dest.exists():
        temp=dest.with_suffix('.part.mp4')
        p=subprocess.run([FFMPEG,'-nostdin','-hide_banner','-loglevel','error','-y','-threads','2','-i',str(source),
            '-map','0:v:0','-an','-sn','-dn','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(temp)],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(f'transcode {dest.name}: {p.stderr[-1000:]}')
        temp.replace(dest)
    before=media_info(source);after=media_info(dest)
    assert abs(before['duration']-after['duration'])<max(.1,2/before['fps']),(source,before,after)
    p=subprocess.run([FFMPEG,'-nostdin','-hide_banner','-v','error','-i',str(dest),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
    assert p.returncode==0 and not p.stderr.strip(),(dest,p.stderr)
    assert faststart(dest) and after['fourcc']=='avc1',dest
    return {'file':dest.relative_to(PACKAGE).as_posix(),'original_sha256':sha(source),
            'sha256':sha(dest),'bytes':dest.stat().st_size,**after,'codec':'h264','pixel_format':'yuv420p',
            'faststart':True,'full_decode_ok':True,'unchanged_speed':True}


SUBJECTS={1:('这名女子','the woman'),2:('左侧年轻男子','the young man on the left'),3:('这名女子','the woman'),
 4:('这名女子','the woman'),5:('左侧坐着的女子','the seated woman on the left'),6:('右侧男子','the man on the right'),
 7:('坐着的女子','the seated woman'),8:('这名女子','the woman'),9:('这名女子','the woman'),
 10:('右侧女子','the woman on the right'),11:('背对镜头的女子','the woman facing away from the camera'),
 12:('右侧穿黑色上衣的女子','the woman on the right wearing a black top'),13:('这名女子','the woman'),
 14:('船上坐着的女子','the woman seated on the boat'),15:('右侧女子','the woman on the right'),
 16:('左侧女子','the woman on the left'),17:('蹲着的女子','the squatting woman'),18:('这名女子','the woman'),
 19:('坐着的女子','the seated woman'),20:('这名女子','the woman'),21:('这名女子','the woman'),
 22:('拿行李箱的女子','the woman with the suitcase'),23:('左侧女子','the woman on the left'),
 24:('站着的成年女子','the standing adult woman'),25:('右侧女子','the woman on the right')}

WARNINGS={1:'source 与 target 都是黑色西装，需检查是否近似 no-op。',
18:'官方归为 dresses，但参考图更像吊带上衣；需核查衣长和服装类别对应关系。',
21:'官方归为 dresses，但参考图是外套；需核查任务粒度和其他衣物保持。',
22:'官方归为 dresses，但参考图看似上下套装；需确认是否符合单件换装定义。',
25:'官方归为 dresses，但参考图是西装外套；需核查是否发生非目标衣物变化。'}


def motion_selection():
    current=json.loads((ROOT/'metadata/motion_reference_selection.json').read_text())
    groups={k:[r for r in current['samples'] if r['class']==k] for k in current['quota']}
    initial=json.loads((ROOT/'metadata/hmdb_initial_unfiltered_selection.json').read_text())['samples']
    def old(prefix):return next(r for r in initial if Path(r['member']).name.startswith(prefix))
    chosen={
        'wave':[old('Song_I_Can_Wave'),old('EVOLUTION_wave'),old('Chris_Benoit'),
                groups['wave'][7],groups['wave'][4],groups['wave'][10],old('veoh_harold')],
        'clap':[groups['clap'][i-1] for i in [1,3,4,7,10,11]],
        'turn':[groups['turn'][i-1] for i in [1,3,5,6,8,10]],
        'walk':[groups['walk'][i-1] for i in [1,5,6,7,8,11]],
    }
    source_ids={'wave':[1,4,7,9,10,12,15],'clap':[2,3,5,6,8,17],
                'turn':[11,13,16,20,21,24],'walk':[2,3,5,9,15,20]}
    out=[]
    for cls,rows in chosen.items():
        for row,source_id in zip(rows,source_ids[cls]):
            path=ROOT/'assets/hmdb'/cls/Path(row['member']).name
            out.append({**row,'id':f'motion_{len(out)+1:03d}','source_cloth_id':source_id,
                        'original_sha256':sha(path),'original_media':media_info(path)})
    assert len(out)==25 and len({r['original_sha256'] for r in out})==25
    write_json(ROOT/'metadata/motion_pairs_selected.json',{'samples':out,'method':'visual selection from fixed seed candidate pool; no model outputs used',
        'reference_repository':current['repo'],'revision':current['revision'],
        'pair_semantics':'independent source and action-reference videos; NO edited target GT',
        'source_note':'real, unedited groundtruth video from TripVVT, reused as SOURCE here'})
    return out,current


def build():
    PACKAGE.mkdir(parents=True,exist_ok=True);(PACKAGE/'media').mkdir(exist_ok=True)
    garment=json.loads((ROOT/'metadata/garment_selection.json').read_text())
    motion,hmdb=motion_selection();jobs=[];cards=[]
    for i,row in enumerate(garment['samples'],1):
        source=ROOT/'assets/tripvvt'/row['input'];target=ROOT/'assets/tripvvt'/row['output'];cloth=ROOT/'assets/tripvvt'/row['cloth_image']
        src=f'media/{row["sample_id"]}_source.mp4';tgt=f'media/{row["sample_id"]}_target.mp4';ref=f'media/{row["sample_id"]}_garment.png'
        jobs.extend([(source,PACKAGE/src),(target,PACKAGE/tgt)])
        if not (PACKAGE/ref).exists():shutil.copy2(cloth,PACKAGE/ref)
        region={'upper_body':('上衣','upper-body garment'),'lower_body':('下装','lower-body garment'),'dresses':('服装','garment/outfit')}[row['cloth_type']]
        zh,en=SUBJECTS[i]
        notice='已有 source＋服装图＋target 三元组。target 为真实视频，source 为合成换装视频；参考图仅有正面，没有背面或完整多图穿搭。'
        if i in WARNINGS:notice+='\n抽查提示：'+WARNINGS[i]
        panels=[{'type':'video','path':src,'label':'SOURCE · 待编辑视频','description':'TripVVT 合成 source，原速完整片段'},
                {'type':'image','path':ref,'label':'GARMENT · 目标服装参考','description':'官方提供的单张服装图'},
                {'type':'video','path':tgt,'label':'TARGET · 配对目标视频','description':'TripVVT 原始真实 target；不是本次模型生成结果'}]
        cards.append({'id':row['sample_id'],'hash':sha(source)[:5],'task':'garment_'+row['cloth_type'],'kind':'garment',
            'title':'参考图换衣 · '+{'upper_body':'上衣','lower_body':'下装','dresses':'官方 dresses 类'}[row['cloth_type']],
            'instruction_zh':f'请将输入视频中{zh}的{region[0]}替换为参考图所示衣物。只改变目标服装，保持人物身份、动作、其他衣物、其他人物、场景和镜头不变。',
            'instruction_en':f'Replace {en}\'s {region[1]} with the item shown in the reference image. Preserve identity, motion, other clothing, other people, scene, and camera.',
            'notice':notice,'panels':panels,'not_applicable':[],
            'provenance':{'dataset':'TripVVT-10K','repo':garment['repo'],'revision':garment['revision'],
               'official_split':row['official_split'],'official_row':row['official_row'],'official_category':row['cloth_type'],
               'source_member':row['input'],'reference_member':row['cloth_image'],'target_member':row['output'],
               'source_sha256':sha(source),'reference_sha256':sha(cloth),'target_sha256':sha(target),
               'original_source_spec':media_info(source),'original_target_spec':media_info(target),
               'group_id':'TripVVT:'+row['output'],'pair_semantics':'synthetic source + garment image -> real target',
               'official_caption':row['text'],'sample_selection_seed':20261030,'split_warning':'official test outputs overlap train; not a clean held-out split'}})
    action_zh={'wave':'挥手／打招呼','clap':'拍手／观众回应','turn':'全身转身','walk':'行走'}
    align={'wave':'action: 打招呼A/B、挥手','clap':'action: 观众回应（近似语义，不是同一动作资产）',
           'turn':'action: 摆臂转身（近似语义）','walk':'action: 向前行走、模特步行（近似语义）'}
    for row in motion:
        i=row['source_cloth_id'];original=garment['samples'][i-1];zh,en=SUBJECTS[i]
        source=f'media/cloth_{i:03d}_target.mp4';ref=f'media/{row["id"]}_reference.mp4'
        reference=ROOT/'assets/hmdb'/row['class']/Path(row['member']).name
        jobs.append((reference,PACKAGE/ref))
        duration=row['original_media']['duration']
        notice='待测输入对：左侧是真实 source，右侧仅演示目标动作。没有编辑后 target / GT，也没有模型结果。只迁移动作，不迁移参考人物的脸、服装、物体或背景。'
        if duration<2:notice+='\n该参考不足 2 秒：当前用于质量审核；未来正式推理需检查模型最短参考时长，不通过变速补长。'
        if row['class']=='walk':notice+='\n请特别审核行走方向／步态是否与 source 已有动作充分不同，以及当前场景是否允许行走。'
        cards.append({'id':row['id'],'hash':row['original_sha256'][:5],'task':'motion_'+row['class'],'kind':'motion',
            'title':'动作替换 · '+action_zh[row['class']],
            'instruction_zh':f'请将 source 中{zh}当前的动作替换为右侧参考视频演示的动作，遵循其肢体运动和节奏。保持 source 人物的身份、发型、服装、场景、其他人物和镜头不变；不要复制参考人物或背景。',
            'instruction_en':f'Replace the current action of {en} in the source with the movement demonstrated in the driving video, following its body motion and rhythm. Preserve the source person\'s identity, hairstyle, clothes, scene, other people and camera. Do not copy the driving actor or background.',
            'notice':notice,'not_applicable':['instruction_correctness','content_preservation'],
            'panels':[{'type':'video','path':source,'label':'SOURCE · 待编辑的真实视频','description':f'TripVVT 未修改真实视频，复用 cloth_{i:03d} 的 target 作为本条 source'},
                      {'type':'video','path':ref,'label':'DRIVING · 独立动作参考视频','description':f'HMDB51 · {action_zh[row["class"]]} · {duration:.2f} 秒；不是 target'}],
            'provenance':{'source_dataset':'TripVVT real groundtruth video','source_member':original['output'],
                'source_revision':garment['revision'],'source_sha256':sha(ROOT/'assets/tripvvt'/original['output']),
                'reference_dataset':'HMDB51','official_page':'https://serre.lab.brown.edu/hmdb51.html',
                'reference_mirror':hmdb['repo'],'reference_revision':hmdb['revision'],'reference_member':row['member'],
                'reference_sha256':row['original_sha256'],'reference_original_spec':row['original_media'],
                'senior_taxonomy_alignment':align[row['class']],'edited_target_available':False,
                'group_id':'TripVVT:'+original['output'],'reference_origin_group':Path(row['member']).name.split('_'+row['class']+'_')[0],
                'pair_semantics':'source + independent driving action reference; unpaired evaluation input',
                'selection':'fixed-seed candidate pool followed by direct visual rejection of head-only, cuts, severe ambiguity, and unsuitable actions'}})
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports=[]
        for i,f in enumerate(as_completed([pool.submit(transcode,j) for j in jobs]),1):
            reports.append(f.result());print(f'validated video {i}/{len(jobs)}',flush=True)
    reports.sort(key=lambda r:r['file']);csv_write(PACKAGE/'technical_validation.csv',reports)
    write_json(PACKAGE/'manifest.json',{'version':NAME,'cards':cards,'source_target_gt_cards':25,'no_target_gt_cards':25,
               'model_inference_performed':False,'default_labels_are_not_audit_results':True})
    assert len(cards)==50 and Counter(c['kind'] for c in cards)=={'garment':25,'motion':25}
    audit=[{'id':c['id'],'hash':c['hash'],'task':c['task'],**{f:'1' for f in FIELDS},'reviewed':False,'notes':''} for c in cards]
    if not (PACKAGE/'audit.csv').exists():csv_write(PACKAGE/'audit.csv',audit)
    inventories=[{'id':c['id'],'hash':c['hash'],'task':c['task'],'source':c['panels'][0]['path'],
                  'reference':c['panels'][1]['path'],'target':c['panels'][2]['path'] if len(c['panels'])==3 else '',
                  'has_target_gt':c['kind']=='garment','instruction':c['instruction_zh'],
                  'group_id':c['provenance']['group_id']} for c in cards]
    csv_write(PACKAGE/'inventory.csv',inventories)
    write_html(cards)
    readme='''# 换衣与动作替换数据审核包（50 条）

解压后用 Chrome / Edge 打开 index.html，无需服务器或联网。

- 换衣 25 条：上衣 9、下装 8、官方 dresses 类 8。SOURCE + GARMENT + TARGET 均有；来自 TripVVT。source 是合成换装视频，target 是真实视频，不是本次模型输出。
- 动作 25 条：挥手 7、拍手 6、全身转身 6、行走 6。SOURCE + DRIVING VIDEO；没有编辑后 target / GT。source 是 TripVVT 未修改真实视频，reference 是独立 HMDB51 动作视频。拍手、转身与师兄动作类别仅为近似语义对应。
- 动作 source 复用了换衣部分的真实 target，也有跨动作复用。inventory 的 group_id 明确标识；50 条审核记录不是 50 个独立 source，也不用于显著性结论。
- 所有视频保留原速和完整时长，仅转 H.264/yuv420p/faststart 并去音轨。低分辨率参考没有上采样伪装高清。短于 2 秒的参考有单独提示，正式测评前需筛选兼容性。
- 服装只有单张正面参考；没有制造背面图。整套多图、舞蹈、系鞋带等不在本包真实参考覆盖内。
- 部分 TripVVT dresses 记录的实际参考像外套或套装，已逐条提示，保留供用户判断数据质量。官方 caption 与类别原样保存在 manifest，不冒充局部编辑指令。
- 正式评测前检查身份/视频分组隔离；TripVVT 官方 train/test 存在原视频重叠。本包不是已验证干净的 held-out benchmark。

审核：1 通过、0 不通过、2 存疑，默认 1 仅为填写起点；另勾选“已人工审核”。动作项无 GT 的两个结果指标禁用、不计入通过率。支持左右键、类别筛选、同步从头播放、原生播放器全屏、CSV/JSON 导入导出；修改不覆盖随包 audit.csv。

来源：
- TripVVT: https://huggingface.co/datasets/TripVVT/TripVVT-10K （CC BY-NC 4.0）
- HMDB51: https://serre.lab.brown.edu/hmdb51.html （使用官方页面列出的 MichiganNLP/hmdb 镜像；来源与 revision 逐条记录）

manifest.json 为逐条语义、来源、原始哈希与规格；inventory.csv 为紧凑总目录；audit.csv 为默认审核表；technical_validation.csv 为逐视频完整解码和技术检查；SHA256SUMS 校验包内文件。
'''
    (PACKAGE/'README.md').write_text(readme)
    write_json(PACKAGE/'package_summary.json',{'cards':50,'garment':25,'motion':25,'videos':len(reports),
        'garment_images':25,'unique_motion_sources':len({c['panels'][0]['path'] for c in cards if c['kind']=='motion'}),
        'technical_full_decode_passed':len(reports),'model_outputs':0,'offline':True,'status':'awaiting_browser_validation'})
    (ROOT/'README.md').write_text('# 单轮参考编辑数据审核\n\n交付 25 组 TripVVT 换衣三元组与 25 组独立动作参考输入对。动作无编辑后 GT。\n\n运行 prepare_assets.py、inspect_assets.py、build_review.py；浏览器验收通过后运行 build_review.py --finalize。详见包内 README 与 manifest。\n')
    print(PACKAGE,flush=True)


def finalize():
    browser=json.loads((PACKAGE/'browser_validation.json').read_text())
    assert browser['status']=='passed' and browser['cards_checked']==50 and browser['external_requests']==0
    assert browser['index_sha256']==sha(PACKAGE/'index.html'),'browser check is stale'
    summary=json.loads((PACKAGE/'package_summary.json').read_text());summary['status']='validated';write_json(PACKAGE/'package_summary.json',summary)
    files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (PACKAGE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACKAGE).as_posix()}\n' for p in files))
    archive=PACKAGE.with_suffix('.zip')
    if archive.exists():
        uploaded=archive.with_suffix('.baidu_upload.json')
        assert not uploaded.exists(),'do not replace an uploaded release'
        suffix=sha(archive)[:8]
        archive.rename(archive.with_name(archive.stem+'.local_draft_'+suffix+'.zip'))
        old_receipt=archive.with_suffix('.sha256.json')
        if old_receipt.exists():old_receipt.rename(archive.with_name(archive.stem+'.local_draft_'+suffix+'.sha256.json'))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert {n.split('/')[0] for n in z.namelist()}=={NAME}
        assert not any('.baidu' in n or 'token' in n.lower() for n in z.namelist())
    receipt={'archive':str(archive.resolve()),'bytes':archive.stat().st_size,'sha256':sha(archive),'cards':50,
        'remote_path':'/apps/bypy/lance_review/'+archive.name,'zip_crc_passed':True,
        'upload_status':'pending','sha256_scope':'local ZIP'}
    write_json(archive.with_suffix('.sha256.json'),receipt);print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');parser.add_argument('--refresh-page',action='store_true');args=parser.parse_args()
    if args.refresh_page:write_html(json.loads((PACKAGE/'manifest.json').read_text())['cards'])
    elif args.finalize:finalize()
    else:build()
