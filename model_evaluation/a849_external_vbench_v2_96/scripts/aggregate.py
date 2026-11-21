"""Per-video metrics and paired group differences, with explicit missingness."""
import argparse
import csv
import math
import statistics
from common import *
from score import DIMS

def run(stage):
    require_job()
    frozen=read(ROOT/'metadata/freeze.json');inputs=rows(ROOT/'metadata/inputs.jsonl')
    selected=[r for r in inputs if stage=='all' or r['id'] in frozen['smoke']]
    folder=ROOT/'metadata/vbench'/stage
    values={};errors=[]
    for dim in DIMS:
        receipt=folder/f'{dim}_receipt.json'
        if not receipt.exists():errors.append(dict(dimension=dim,error='missing_receipt'));continue
        r=read(receipt)
        assert r['contract']['manifest_sha256']==frozen['input_manifest_sha256']
        mapping=r['contract']['mapping']
        results=[]
        if r['status']=='complete':
            path=folder/f'{dim}_eval_results.json';assert sha(path)==r['result_sha256'];results=[read(path)]
        else:
            results=list(r.get('per_video_results',{}).values());errors+=r.get('errors',[])
        for result in results:
            for entry in result[dim][1]:
                key=Path(entry['video_path']).stem
                assert key in mapping
                ident=mapping[key];raw=entry['video_results']
                if not isinstance(raw,(int,float,bool)) or not math.isfinite(float(raw)):
                    errors.append(dict(video=key,dimension=dim,error='nonfinite_score'));continue
                score=float(raw)/100 if dim=='imaging_quality' else float(raw)
                values[(ident['id'],ident['model'],dim)]=(raw,score)
    records=[];pairs=[]
    for item in selected:
        for dim in DIMS:
            paired={}
            for model in ['ltx_base','a849','vace']:
                p=ROOT/'outputs'/model/f'{item["id"]}.json'
                generation=read(p) if p.exists() else {'status':'missing'}
                v=values.get((item['id'],model,dim));paired[model]=v[1] if v is not None else None
                records.append(dict(id=item['id'],group=item['group'],subgroup=item['subgroup'],model=model,dimension=dim,status='complete' if v is not None else 'missing_score' if generation['status']=='complete' else 'missing_generation',raw_score=v[0] if v else None,score=v[1] if v else None,output_sha256=generation.get('output_sha256'),normalization='raw/100' if dim=='imaging_quality' else 'official_per_video_score'))
            a,b=paired['a849'],paired['ltx_base']
            pairs.append(dict(id=item['id'],group=item['group'],dimension=dim,base=b,a849=a,delta=a-b if a is not None and b is not None else None,vace=paired['vace']))
    prefix='' if stage=='all' else 'smoke_'
    with (ROOT/'metadata'/f'{prefix}per_video_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    path=ROOT/'metadata'/f'{prefix}paired_differences.jsonl'
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs)
    tmp=path.with_suffix('.tmp');tmp.write_text(content);tmp.replace(path)
    groups={}
    for group in ['all','identity','hair','background','native','tripvvt']:
        group_rows=[r for r in records if group=='all' or r['group']==group]
        group_pairs=[r for r in pairs if group=='all' or r['group']==group]
        groups[group]={}
        for dim in DIMS:
            models={}
            for model in ['ltx_base','a849','vace']:
                scores=[r['score'] for r in group_rows if r['dimension']==dim and r['model']==model and r['score'] is not None]
                models[model]=dict(n=len(scores),mean=statistics.mean(scores) if scores else None)
            matched=[r for r in group_pairs if r['dimension']==dim and r['delta'] is not None]
            deltas=[r['delta'] for r in matched]
            groups[group][dim]=dict(models=models,paired_n=len(matched),a849_minus_base_mean=statistics.mean(deltas) if deltas else None,delta_median=statistics.median(deltas) if deltas else None,positive=sum(d>0 for d in deltas),negative=sum(d<0 for d in deltas),zero=sum(d==0 for d in deltas),paired_base_mean=statistics.mean(r['base'] for r in matched) if matched else None,paired_a849_mean=statistics.mean(r['a849'] for r in matched) if matched else None)
    atomic(ROOT/'metadata'/f'{prefix}group_metrics.json',dict(groups=groups,errors=errors,expected_scores=len(selected)*18,valid_scores=len(values),manifest_sha256=frozen['input_manifest_sha256'],missing_values='null/blank, never zero-filled',dynamic_degree='official RAFT boolean per video; mean is dynamic fraction, not continuous motion magnitude or unconditionally better quality'))
    if stage!='all':return
    def fmt(v):return '缺失' if v is None else f'{v:.5f}'
    lines=['# A849 外部视频编辑评测：96条扩展版','',f'冻结{len(selected)}条任务，单种子42；有效逐视频指标{len(values)}/{len(selected)*18}。所有计算仅testqueue；直接视觉审核，无Gemini。','',
        '共同规格为49帧、12fps，取样0至4秒（含端点），编码长度49/12秒；保持比例补边1280×736或736×1280。LTX两组仅LoRA有无不同，v7/40步/CFG3/STG1 block28/rescale0.7/ref_cfg1。VACE40步/CFG5/flow_shift5，复用同source和参考图、source-only mask，可编辑RGB填128。','',
        '六项均使用固定revision的官方VBench custom_input。imaging_quality逐视频原值为0–100，表中除100；其余沿用官方逐视频值。dynamic_degree为逐视频布尔判定，均值是动态视频比例，不是连续运动幅度，增减须结合source动作和视觉编辑保持解释。未计算官方总分或榜单成绩。','',
        '|指标|基座均值|A849均值|A849−基座配对均值|配对n|VACE均值|','|---|---:|---:|---:|---:|---:|']
    lines.insert(4, '这里的ltx_base是同版dev权重在DataMake V7条件协议下不加载A849的对照，不是官方Ingredients工作流，也不代表官方最佳参考编辑能力。')
    for dim in DIMS:
        g=groups['all'][dim];m=g['models'];lines.append(f'|{dim}|{fmt(m["ltx_base"]["mean"])}|{fmt(m["a849"]["mean"])}|{fmt(g["a849_minus_base_mean"])}|{g["paired_n"]}|{fmt(m["vace"]["mean"])}|')
    lines+=['','## 按任务配对差值','', '|任务|指标|A849−基座均值|配对n|正/负/零|VACE均值|','|---|---|---:|---:|---|---:|']
    for group in ['identity','hair','background','native','tripvvt']:
        for dim in DIMS:
            g=groups[group][dim];lines.append(f'|{group}|{dim}|{fmt(g["a849_minus_base_mean"])}|{g["paired_n"]}|{g["positive"]}/{g["negative"]}/{g["zero"]}|{fmt(g["models"]["vace"]["mean"])}|')
    lines+=['','## 检查范围与解释','',
        'A849的849契约及750/500/250保留run.json构成训练谱系；UE、CoinVE、TripVVT共6份train/development清单哈希与训练记录一致。source、处理前后参考图实际SHA256与训练source/target/reference记录无精确重合。这未排除重编码、近重复、基座预训练重合。详见training_overlap.json。','',
        'TripVVT为已有本地独立测试划分，保留已有换衣指令模板和官方服装图；官方caption仅审核。dresses沿用数据类别，参考实际可能是连体裤或外套。OmniEdit原生24条保留原指令，含外观/服装/颜色及5条表情任务；48条为自建参考图任务，无GT。','',
        '逐条失败继续运行；缺失分数留空，不计0。统计只采用同任务、同指标均有效的配对。有效低质量输出不剔除。96条任务单种子可观察改善/退化趋势，不能证明统计意义上的质量不变。','',
        '主体/背景一致性主要衡量输出内部一致性，不直接证明编辑成功或与source相同。静止/动作丢失可能取得较高一致性与平滑度，须结合dynamic_degree及直接时间序列审核。视觉观察见visual_observations.json；输入协议检查与输出质量判断分开记录。','']
    lines+=['LTX两组均由官方H.264/YUV420 BT.709、CRF19/veryfast编码；VACE沿用PyAV RGB转YUV420、CRF20/fast，跨VACE的画质分数可能包含编码差异。VBench指标源码未修改，六项在PyTorch2.10/CUDA12.6上完成实跑检查，未证明与官方旧依赖栈逐位数值一致。完整版本和权重SHA保存在metadata。','']
    if frozen.get('revision')=='whitehair_fullframe_v2':
        lines+=['用户查看首批14条预览后明确授权取消smoke关卡、直接全量。当前4张身份参考保留完整原图及分辨率，只把分割出的头发填RGB255，非头发像素按位保持，不裁剪。4条身份任务重新生成；11条输入完整记录完全相同的旧有效输出按reuse_outputs.json复用，原始生成契约未改写。早期裁脸灰底参考和预览结果归档，不计入当前身份评测。','']
    if (ROOT/'metadata/task_limitations.json').exists():
        lines+=['TripVVT_04原指令没有明确左右人物，source含两人。VACE推理前按source规划的mask覆盖右侧人物，而A849改动了左侧人物的花纹长裤。本条编辑成功判定保留目标歧义，VACE目标范围比较受限；不根据输出换题、也未依据审核GT修改mask，全部有效输出及VBench分数照常保留。详见task_limitations.json。','']
    lines+=['TripVVT_08同样保留左右人物目标歧义：source-only VACE mask覆盖右侧女性，A849改变左侧男性上装；不将本条当作严格同对象编辑成功对照。background_05的source首帧原本为黑帧，三模型使用同一精确时间窗口，未为结果好看而移位或替换。详见task_limitations.json。','']
    if (ROOT/'metadata/vace_prompt_corrections.json').exists():
        lines+=['native_03的VACE辅助目标描述在该模型推理前，将没有原指令依据的人物称谓改为中性person；原官方指令/source/reference未变，实际prompt和修正记录保留。','']
    lines+=['身份与发型参考均去除背景至RGB128；保持完整原始画幅和分辨率。身份图仅头发填RGB255，其他人物前景像素保持原值；发型图保留真实发色。原24条历史输出只复用16个输入完整相同的任务，共48条，身份/发型组全部重新生成。','','96条任务对应83段不同source；同一视频可用于不同编辑任务，详见source_clusters.json，不当作96段独立视频。','','本轮官方Ingredients IC-LoRA只在后续队列，未纳入这三模型288条结果。','']
    visual=ROOT/'metadata/visual_observations.json'
    if visual.exists():
        observations=read(visual)
        lines+=['## 直接视觉观察','',observations.get('summary',''),'',observations.get('trend',''),'']
    (ROOT/'metadata/report.md').write_text('\n'.join(lines))
    print('valid scores',len(values),'/',len(selected)*18,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','all'],required=True);a=p.parse_args();run(a.stage)
