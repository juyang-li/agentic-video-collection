"""Finalize the assistant's directly inspected qualitative observations."""
from collections import Counter
from common import *

if __name__ == '__main__':
    require_job()
    p=ROOT/'metadata/visual_observations.json'
    result=read(p)
    assert len(result['samples'])==96
    for sid,sample in result['samples'].items():
        assert set(sample['models'])=={'ltx_base','a849','vace'}
        for model,note in sample['models'].items():
            assert note['reviewed'] and note['output_sha256']==read(ROOT/'outputs'/model/f'{sid}.json')['output_sha256']
    result['inspection_complete']=True
    result['summary']='已直接检查全部96条任务、288条模型输出的共同0–4秒时间序列抽帧；疑点补查49帧或放大ROI，证据文件和逐条备注保留。所有有效低质量结果照常纳入，未依据输出换题、改mask或重写参考。观察为单一审核者的定性记录，success/partial/failed/uncertain不是经独立标注验证的官方编辑准确率；身份参考相似度未作量化验证。'
    result['trend']='A849相对DataMake V7无LoRA对照，在多数参考服装、背景及部分发型任务上更能执行编辑，并通常更贴近source动作和镜头；基座常保留原外观，却发生动作阶段、构图或相机漂移，若干高运动任务出现早期彩色破坏。A849也有明显失败：身份组常把参考的白色头发区域复制进输出，面部替换未稳定成立；发型组常复制参考发色、改变面部，hair_16还误改两人；年龄/去胡须及若干服装结构仍失败或部分完成。原生表情并非A849全面更好，native_20/21基座表情更明确，A849在native_24更好保持原侧脸姿态。VACE通常保持source动作较好，年龄/部分表情任务有明确变化（native_22可见泪痕），但参考身份/发型多数未完成，服装颜色/类别有变化而结构细节常不足；新增背景任务多出现额外人物/重复人物，背景替换成功与内容保持必须分开判断。TripVVT_04和_08保留左右人物目标歧义，跨VACE的编辑成功对照受限。以上只描述本批单种子现象，需与六项VBench逐项差值共同解释，不证明统计意义上的质量不变。'
    groups={}
    task_groups={r['id']:r['group'] for r in rows(ROOT/'metadata/inputs.jsonl')}
    for group in ['all','identity','hair','background','native','tripvvt']:
        subset=[s for sid,s in result['samples'].items() if group=='all' or task_groups[sid]==group]
        groups[group]={model:{field:dict(Counter(s['models'][model][field] for s in subset)) for field in ['edit','preservation']} for model in ['ltx_base','a849','vace']}
    result['qualitative_counts']=groups
    result['count_interpretation']='Single-reviewer descriptive labels only; not official accuracy or independently adjudicated success rates.'
    atomic(p,result)
    print('All 288 direct observations finalized',flush=True)
