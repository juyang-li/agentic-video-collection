"""Generate the local dataset handoff from the completed preparation manifest."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def generate():
    preparation=json.loads((ROOT/'metadata/preparation.json').read_text())
    if preparation['status']!='prepared' or preparation['total']!=11000:raise RuntimeError('completed 11K preparation is required')
    rows=[json.loads(line) for line in (ROOT/'manifest.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=11000:raise RuntimeError('final manifest is incomplete')
    from collections import Counter
    import prepare as p
    report=json.loads((ROOT/'metadata/similarity_exclusions.json').read_text())
    p.csv_write(ROOT/'crops.csv',[{k:r[k] for k in ('sample_id','dataset','split','source','target','source_start_seconds','target_start_seconds','common_end_seconds','temporal_policy')} for r in rows])
    summary=dict(status='prepared_local',total=len(rows),dataset_split_counts=dict(Counter(r['dataset']+':'+r['split'] for r in rows)),
                 manifest_sha256=p.sha(ROOT/'manifest.jsonl'),independent_acceptance_performed=False,
                 review_package_requested=False,upload_requested=False,usage_document='HANDOFF_FOR_COLLEAGUE.txt',reader='dataset_reader.py')
    p.save(ROOT/'local_dataset_summary.json',summary)
    lines=[
        'CoinVE + TripVVT 11K：给师兄的接入说明', '',
        '状态：11,000 对本地准备完成。按用户最新要求，不执行独立验收、不制作新增审核包、不打包上传。准备阶段的必要技术筛选和 CPU 相似度评分保留。',
        '', '1. 使用路径',
        f'共享盘根目录：{ROOT}',
        f'完整清单：{ROOT}/manifest.jsonl',
        f'表格清单：{ROOT}/manifest.csv',
        f'共同裁剪区间：{ROOT}/crops.csv',
        f'Python 读取接口：{ROOT}/dataset_reader.py',
        f'本地统计：{ROOT}/local_dataset_summary.json',
        '本次不新增正式 ZIP 或网盘上传。此前已交付的随机 50 对滚动审核包仍保留，但不是正式训练数据入口。',
        '共享盘清单引用本次下载和复用的历史原媒体；必须保持现有相对目录结构。清单路径始终相对清单文件所在目录，不能相对当前工作目录。',
        '', '2. 数量与条件',
        'CoinVE：训练 8,000，验证 400，测试 400；TripVVT：训练 2,000，验证 100，测试 100。',
        '总计训练 10,000，验证 500，测试 500。CoinVE/TripVVT 为 80%/20%。',
        'CoinVE：两任务/三任务为 70%/30%，首条操作 Add/Remove/Replace 为 25%/30%/45%，交叉分层；所有子指令和顺序完整保留。含任何 Stylize 的整对排除。',
        'TripVVT：训练 upper_body/lower_body/dresses 为 667/667/666；验证、测试各为 33/33/34。',
        '', '3. 字段与模型输入',
        'sample_id：稳定样本 ID；dataset：coinve 或 tripvvt；split：train、validation、test；group_id：原始媒体来源组。',
        'source / target：原始输入/配对目标视频；*_sha256：原始媒体 SHA256。视频未按某个训练模型统一转码。',
        'prompt：完整编辑指令。CoinVE instructions 为有序原子指令数组，operations 为对应操作数组；不要只取第一条指令。',
        'reference：TripVVT 的官方服装参考图，是必要输入条件；CoinVE 无该字段，不虚构参考条件。',
        'caption：TripVVT 官方场景/目标描述，单独保留，不能替代换衣指令。TripVVT prompt 来自明确引用参考图的本地模板。',
        'original_metadata / repo_id / revision / metadata_index：可复查的官方来源。mask/pose 仅保留来源路径，本次不承诺交付这些附加条件。',
        'CoinVE 接口：source video + 完整有序 prompt -> target video。',
        'TripVVT 接口：source video + garment reference image + prompt -> real target video。',
        '', '4. 时间轴与训练读取',
        'common_start_seconds / source_start_seconds / target_start_seconds 均为 0；common_end_seconds 是两侧共同有效区间结束。',
        'CoinVE 共同区间至少 4 秒。以同一组绝对时间戳解码 source 和 target，不能分别按整段百分比取帧。',
        '不拉伸时间轴，不补齐长尾。原文件 fps、帧数或时长可以不同；训练实现负责选择兼容的 fps、帧数、空间缩放。',
        'dataset_reader.sample_times(...) 会在指定帧数超出共同区间时抛错。较短的 TripVVT 没有 4 秒门槛，训练侧应按样本区间选择帧数。',
        '', '5. 最小接入示例',
        'from dataset_reader import iter_samples, sample_times',
        "for row in iter_samples('manifest.jsonl', split='train', dataset='tripvvt'):",
        '    times = sample_times(row, fps=12, frames=49)',
        "    # 用训练侧解码器在相同 times 解码 row['source'] 和 row['target']",
        "    # garment = row['reference']; instruction = row['prompt']",
        '    break',
        '', f'命令行：python {ROOT}/dataset_reader.py {ROOT}/manifest.jsonl --split train --dataset coinve --fps 12 --frames 49 --limit 1',
        'read_example.py 另外提供一次相同时间戳的 FFmpeg PNG 抽帧示例，需要 ffmpeg 在 PATH 中。',
        '', '6. 隔离与质量边界',
        'CoinVE 同一原始来源组、TripVVT 同一真实 target 不跨 split；历史训练不能进入新验证/测试，历史验证/测试/冻结评测不能进入新训练。',
        '不沿用存在 target 重叠的 TripVVT 官方 train/test。历史来源约束见 metadata/history.json，媒体哈希也参与交叉检查。',
        '仅运行元数据、FFmpeg、CPU 图像指标，不使用视觉模型。技术通过不等于语义编辑效果或人工审核通过。',
        f"TripVVT 整体相似筛除率：{report['overall_rate']:.4%}；各类别均不超过 15%。",
        '相似规则四项同时满足才命中：pHash 中位数<=2、SSIM 中位数>=0.95、全图 MAE<=0.02、区域 MAE<=0.035；它不是已证实编辑失败标签。',
        '', '7. 凭据与复用',
        'local_dataset_summary.json：本地准备状态和分集统计；metadata/technical/：逐对技术探测；metadata/scores/：CPU 相似度分数；metadata/similarity_exclusions.json：相似筛除清单。技术失败保存在 metadata/technical/ 对应 JSON 的 reason 字段。',
        '原视频、参考图、配对关系、完整 prompt、裁剪区间和来源以 manifest.jsonl 为正式入口。未运行独立全量验收，不把技术合格等同于人工或语义质量通过。',
        '', '8. 执行限制',
        '所有集群作业只能提交到 testqueue，必须检查 SLURM_JOB_PARTITION；排队则休眠等待，禁止换分区。',
        'testqueue 要求申请至少 1 张 GPU，但这条数据准备流水线只使用 CPU。']
    for cat,v in report['categories'].items():lines.append(f"TripVVT {cat}: 技术合格 {v['technical_qualified']}，相似规则剔除 {v['excluded']}，比例 {v['rate']:.4%}。")
    (ROOT/'HANDOFF_FOR_COLLEAGUE.txt').write_text('\n'.join(lines)+'\n')
    return ROOT/'HANDOFF_FOR_COLLEAGUE.txt'

if __name__=='__main__':print(generate())
