# frozen300：四模型VBench custom_input六项

**最终范围更新（用户于2026-11-15明确要求“不用静态筛选”）：仅交付完整300条、四模型的六项指标，共7200份分数。停止剩余RAFT筛选，不采用静态共同子集；Temporal Flickering按全量原始分数报告，受正常运动影响。已生成的部分静态缓存保留但不进入最终结果。授权记录见metadata/user_override.json。**

**已完成并通过全量验收**：7200/7200份有限分数，评分失败0；336行总体/数据集/任务均值独立复核通过。正式结果见[report/REPORT.md](report/REPORT.md)，逐视频分数见[per_video_metrics.csv](report/per_video_metrics.csv)，验收见[final_validation.json](metadata/final_validation.json)。本任务静态筛选step 202896.19按用户要求取消，allocation 202896保留；取消不计作六项指标评分失败。

依据上级VBENCH_README.md，评价现有冻结300条、四模型1200个输出，不重跑生成。模型准确名称为LTX2.5 distilled＋V7（8步、无LoRA）、dev＋A849＋V7（40步）、H3-PDD8和VACE-LightX2V4；远端结果为加速版，不能写成普通H3 50步或VACE40步。

本批保留UE200/CoinVE75/TripVVT25全部样本，不按效果筛选。输入manifest SHA256为386caf3ca11ff5ad59b59ceac16088e24864c913fb844015cc3a08e8f33614c0。输出引用本地推理目录和frozen300_gt_metrics_20261115/remote固定revision下载结果，逐项SHA核验，不改写原视频/回执。GT不进入本轮VBench。

## 评分媒体与CPU准备

全部使用各行冻结12fps时间轴，UE121帧、真实49–97帧，保留合法端点。LTX/A849/VACE完整原始输出直接评分；H3原生24fps输出按实际生成帧0,2,…,2(N−1)投影为N帧12fps，用libx264rgb/rgb24/crf0无损编码。每个派生视频完整解码，RGB逐帧SHA与原生选中帧一致，额外原生尾帧不计入。保留各模型原生画布，未采用SSIM测评的去黑边/统一缩放版本；尺寸与编码差异仍会影响成像分数。

SOURCE按同一冻结采样帧索引和LTX contain画布派生12fps RGB无损视频，仅用于官方RAFT静态筛选的共同交集，不能将它误作SOURCE相似度评分。1200条输出和300个SOURCE映射到评分媒体SHA后去重计算，报告仍按任务计权。media/scoring及source派生仅本目录新增，不修改旧结果。

prepare.sh在login上4路准备媒体（最初2路，小测后依据login负载和可用内存提高至4路），之后score.py raw在login调用官方temporal_flickering.cal_score；不为CPU工作申请GPU。准备身份和输出/评分SHA见metadata/media及scoring_media.jsonl/source_media.jsonl。

## 官方实现与GPU执行

VBench固定commit fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490；只读复用a849_external_vbench_v2_96/runtime的已验证环境和评分权重，不升级共享环境。官方源码与全部评分模型权重哈希存metadata/implementation.json。custom_input、local=True、imaging_quality_preprocessing_mode=longer；保存raw_score与score，Imaging Quality只除100一次，其他沿用返回值。报告六列统一score×100；不平均成官方Total Score。不复用未经完整协议兼容核对的历史分数。

五项GPU指标：subject_consistency、background_consistency、aesthetic_quality、imaging_quality、motion_smoothness。第六项temporal_flickering的全量原始值在CPU算。GPU质量评分每批64条，批量未返回/失败的媒体定位到单视频，失败明确记录，不以空、0或NaN充数。

GPU先运行六条×四模型的五项质量指标和30份任务媒体RAFT静态筛选，GPU小测120项质量分及筛选齐全且有效后扩展全部。为了提前暴露技术问题，缺少的小测媒体在login另以单路准备，全量准备继续；小测可从逐条回执读取媒体，使用VB_SMOKE_ONLY=1在独立step提前验证。全量入口再次检查同一门槛，按兼容媒体SHA复用已完成的小测分数，不重复算分。官方RAFT StaticFilter.infer不修改阈值，逐媒体保存实际params、源码及raft-things.pth哈希。筛选涉及SOURCE及所有模型输出，预期任务级1500次，实际按SHA去重。

现有testqueue分配202896于本任务启动时重新核验：剩余约8小时、4×H200空闲。supervisor在login把已准备完整的样本冻结为批次清单，CPU准备与GPU评分并行。每个批次的scoring/source清单及SHA保存在metadata/batches；四个GPU worker读取同一批固定清单，不在计算途中加入新视频。批次之间按媒体SHA及同一实现复用已完成分数，仅计算新增/缺失项；出现技术失败先停止并记录，不自动重试整批。所有300条准备好后仍会执行覆盖全量的最终一批，再等待login上的原始闪烁评分和全量汇总验收。

每个GPU批次启动前重新核实分区、RUNNING状态和剩余时间，优先用独占step（56CPU、384GB、4H200，最长7小时并预留5分钟），使用现有VBench协作锁、本任务锁和GPU UUID锁，启动时检查每卡显存。各批次日志按phase区分。只结束本任务step，不取消用户allocation。若超时或分配不可用，保留分数与错误，由当前任务检查原因后仅续算缺失项，不擅自切分区或重跑推理。run.sbatch为全量准备完成后的独立testqueue四卡12小时备用入口，不自动覆盖历史分数。

## 最初静态方案与最终报告范围

最初方案拟另计算共同集合S：SOURCE与四模型输出均由官方RAFT判断静态的相同任务ID；缺失/失败则pending，筛选全部完成且交集空才记N/A，非空缺分数则incomplete。用户随后取消该环节，因此本轮最终报告的静态状态是not_requested，不是空交集N/A，也不把未完成筛选当作非静态。

report/REPORT.md、summary.csv/json按总体/数据集/任务分组；per_video_metrics.csv保留task/model/dimension/media SHA、raw_score/score/status；static_filter_status.json明确用户取消静态筛选且不使用部分缓存；failures.json记录六项评分失败。预期7200个任务评分键必须逐项齐全且有限；336行总体和分组均值由验收脚本独立复核。

本轮交付自动指标和技术验收，不声称已对1200条完成直接人工编辑质量审核，不上传新包。VBench质量/内部一致性不代表编辑指令完成、参考身份准确或保持SOURCE。

## 状态

2026-11-15提前GPU小测已通过：step 202896.17，Slurm COMPLETED、ExitCode 0:0，耗时00:03:38。六任务×四模型的五项质量分数120/120，以及SOURCE和输出的官方RAFT筛选30/30均完成且无失败；详见metadata/gpu_smoke.json、early_smoke_completion.json。此为技术小测，不代表300条全量完成或人工编辑质量通过。

首批122任务、488个模型输出已于17:16:40（Asia/Singapore）在step 202896.18启动，CPU继续准备后续样本；实时状态以metadata和Slurm为准。

metadata/pairing_audit.json：300任务、1200输出的完整英文指令、SOURCE、参考顺序、时间范围及输出SHA逐项匹配通过；preparation.json：CPU媒体准备完成；raw_completion.json：CPU原始闪烁流程结束；gpu_smoke.json：GPU门槛通过；submission.json/gpu_process.json/run_status.json：GPU启动/运行；completion.json：全量汇总是否complete；final_validation.json：按用户最终范围验收7200个评分键、336行分组均值及300个H3无损派生证明，静态筛选标为not_requested。logs保存准备、CPU评分、各GPU及supervisor日志。submitted/running或存在report都不代表全量验收通过。

静态共同集合的缺失/空集/共同任务规则已由test_contract.py验证。稳定运行后遵循项目15分钟人工式检查节奏，底层日志继续自动落盘。
