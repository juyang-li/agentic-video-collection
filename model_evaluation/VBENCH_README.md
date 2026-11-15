# VBench 视频编辑测评经验：给接手 Agent 的 README

更新：2026-11-15。依据冻结300条四模型测评及历史 A849 v2/v3/v4 的实际脚本、回执和验收结果整理。本文区分当前项目口径、官方方法背景及历史案例；不是通用排行榜复现脚本。

**当前默认：Temporal Flickering（`temporal_flickering`）不做静态筛选。用户已明确要求对全部有效输出计算原始分数；不运行RAFT筛选、不取SOURCE与模型输出的静态交集，不把静态筛选列为完成或验收条件。历史静态结果保留，不作为新任务默认。**

## 1. 先明确评测口径

- VBench 这六项衡量输出视频的质量和内部一致性，不能代替“是否完成编辑”“是否编辑正确人物”“是否保持源视频内容”的审核。
- 本轮使用 **VBench 的 custom_input 模式**，不是在官方完整 prompt suite 上跑出的排行榜总分。不要把六项直接平均后称为官方 Total Score。
- 同一任务必须比较同一源时间段、指令和参考。输入、mask、采样器或权重发生变化，应记录到新的推理配置和回执，不能继续声称所有视频使用同一协议。
- 低质量但有效的生成结果保留；无法解码、生成失败、评分失败分别记状态，缺失值留空/null，不填零，不静默排除。
- 新测试集另建目录，保留旧清单、输出和报告。调整样本后不能依据两个版本的总体均值差宣称模型进步。

开始新工作前阅读项目 [AGENTS.md](../AGENTS.md)、[任务描述](../任务描述.md)；涉及人工审核再读 [审核规范](../DATASET_AUDIT_GUIDE.md)。输出与GT的SSIM/PSNR/LPIPS另见 [三指标 README](SSIM_PSNR_LPIPS_README.md)。已完成的300条和历史v4均不需要重跑。本页相对链接指向项目工作区；单独分发README不包含所链接的脚本、数据或权重。

## 2. 六项指标与数值尺度

| 正式名称，均为 ↑ | 脚本 dimension | 解读时注意 |
|---|---|---|
| Subject Consistency | `subject_consistency` | 输出内主体一致，不等于与参考身份匹配，也不等于保持源人物 |
| Background Consistency | `background_consistency` | 输出内背景一致，不等于成功换成目标背景 |
| Aesthetic Quality | `aesthetic_quality` | 美学评分，不能判定编辑指令是否完成 |
| Imaging Quality | `imaging_quality` | 对分辨率、编码及预处理口径敏感 |
| Temporal Flickering | `temporal_flickering` | 相邻帧像素变化会混入正常运动，详见下一节 |
| Motion Smoothness | `motion_smoothness` | 平滑不等于动作正确，也不等于运动充分 |

本轮记录原始值 `raw_score` 和统一尺度 `score`：Imaging Quality 原始值除100，其余五项沿用官方返回值。保存精度完整，展示时可以将六项统一乘100；不能只把部分列乘100，也不能对 Imaging Quality 重复除100。

Dynamic Degree 不列入新版六项主表。历史结果保留；它可以辅助解释运动量，但高低不是简单的优劣关系。

## 3. Temporal Flickering当前口径：全量、不做静态筛选

### 全量原始分数

在login CPU对全部有效输出调用官方 `vbench.temporal_flickering.cal_score`，每模型分别按视频等权汇总。当前冻结300条测评为每模型300条，不先按运动量或静态标签删样本。

当前固定代码计算的是：`1 - mean(相邻帧逐像素绝对差) / 255`。正常走路、转头、相机移动也会使相邻帧不同，所以分数降低不一定表示闪烁更严重；冻结画面反而可能得高分。全量结果必须标为“全量原始分数，受正常运动影响”，不能直接当成纯闪烁排名。

当前流程不调用 `static_filter.StaticFilter.infer`，不要求SOURCE参与筛选，不计算静态共同子集。取消/不采用静态筛选记录为 `not_requested`；不能伪装成筛选完成但空交集的N/A，也不能把未完成筛选当作非静态。已生成的部分静态缓存可保留，但不进入当前报告，不阻碍六项评分完成。依据见300条实验的 [用户指示](frozen300_vbench_20261115/metadata/user_override.json) 和 [最终静态状态](frozen300_vbench_20261115/report/static_filter_status.json)。

### 官方方法背景与历史静态案例（不作为当前执行要求）

[官方 README](https://github.com/Vchitect/VBench#-evaluation-method-suite)要求在评估 Temporal Flickering 前筛出静态视频；本地固定版本的计算函数也明确以静态视频为适用条件。这里的“静态”是通过官方工具的运动阈值判断，不要求每个像素完全相同。

上述官方适用条件与本项目当前“全量原始分数”口径不同，不能宣称后者已经复现官方静态闪烁基准。历史v4曾在官方工具基础上额外采用配对规则：

```text
S = {任务ID：source 和所有参评模型输出均被官方工具判为静态}
每个模型的静态闪烁均值 = 该模型在同一集合 S 上的 cal_score 均值
```

该交集是历史项目比较设计，不是所有论文的统一要求。历史v4共同子集仅2条：`object_addition_02`、`object_addition_03`；v3为4条，另外含两个已移出的object removal任务。n=2的均值只是补充诊断，不能代表全量116条，也不能套用到当前300条。不复制生成帧、不插帧、不冻结输出以制造静态样本。

### 论文怎么报？MoCha 有无明确说明？

不要从论文表头的“Temporal Flickering”推断作者必然计算了全量或静态共同子集。以下保留2026-11-15的文献核查记录，不改变当前项目不做静态筛选的默认要求。

核查 [MoCha arXiv v2，第5.1/5.2节及表2](https://arxiv.org/html/2601.08587v2#S5)（2026-11-15）：它采用这六项 VBench 指标，但这些说明没有披露静态筛选、筛后数量或跨模型共同任务规则。因此目前只能记录“该细节未明确披露”，不能声称 MoCha 使用了全量，也不能把我们的共同交集规则归给 MoCha。相同指标名称不意味着数据集、时间范围和预处理可直接对比。

## 4. 先确定真正送去评分的视频

当前300条按各样本冻结时间轴评价完整区间，UE为121帧、真实视频为49–97帧，统一12fps；每模型31212帧。保留合法端点，MP4时长N/12不必等于语义时间区间长度，不能为了凑整数秒删除最后一帧。

| 模型/文件用途 | 当前300条规格 |
|---|---|
| LTX2.5 distilled＋V7、dev＋A849＋V7、VACE-LightX2V4 | 原始输出直接评分，N帧、12fps；1280×736或736×1280 |
| H3-PDD8原生输出 | 原生24fps；1344×768或768×1344，完整原件保留，额外尾帧不计入评分 |
| H3评分视频 | 抽取真实生成帧0,2,…,2(N−1)，N帧、12fps；RGB无损编码 |
| 浏览副本 | 如另做H.264/yuv420p副本，仅供播放，不替换评分媒体 |

历史v4仅评价0–4秒、49帧；其H3原生107帧的输入映射为 `source_index=min(48,(j+1)//2)`。这是历史输入适配，不是当前300条统一裁49帧的依据。评分输出始终抽真实生成帧，不做时间压缩或人工补帧。

H3评分文件用 `libx264rgb / rgb24 / crf=0`，写出后重新解码为RGB24，逐帧像素哈希必须与原生选中帧一致。普通H.264浏览副本存在有损压缩，不能拿来替代评分文件或覆盖它的分数缓存。

VBench保留各模型原生画布，尺寸/编码仍有差异，不能把所有成像分差都归因于模型。不要拿SSIM/PSNR/LPIPS去padding、统一有效720p后的媒体替代本轮VBench输入。若新实验改为统一分辨率或统一编码，先冻结新协议并重算受影响分数，不能混用旧分数。

## 5. 固定实现、环境和预处理

本轮 VBench commit：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。

本机复用环境：`a849_external_vbench_v2_96/runtime/vbench/bin/python`。实测版本为 Python3.12.16、torch2.10.0+cu126、torchvision0.25.0+cu126、NumPy2.5.4、PyAV19.0.2、opencv-python5.0.0.93、timm1.0.12、pyiqa0.1.14.1。此列表是环境记录，不是跨机器已验证的安装锁文件；迁移时不要直接复制包含绝对路径的venv。

质量维度使用官方 `VBench.evaluate(..., local=True, mode='custom_input')`，本轮 `imaging_quality_preprocessing_mode='longer'`。复现时固定该设置，不要因默认值变化改变预处理。

缓存变量：

```bash
# 在 video_collection/model_evaluation 下查看现有布局；不要覆盖已完成产物。
BENCH_BASE="$PWD/a849_external_vbench_v2_96"
VB_PY="$BENCH_BASE/runtime/vbench/bin/python"
export HF_HOME="$BENCH_BASE/runtime/hf"
export TORCH_HOME="$BENCH_BASE/runtime/torch"
export VBENCH_CACHE_DIR="$BENCH_BASE/runtime/vbench_models"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

当前逐分数保存官方commit、包装脚本SHA、媒体SHA及实现身份；raw另存官方函数源码SHA。[implementation.json](frozen300_vbench_20261115/metadata/implementation.json)记录官方源码和评分权重SHA，[environment.json](frozen300_vbench_20261115/metadata/environment.json)及[pip_freeze.txt](frozen300_vbench_20261115/metadata/pip_freeze.txt)记录实际环境。历史static记录保留RAFT身份，不是当前必需项。不要误认为一个包装脚本SHA就证明全部模型权重相同。

## 6. 缓存、重试与汇总

**复用生成结果**：检查源内容、采样时间点、提示词、参考图及顺序、mask，以及模型/LoRA/采样器/步数/CFG/seed等实际推理配置。改任一相关条件，就不能把旧生成结果冒充新生成；重编号要保留旧ID和来源证明。

**复用分数**：至少要求实际评分媒体SHA完全一致，且官方实现、评分模型权重、预处理和归一化口径兼容。文件名相同、任务ID相同或浏览效果相似都不够。更换评分实现时重新计算，不能盲目复制旧缓存。本轮 `reuse_scores.py` 面向已知同协议的父目录，不是任意实验之间的通用兼容检查器。

- 一份媒体可只计算一次指标，再映射到多个任务；汇总仍按冻结任务表计数，不能意外按去重后的视频数改变任务权重。
- 建议保留 `(task_id, model, dimension, output_sha256, raw_score, score, status)`。结果数量与键集合都要检查，不能只看结果JSON存在。
- 批量评测失败后按单视频定位；保留错误和成功项，只补算失败/缺失项。不要用空值、NaN或0掩盖错误。
- 官方流程可能跳过读取失败视频，聚合文件生成不代表全量成功。校验每个预期任务/模型/维度实际返回有限数值。
- 临时评分目录只放当前待算媒体，并用SHA命名映射回回执；清除本任务遗留的临时软链接，避免旧视频混入，不删除源媒体。
- 各模型有效数量不同，先报告各自n和缺失原因；公平比较使用同任务ID的配对交集。按任务分组汇总，也保留逐视频分数和视觉审核。
- 视觉审核查看完整source、参考和输出时间序列，绑定输出SHA和审核图SHA；未经查看不能自动标reviewed。历史v4曾取0、0.5、…、4秒共九个时刻；当前300条交付自动指标和技术验收，未声称1200个输出已完成人工效果审核。

## 7. CPU/GPU 分工及安全续跑

本项目规定纯CPU任务在login执行；GPU任务只用 `testqueue`。不要为纯CPU任务占GPU，也不要伪造 `SLURM_JOB_PARTITION` 绕过入口检查。

| 任务 | 当前运行位置 |
|---|---|
| 清单/哈希、媒体对齐、H3帧派生、审核截图、汇总、验收、打包上传 | login，CPU |
| 官方 `cal_score` 全量闪烁 | login，CPU |
| 其余五项质量指标 | testqueue，GPU |
| RAFT静态筛选、静态共同子集 | 不执行，不是当前验收项 |
| 模型推理、人物解析/SAM2 mask生成 | testqueue，GPU |

当前四卡质量评分按维度分片；单卡绑定用实际分配的 `CUDA_VISIBLE_DEVICES` 数组，不照抄物理卡号。共享GPU时先协调推理与评分的锁，避免两套常驻模型争显存。

续跑既有allocation前，确认实时授权、剩余时间及现有step。用户要求保留allocation时，只结束经核实属于本任务的具体step，不能取消整个job。本文和旧脚本里的202896是本轮历史任务号，不是未来可直接使用的资源。

## 8. 脚本入口与最小执行顺序

从此文件所在目录可打开：

| 用途 | 已跑通的参考入口 |
|---|---|
| 五项质量指标、CPU原始闪烁 | [300条score.py](frozen300_vbench_20261115/scripts/score.py) |
| 六项汇总、分组、不采用静态的报告分支 | [300条aggregate.py](frozen300_vbench_20261115/scripts/aggregate.py) |
| 全量评分键、均值、媒体身份验收 | [300条validate.py](frozen300_vbench_20261115/scripts/validate.py) |
| 环境变量 | [env.sh](frozen300_vbench_20261115/scripts/env.sh) |
| 同媒体分数复用 | [reuse_scores.py](a849_external_vbench_v4_116/scripts/reuse_scores.py) |
| H3评分/浏览帧派生，位于扩充目录 | [h3_derive.py](a849_v4_extensions16_20261115/scripts/h3_derive.py) |
| 审核图及SHA绑定审核记录 | [visual.py](a849_external_vbench_v4_116/scripts/visual.py) |
| 生成协议与复用结果核对 | [assemble.py](a849_external_vbench_v4_116/scripts/assemble.py) |
| 历史v4验收（含旧静态要求） | [validate.py](a849_external_vbench_v4_116/scripts/validate.py) |
| 离线审核页及ZIP | [review.py](a849_external_vbench_v4_116/scripts/review.py) |

这些是**具体实验脚本**，不是通用CLI。新任务先调整 `common.ROOT`、父目录、模型集合、cohort/数量、锁、路径和验收条件，再执行。历史 `contracts.py` 的旧三模型常量不可直接用于四模型。当前案例的静态关闭由 `metadata/user_override.json` 中的 `request="不用静态筛选"`、`static_filter_required=false` 显式记录；使用该实现的新任务必须落实同一决策并确认 `static_required()` 返回false，不能以为改了README就改了脚本默认值。

**不要直接运行旧混合wrapper。** 历史v4的 `score_gpu.sh` 会启动静态筛选；300条的 `gpu.sh` / `check_smoke.py` 还保留曾经检查30份静态记录的小测逻辑。移植无静态流程时应移除这类旧门槛，保留六项指标的完整性检查。本次文档更新没有修改这些脚本；以下示例只调用raw和quality，不调用static或旧静态小测。

新目录、评分媒体及关闭静态的配置准备好后，最小调用顺序如下；`TASK_ROOT` 必须指向已适配的新任务，不能直接指向已完成目录：

```bash
# 1. CPU：核验冻结输入及实际输出，准备评分媒体，复用兼容分数。
"$VB_PY" "$TASK_ROOT/scripts/score.py" raw

# 2. 在真实获分配的testqueue GPU step内部执行。
# 每worker绑定一张实际分配的GPU；四个worker分别传SHARD=0、1、2、3。
# 此片段不是Slurm提交脚本，不可在login伪造SLURM变量运行。
"$VB_PY" "$TASK_ROOT/scripts/score.py" quality --shard "${SHARD:?}"

# 3. 所有分片完成后回到login汇总、技术验收。
# 是否另做人工审核/审核包，以该任务授权为准。
"$VB_PY" "$TASK_ROOT/scripts/aggregate.py"
"$VB_PY" "$TASK_ROOT/scripts/validate.py"
```

不要重启已完成实验的旧supervisor/finalizer来“查看状态”；它们可能重新提交任务、重建清单或打包。查看当前300条只需读metadata，或运行只读 [status.py](frozen300_vbench_20261115/scripts/status.py)。

## 9. 经验与验收标准

历史及当前实验的经验：

| 现象 | 处理经验 |
|---|---|
| 背景mask吞掉手指、手臂和鞋子 | 编辑区膨胀可能侵入前景；核对解析标签是否包含鞋子，并看完整时间序列 |
| 发色mask误含胡须、墨镜，个别帧突然塌缩 | mask只能依源视频修正；检查逐帧面积，结合可靠关键帧跟踪；不能只看首帧就冻结 |
| 代码迁移后同名 `media` 模块导入错位 | 核对实际导入文件与函数签名；明确传fps，避免依赖另一个版本的默认参数 |
| 只把VACE40步改成4步 | 不成立：需要完整加速权重加载及配套采样器、sigma和CFG；按实际回执校验，不能统一断言全部40步 |
| H3浏览视频可播但画质分受转码影响 | 浏览与评分文件分开；评分派生帧做RGB像素一致性核对 |
| 文档说不做静态，旧wrapper仍启动RAFT或等待静态小测 | 检查实际调用链和验收条件；当前默认只做六项评分，静态记录not_requested，不以缺失静态缓存判失败 |
| 看到高一致性就认为编辑成功 | 未编辑的源场景也可能得高一致性，必须查看指令完成度 |

自动指标验收要求：冻结输入可追溯、评分媒体完整解码、预期六项评分键齐全且有限、归一化正确、逐视频与总体/分组均值可独立复核、报告n和时间范围清楚。**静态筛选不是验收项，缺失静态结果不构成失败。** 当前300任务×4模型×6项=7200份分数、336行总体/分组统计均已验收通过，静态状态为not_requested。

若另行交付人工审核包，再核对真实审核记录绑定实际输出、离线链接和ZIP本地CRC；未经人工查看不得标reviewed。只验证少量smoke不能宣称全量通过。

历史v4事实校验点：116任务×4模型=464结果、2784分数；464条直接审核；580次评分/浏览视频完整解码；116次H3原生→评分帧像素检查；静态共同子集2条。新增8条TripVVT为看过输出后的筛选，所有32条候选及排除理由保留。这些历史数量、筛选规则不能套用到当前300条或新实验。

当前事实入口：[300条报告](frozen300_vbench_20261115/report/REPORT.md)、[逐视频分数](frozen300_vbench_20261115/report/per_video_metrics.csv)、[分组统计](frozen300_vbench_20261115/report/summary.csv)、[验收回执](frozen300_vbench_20261115/metadata/final_validation.json)。历史复现另见[v4报告](a849_external_vbench_v4_116/metadata/report.md)、[实际推理配置](a849_external_vbench_v4_116/metadata/inference_protocols.json)、[历史验收](a849_external_vbench_v4_116/metadata/final_validation.json)。机器数据优先于聊天中的旧进度估计。

审核包默认连续滚动、字母顺序及连续编号，完整指令和输入/输出同屏；TripVVT保留官方caption与GT审阅入口，但GT不作为推理条件。VACE辅助输入/overlay视频仅放全包前两条。授权上传时只传ZIP，以上传器成功回执为准，不额外远端验哈希/大小或下载回读。
