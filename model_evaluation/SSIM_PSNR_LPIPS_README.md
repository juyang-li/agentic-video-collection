# SSIM、PSNR、LPIPS：视频编辑对 GT 测评方式与经验

更新：2026-11-15。依据已完成的冻结300条、四模型实验整理，供后续测评和接手 Agent 使用。本文区分已实测协议、历史运行条件和迁移建议；不是任意视频都能直接调用的通用 CLI。

**当前比较对象仅为模型输出与对应 GT，不与 SOURCE 比。** 三项指标表示接近单一 GT 的程度，不直接判断编辑指令是否完成、身份是否匹配参考，或时序是否稳定。动作、镜头等编辑可能有合理多解；全图指标也可能被大面积未变化区域主导。输出自身质量与一致性另见 [VBench README](VBENCH_README.md)。

开始前阅读 [AGENTS.md](../AGENTS.md) 和 [任务描述](../任务描述.md)。涉及人工审核再读 [审核规范](../DATASET_AUDIT_GUIDE.md)。本页相对链接指向项目工作区；单独分发 README 不包含所链接的实验数据、权重或脚本。

## 1. 冻结样本与实际比较范围

- 用稳定 ID/UID 关联输入、GT、输出及运行回执，核对完整指令、SOURCE、参考顺序、时间范围、模型权重、采样参数和媒体 SHA256。不能仅按文件名配对，也不能将不同加速版本统称为同一基线。
- 固定输入清单、GT清单和协议身份；GT只用于评分，不进入模型推理条件。新测评另建目录，不改写原始视频和历史 JSON。
- 评价完整冻结时间轴的全部12fps采样帧。本例 UE 为121帧，真实视频为49–97帧；不得为省时只取几帧，也不能把所有样本统一截成历史4秒版本。
- GT按冻结 `sample_frame_indices` 取帧；输出按实际 PTS 选择最近时刻，距离相等取后帧。本实现以1e-9秒处理数值平局；普通12fps输出时间误差限为1e-5秒，H3为1/48秒加1e-6容差。
- H3原生24fps输出通常对应真实生成帧 `2*i`；尾部原生对齐帧不参与评分。验证完整原生流的画幅、帧数、PTS及末端覆盖，不插帧、不复制输出帧、不做光流、时间偏移搜索或最优配准。

## 2. 统一有效画面，保留颜色与缩放口径

PyAV解码到RGB，按原始宽高和协议 contain 几何推导固定 padding。**不能检测黑色像素后再裁剪**，否则会误删真实黑色内容。1920×1080放入1280×736的例子，固定去掉上下各8像素；竖屏相应去掉左右各8像素。

本例最终有效画面统一为1280×720或720×1280。GT原件直接缩放一次，输出先裁协议 padding，再用 Pillow LANCZOS缩放到同尺寸；H3有效画面也按相同目标尺寸处理。保持原生解码颜色语义，不做自动颜色匹配。

该预处理属于本项目冻结协议，不代表所有论文的唯一做法。更换尺寸、插值、裁剪或颜色口径后需新建协议并重算，不能混用缓存。它与 VBench 保留模型原生画布的协议不同，**不能把三指标归一化后的媒体直接替代 VBench 评分媒体**。

## 3. 指标参数与汇总

| 指标 | 本例固定计算方式 |
|---|---|
| SSIM ↑ | scikit-image 0.26.0；RGB浮点 `[0,1]`；`data_range=1`、`channel_axis=-1`、`gaussian_weights=True`、`sigma=1.5`、`win_size=11`、`use_sample_covariance=False`；RGB通道平均 |
| PSNR ↑ | RGB归一化到 `[0,1]`，用float64计算逐帧MSE，再计算 `-10*log10(MSE)`，单位dB；不先跨帧平均MSE再求PSNR |
| LPIPS ↓ | 官方 lpips 0.1.4，`net=alex`、`version=0.1`、FP32、`eval()`、`inference_mode()`；输入为RGB `[-1,1]`，不再额外归一化 |

LPIPS关闭AMP和TF32，保留完整有效720p画面，不缩到256，也不分块拼接。初始batch为4，OOM只降为2或1；不能为了继续运行偷偷降低分辨率或改网络。

每项均为 **逐帧分数 → 视频内算术平均 → 视频间等权平均**。分别给出总体、数据集和任务组结果及有效n，不能把所有帧混在一起平均，让长视频获得更大权重。缺失或失败不能填0或静默剔除；模型有效样本不齐时另列相同任务ID的配对交集。

MSE为0时PSNR为正无穷。本实现保存 `psnr_db=null`、`psnr_infinite=true`，不封顶为某个大数，也不当作缺失。任一帧无穷时视频均值延续该标记，任一视频无穷时对应分组也延续；报告展示为∞。本次正式1200对没有该情况。非法NaN与这种有明确含义的无穷标记应分开处理。

## 4. CPU/GPU执行与断点复用

默认纯CPU工作在login执行：配对/哈希、解码、对齐、SSIM和PSNR、汇总及验收。SSIM和PSNR在同一遍CPU计算中一起得到，没有单独记录两者各自耗时。LPIPS使用 `testqueue` GPU；不为CPU评分新申请GPU，也不伪造分区变量。

历史实验经用户明确授权，曾使用已有allocation的空闲CPU与login分片并行；那是既有资源的授权使用，不是以后默认提交CPU作业的依据。复用allocation前重新核实授权、资源、剩余时间和step；历史job号不可直接照抄。

先做小测并查看首/中/末GT与输出并排预览，确认横竖屏、padding、PTS及H3尾帧处理，再扩大到全量。预览用于几何审核，不能自动标记编辑效果通过。本例小测为六任务×四模型共24对，覆盖UE、CoinVE、TripVVT。

复用分数至少要求GT、输出、采样对齐和数值协议身份一致；核对原件及SHA，不只核对ID。CPU阶段的分数可由LPIPS阶段读取复用，避免再次计算SSIM/PSNR。当前CPU脚本跳过缓存时只显式检查协议和输出SHA，LPIPS入口还会检查GT SHA；迁移时应在复用前补齐GT身份审核，不能把较少的检查当作完整缓存合同。

不落盘全部规范化无损帧，避免耗尽共享存储；逐视频读取并复用同任务GT即可。记录实际环境、指标与AlexNet权重SHA、batch、运行位置、逐条状态及耗时；不要升级共享推理环境来修复评分依赖。

## 5. 已遇到的问题与验收方法

| 经验 | 处理方式 |
|---|---|
| NumPy标量导致JSON序列化失败 | 写盘前显式转为Python `float`、`bool`；本例曾因此中止CPU阶段。保留异常回执，修复后校验缓存再续跑 |
| 改线程或迁移节点后担心分数混用 | 保存执行迁移记录，确认数值函数和协议不变；线程数、节点及scope记录为执行条件，不冒称一次干净基准 |
| 同画面但帧有偏移，或把原生尾帧算进去 | 检查每帧目标时刻、所选索引/PTS及末端覆盖；不能根据分数高低搜索最佳对齐 |
| LPIPS显存不足 | 仅降低batch，不改720p、FP32、网络或分块方式；记录最终batch |
| 汇总文件生成但不齐全 | 检查预期 `(ID, model, frame)` 键和每视频帧数，核对GT/输出/协议SHA；不能只看文件数量 |

CPU核心检查包括：固定padding、同图SSIM=1、同图PSNR无穷、独立PSNR公式、构造24fps视频抽偶数帧、排除尾帧、错误画幅拒绝。GPU检查包括：同图LPIPS≈0、亮度扰动可区分、对称性、batch一致性。随后独立复核逐帧到逐视频、逐视频到总体/分组的均值。

技术校验不等于人工编辑效果审核。缺失、解码失败、评分失败需记录原因；仅有smoke通过不能宣称全量通过。不同指令、模型版本或新增/删去样本后的均值不应直接当作模型改进证据。

## 6. 本轮实测速度

一“对视频”指一个模型输出与其GT，300任务×四模型=1200对，不是1200个任务。条件为有效720p横竖屏、12fps、49–121帧，全帧评分；以下包含实际解码、对齐、读盘等开销。

| 阶段 | 历史运行条件 | 实测吞吐 |
|---|---|---:|
| SSIM＋PSNR | login，4个CPU工作线程；两节点分片后的本地新增527对 | 约250对/小时 |
| SSIM＋PSNR | 已授权现有分配的纯CPU step；16 CPU、128GB、12个工作线程；远端600对 | 约400对/小时 |
| SSIM＋PSNR整体 | CPU阶段先后迁移并两处分片；1200对从首条落盘到全部完成约2小时22分钟 | 约500对/小时 |
| LPIPS | 单H200，16 CPU、128GB；AlexNet、FP32、batch4；作业204990耗时1小时14分08秒 | 约970对/小时 |

从首条CPU结果（11:59:39）到LPIPS作业结束（15:36:07），这轮三项完整流程约3小时36分钟，折合330对/小时。CPU两处分片的模型和视频不同，阶段有先后和尾部等待；不能简单相加为持续650对/小时，也不能据此认定12线程相对4线程的纯加速比。单条回执的 `seconds` 不包含所有GT读取开销，整体吞吐应使用实际阶段/作业墙钟时间。

以上是2026-11-15这一批的实测，不是任意时长、分辨率或机器的通用保证。SSIM和PSNR没有独立计时，不编造各自速度；也没有四卡LPIPS独立实测。

## 7. 事实入口与迁移顺序

现有实现位于 [frozen300_gt_metrics_20261115](frozen300_gt_metrics_20261115/README.md)。参考入口如下，均有本实验路径和模型常量，**不是新数据可直接运行的通用工具**：

| 用途 | 参考 |
|---|---|
| 冻结数值协议 | [protocol.json](frozen300_gt_metrics_20261115/metadata/protocol.json) |
| 对齐、归一化、指标核函数及LPIPS阶段 | [evaluate.py](frozen300_gt_metrics_20261115/scripts/evaluate.py) |
| CPU阶段 | [cpu_stage.py](frozen300_gt_metrics_20261115/scripts/cpu_stage.py) |
| CPU小测、最终独立复核 | [test_cpu.py](frozen300_gt_metrics_20261115/scripts/test_cpu.py)、[final_audit.py](frozen300_gt_metrics_20261115/scripts/final_audit.py) |
| 汇总、GPU提交配置 | [report.py](frozen300_gt_metrics_20261115/scripts/report.py)、[run.sbatch](frozen300_gt_metrics_20261115/scripts/run.sbatch) |
| 实际依赖及权重身份 | [environment.json](frozen300_gt_metrics_20261115/metadata/environment.json) |
| CPU历史分片 | [two_node_cpu.json](frozen300_gt_metrics_20261115/metadata/two_node_cpu.json) |

迁移顺序：新建实验目录并冻结GT/输出和协议 → 适配路径、ID、模型集合及数量 → CPU小测与几何预览 → login计算SSIM/PSNR → testqueue最小LPIPS验证及全量 → CPU汇总和独立验收。检查旧controller/supervisor的副作用，不为查看状态重启它们。

本例共1200对、124848帧对，每模型31212帧；四模型为LTX2.5 distilled＋V7、dev＋A849＋V7、H3-PDD8和VACE-LightX2V4。完整结果见 [报告](frozen300_gt_metrics_20261115/report/REPORT.md)、[分组统计](frozen300_gt_metrics_20261115/report/summary.csv)、[逐视频](frozen300_gt_metrics_20261115/report/videos.csv)、[逐帧](frozen300_gt_metrics_20261115/report/frames.jsonl)；验收见 [completion_audit.json](frozen300_gt_metrics_20261115/metadata/completion_audit.json)。依赖和机器数据优先于聊天中的估计，不重跑已完成结果、不自动上传实验数据。
