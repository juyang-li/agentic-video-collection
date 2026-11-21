# VBench视频质量测评：外部使用指南与最小示例

项目正式运行与实测经验见[测评经验汇总](METRIC_EVALUATION_LESSONS.md)。通用示例的CPU测试与正式实验验收分开记录，不能据此宣称通用示例已完成独立GPU验收。

面向在自己Linux/NVIDIA机器上评测任意数量生成视频的读者。随包提供[评分示例](metric_examples/vbench_metrics.py)、[共享视频与计时模块](metric_examples/common.py)和[清单示例](metric_examples/manifest.example.jsonl)；不依赖原作者的机器、数据、账号或调度器。它只做评分，不生成视频，不是完整测评平台。

**Temporal Flickering全量计算，不做静态筛选。** 示例没有RAFT筛选、SOURCE与模型静态交集或静态验收门槛。闪烁必须标为“全量原始分数，受正常运动影响”。GT保真度另见[SSIM／PSNR／LPIPS指南](SSIM_PSNR_LPIPS_README.md)。

## 1. 六项指标与适用范围

| 名称，均为↑ | dimension | 含义 |
|---|---|---|
| 主体一致性 | subject_consistency | 输出内主体外观一致，不等于匹配参考身份 |
| 背景一致性 | background_consistency | 输出内背景一致，不等于成功编辑背景 |
| 美学质量 | aesthetic_quality | 构图、颜色等美感，不判断指令完成 |
| 成像质量 | imaging_quality | 成像质量，对尺寸、编码和预处理敏感 |
| 原始闪烁分数 | temporal_flickering | 邻帧像素变化，混入正常运动 |
| 运动平滑度 | motion_smoothness | 运动流畅，不等于动作正确或运动充分 |

使用官方`custom_input`模式，不是官方完整prompt suite排行榜，不将六项简单平均称为官方Total Score。只需要输出，不用GT或SOURCE作指标条件。模型间应比较相同id/时间段；输入或生成配置变化需要记录，不能以高一致性代替编辑成功审核。

保存`raw_score`与统一尺度`mean`：Imaging Quality的官方raw值除100一次，其他五项沿用；展示可把六项全部乘100，不能只变部分列或重复归一化。Dynamic Degree不在这里的六项中。

官方原始闪烁函数约为`1 - mean(相邻帧绝对差)/255`。走路、转头、相机移动会降低分数，冻结画面可能得高分。因此不把这一列当作纯闪烁排名。官方固定函数的注释以静态视频为适用条件；本文选择全量原始分数，应明确这一口径，不声称复现官方静态闪烁基准。用户未要求静态筛选时不启动该环节，状态为`not_requested`，不是空交集N/A。

## 2. 安装：独立环境与官方源码

先按[PyTorch官方选择器](https://pytorch.org/get-started/locally/)安装匹配驱动的torch/torchvision。参考环境为Python3.12、torch2.10、CUDA12.6，并非对所有GPU的保证。然后：

```bash
python -m pip install -r metric_examples/requirements-vbench.txt
git clone https://github.com/Vchitect/VBench.git third_party/VBench
git -C third_party/VBench checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
python metric_examples/vbench_metrics.py --help
```

示例直接从`--vbench-root`指定的官方源码导入，并检查commit和实际导入位置；不包含或修改上游源码。这个固定版本的setup.py对CUDA版本有旧限制，所以不要在现代CUDA环境直接照抄`pip install -e VBench`。此处采用源码导入和所用六项的依赖清单，不承诺上游全部其他维度也可用。上游背景见[官方仓库](https://github.com/Vchitect/VBench)。

首次默认允许官方实现下载公开权重，可能耗时较长，网络失败会明确失败。可用标准`HF_HOME`、`TORCH_HOME`、`VBENCH_CACHE_DIR`指定自己的缓存路径；不复制他人的虚拟环境。离线/已备权重场景使用`--local-weights`，必须按该固定版本官方`init_submodules(local=True)`的目录布局准备权重。示例不附模型、不读取任何私有凭据，也不自动修补依赖冲突。

## 3. 任意数据集的输入与评分媒体

JSONL每行是一个模型输出，`id`与`prediction`必填，`model`可选，默认default；GT字段可有但不读取。相对路径相对清单目录，`(id, model)`不能重复：

```json
{"id":"clip_001","model":"editor_a","prediction":"outputs/clip_001.mp4"}
```

主教程预设与10秒UE时间范围对齐：相对首帧0–10秒、12fps、含末端121个采样点。最近PTS匹配，平局取后帧；24fps输出通常为0,2,…,240。覆盖不足或重复选择生成帧报错，不补帧、不根据效果选时段、不冻结生成帧。完整尾帧会解码检查，超出评分区间的原生尾帧不计入。

`--start`、`--end`、`--fps`、`--endpoint exclusive`可改协议；标准左闭右开10秒/12fps对应120帧。原件本身必须已对齐语义起点，脚本不会搜索偏移。恒定名义fps及单一画幅是最小示例的前提；异常/VFR视频需先确认取帧策略。

**保留输出原生画布，不执行GT指标的去padding/统一720p。** 例如1280×736和1344×768会保留不同画布，成像分差不能全归因于模型。为使时间投影可审计，示例统一写为libx264rgb/rgb24/crf0无损评分文件，重新解码验证所选RGB像素一致性。仅改变时间采样与封装，不裁剪、不缩放；不能用有损浏览副本替代评分文件。

这些评分文件写入本次输出目录，适合小规模示例。大数据集会占用空间，使用者需预留存储或另做有记录的逐批处理；本示例不擅自删除缓存或已有结果。更改画布、fps、编码或预处理后，不能混用旧分数。

## 4. 单卡运行与完整性

先运行自造样例验证安装，128×72只是功能测试，不是720p速度基准：

```bash
python metric_examples/make_demo.py --output-dir demo_data
python metric_examples/vbench_metrics.py --manifest demo_data/manifest.jsonl \
  --output-dir demo_vbench --vbench-root third_party/VBench --device cuda:0
```

自己的数据小测与全量：

```bash
python metric_examples/vbench_metrics.py --manifest my_outputs.jsonl \
  --output-dir pilot_vbench --vbench-root third_party/VBench \
  --device cuda:0 --torch-threads 2 --limit 8
# 确认预处理、数值和用时后，另用新目录，去掉--limit评完整清单。
```

默认六项都算，单卡顺序执行；五项质量指标使用GPU，全量原始闪烁在CPU计算，不做静态筛选。仅运行CPU闪烁可传`--dimensions temporal_flickering --device cpu`，仍需要上游可导入的依赖。示例不读取Slurm变量、不申请资源；集群读者自行按本地规则获得GPU，再运行普通命令。

`--dimensions`可选逗号分隔的指标子集，报告会明确实际指标；不能拿子集称六项全量。官方聚合可能略过不可读视频，示例会核对每个预期视频是否返回有限值，记录遗漏/失败，不填0。部分维度成功时保留它们，整条不齐则failed，进程返回非零。

结果为`results.jsonl`、`summary.csv/json`、`implementation.json`及`official_results/`；每模型每项按视频等权平均，报告n、预期n及参与id。比较模型时核对相同id，而不只看数量。没有自动人工通过、默认重试或断点平台；输出目录须新建或为空。

## 5. 小测速度估计与实际用时报告

每次还生成`timing.json`与`timing_report.md`，记录硬件、版本、线程、设备、协议、实际成功/失败数、完整输出视频/小时和帧/秒。**一个视频指一个模型输出，不是一个标量分数或多个模型的一道任务。** 五项GPU指标的联合吞吐也不能用五个标量数之和冒充视频数。

实际总墙钟保留首条及冷启动，包括启动、输入探测、评分媒体准备与像素验证、官方API及指标文件写盘。计时报告自身写盘和进程退出在终点之外。GPU维度前后同步CUDA。每个维度记录`dimension_api_wall`，其中含模型加载、可能下载、解码及计算；官方API没有独立暴露的子阶段不虚构成纯GPU前向时间。

将清单前约8行安排为代表样本，用`--limit 8`小测。完整清单同规格且小测无失败时，脚本按“小测墙钟×全量输出数/小测数”给估计；混合原生尺寸、fps、编码或帧数头信息则不输出单一估计，应分组小测。估计可能放大固定加载成本，也不保证覆盖所有长尾，不能称置信区间或性能保证。

如果需要看暖缓存速度，再用新目录重复小测并声明`--cache-state warm`；首次报告仍保留。该字段是用户声明，不是缓存自动检测。默认单卡串行，不能将未来多卡线性外推写成实测；自行并行时使用整体墙钟，不相加重叠阶段。样式见[用时报告样例](metric_examples/timing_report.example.md)。

## 6. 历史参考与验收清单

历史一次测试处理4–10秒、12fps、49–121帧、约720p横竖屏输出，权重已在本地：五项GPU指标使用4张H200、56 CPU可用、384GB主存，按阶段墙钟分离约554个输出/小时；CPU单worker的原始闪烁约1775个输出/小时。四卡质量阶段总计约2小时10分钟处理1200个输出。这是旧混合视频的条件，期间其他卡部分时间还在跑后续RAFT，并非专门关闭静态筛选的独立复测，也没有单卡吞吐实测。不能据此保证你的纯10秒视频、另一张显卡或本示例串行实现的速度。

交付前检查：输入身份与时间范围可追溯；评分视频完整解码；无损投影像素校验；每个期望视频/维度键都有有限值或明确错误；成像质量只归一化一次；逐视频及模型均值正确；用时单位和冷启动范围清楚。静态筛选不在验收项中。视觉审核如另行进行，需真实查看完整时间序列并绑定输出，不能由指标自动认定编辑成功。
