# 视频编辑对GT测评：SSIM、PSNR、LPIPS

项目正式运行与实测经验见[测评经验汇总](METRIC_EVALUATION_LESSONS.md)。通用示例的CPU测试与正式实验验收分开记录，不能据此宣称通用示例已完成独立GPU验收。

这份指南面向在自己Linux/NVIDIA机器上评测新数据集的读者。你只需要自己的GT和生成视频，不需要原作者的数据、目录、账号或集群。随包提供[最小示例](metric_examples/gt_metrics.py)、[共享模块](metric_examples/common.py)、[输入样例](metric_examples/manifest.example.jsonl)和[测试](metric_examples/test_examples.py)。这不是训练或视频生成程序，也不包含任务调度、断点续跑平台或私有下载器。

主线是常见的10秒UE合成编辑对。**只比较输出与对应GT，不与SOURCE比较。** 三指标反映对单一GT的保真度，不直接代表编辑指令完成、参考身份匹配或时序质量。动作和镜头可以有合理多解，未编辑的大面积背景也会主导全图指标。输出自身质量另见[VBench指南](VBENCH_README.md)。

## 1. 安装与输入

建议新建独立Python 3.12环境，先按[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)安装匹配驱动的torch和torchvision，再安装：

```bash
python -m pip install -r metric_examples/requirements-base.txt
python metric_examples/gt_metrics.py --help
```

依赖清单记录了已使用版本，不是所有GPU/驱动组合的安装保证。参考测试环境是Python3.12、torch2.10、CUDA12.6；不要复制别人的虚拟环境或把CUDA12.6强加给自己的驱动。AlexNet与LPIPS权重来自公开软件的官方下载/缓存，首次使用可能联网；离线机器需事先准备这些缓存。不包含任何权重或凭据。

输入为UTF-8 JSONL，每行是一对输出与GT，路径相对于清单文件所在目录：

```json
{"id":"clip_001","model":"editor_a","prediction":"outputs/clip_001.mp4","gt":"targets/clip_001.mp4"}
```

`id`和`prediction`必填，GT指标还要求`gt`；`model`可省略，默认为default。`(id, model)`不能重复。数量、模型名称和数据集名称不固定。不同模型比较时，同一id必须对应同一语义片段和GT；脚本无法识别语义上错配的文件。示例只读取本地视频，数据需由使用者事先准备。

## 2. 我的做法：先时间轴，再有效画面

### 时间轴

默认预设为相对视频首帧的0–10秒、12fps、含末端，共121个采样点：`t=i/12, i=0..120`。两份输入必须已在语义上从同一起点开始；有起点偏移时先由使用者正确裁剪，不能根据分数搜索最佳偏移。原始容器首帧PTS会记录并归零为视频相对时间。

按实际PTS找最近帧，平局取后帧，数值平局精度1e-9秒。24fps输出通常选择0,2,…,240帧；额外尾帧完整解码检查后排除，不压缩时间、不插帧、不重复生成帧。最近时刻误差限为半个名义原生帧间隔加1e-6秒；重复选择生成帧或覆盖不足会失败。

GT有一个明确边界例外：例如30fps、300帧的10秒原件，末帧PTS约9.9667秒，但显示区间覆盖到10秒。仅最后一个采样点允许在一原生帧间隔内映射到这类合法末帧，并记录`gt_endpoint_quantized`；不允许用这个例外补长生成结果。

121帧是本文的显式预设，不是“所有10秒视频必然有121帧”。若你的协议为左闭右开0–10秒、12fps共120帧，使用`--endpoint exclusive`并在报告中保持该口径；不要静默混用。`--start`、`--end`、`--fps`也可配置，间隔乘fps须为整数。

### 有效画面

GT直接缩放一次；输出只在已知协议下去padding，再用Pillow LANCZOS缩放到相同有效尺寸。默认`--size auto720`按GT横竖屏选择1280×720或720×1280；其他宽高比使用例如`--size 720x720`。脚本拒绝任意拉伸宽高比。

默认不裁padding。若你明确知道输出使用GT原始宽高比的contain协议，传`--prediction-padding contain`：1920×1080放入1280×736会去掉上下各8像素；竖屏相应去左右各8像素。**不检测黑色像素猜边框。** 非contain裁剪可在清单中明确给出`gt_crop`或`prediction_crop`，格式为整数`[x,y,width,height]`；prediction_crop不可与contain同时使用。

解码使用PyAV转RGB，不自动颜色匹配。示例检查完整原生流的PTS和画幅，不支持视频内部变分辨率；VFR或异常时间戳应先确认协议。它不会自动修复错误数据。

## 3. 固定数值参数与汇总

| 指标 | 本示例参数 |
|---|---|
| SSIM ↑ | scikit-image，RGB浮点[0,1]，data_range=1，channel_axis=-1，11×11高斯窗，sigma1.5，use_sample_covariance=False |
| PSNR ↑ | RGB[0,1]，float64逐帧MSE，再求−10 log10(MSE)，单位dB；不先跨帧平均MSE再求PSNR |
| LPIPS ↓ | lpips0.1.4，AlexNet/version0.1，FP32，eval和inference_mode，输入RGB[-1,1]；无AMP、TF32、额外缩至256或分块 |

LPIPS默认batch4，OOM仅降batch，不改图像尺寸或精度；实际batch记录到结果。SSIM与PSNR使用联合核函数遍历，选择其中一项仅控制输出，不能把该联合计时称作另一项的独立耗时。

每项先逐帧平均得到视频均值，再对视频等权平均。模型汇总给出实际n、预期n和参与id；比较模型时还应核对参与id一致，不只看n一样。失败不会填0。某视频部分指标成功而另一项失败时，保留已计算值，同时将整条状态标failed，报告不宣称全量通过。

PSNR零MSE表示正无穷，保存`psnr_db=null`加`psnr_infinite=true`，不封顶，不当作缺失。任一帧无穷则对应视频均值无穷，任一视频无穷则对应模型汇总延续该标记。非法NaN不是这个例外。

## 4. 可直接验证的命令

先生成很小的自造10秒样例，验证安装、24fps对齐及恒等分数。这是功能测试，不是720p速度基准：

```bash
python metric_examples/make_demo.py --output-dir demo_data
python metric_examples/gt_metrics.py --manifest demo_data/manifest.jsonl \
  --output-dir demo_cpu --metrics ssim,psnr --device cpu --size 128x72
python metric_examples/gt_metrics.py --manifest demo_data/manifest.jsonl \
  --output-dir demo_lpips --metrics lpips --device cuda:0 --size 128x72
python metric_examples/test_examples.py
```

正式数据例子：

```bash
python metric_examples/gt_metrics.py --manifest my_pairs.jsonl \
  --output-dir pilot_gt --limit 8 --device cuda:0 \
  --cpu-workers 4 --torch-threads 2 --lpips-batch-size 4 \
  --prediction-padding contain
```

仅在确知contain时使用最后一个参数。清单前8行由你安排为有代表性的样本，不能按效果挑选；混合横竖屏、分辨率或解码格式时分别小测。去掉`--limit`即可评完整清单。输出目录必须新建或为空，避免混入旧结果；示例不做自动缓存复用或续跑。

普通工作站可串行运行三项；受调度器管理的机器应遵循自己的资源规则。示例不要求Slurm，不读取分区变量，不申请资源。若CPU与GPU阶段分开运行，用不同输出目录及`--metrics ssim,psnr`、`--metrics lpips`，按id/model合并结果，不能当作不同样本直接拼接取均值。

## 5. 速度估计与实际用时报告

每次生成`results.jsonl`（逐帧、逐视频、取帧审计及失败）、`summary.csv/json`、`timing.json`和`timing_report.md`。报告给出硬件、版本、线程、batch、协议、成功/失败数、视频/小时、帧对/秒和实际墙钟时间。**一对视频指一个模型输出与GT，不是一个标量分数，也不是包含多个模型的任务。**

GT计时包含启动、预检、加载AlexNet/可能下载、解码/对齐/哈希、SSIM＋PSNR联合计算、LPIPS传输及计算、指标结果写盘；GPU计算边界同步CUDA。首条和冷启动不隐藏。报告自身写盘与进程退出不在墙钟终点内。`--cache-state cold|warm|unknown`只是用户声明，脚本不假装能自动识别全部缓存。

`--limit 8`仍读取完整清单的头信息。只有完整清单的原生画幅、fps、编码、帧数头信息及归一化配置一致，且小测无失败时，才给同规格外推：

```text
预计全量秒数 = 小测实际墙钟秒数 × 清单输出数 / 小测输出数
视频/小时 = 完整成功输出数 × 3600 / 实测墙钟秒数
帧对/秒 = 完整成功视频的目标帧数之和 / 实测墙钟秒数
```

估计包含小测的启动开销，可能高估大批量的固定开销；它不是置信区间，也不是保证上界。异质或探测失败的数据不输出单一估计，须按规格分组小测。可以另跑一个新输出目录记录暖缓存小测，与首次运行分别保存；不删除首次开销后冒称总用时。

本示例是串行CPU/GPU工作流，总用时用实际墙钟；若自行实现CPU/GPU流水并行，应报告整段墙钟，不能相加并发阶段、也不能相加吞吐。单卡线性乘四不等于四卡实测。报告样式见[用时报告样例](metric_examples/timing_report.example.md)。

## 6. 经验、验收和历史参考

- 先检查首、中、末帧，确认同一语义时间段、横竖屏和padding。几何审核不等于编辑效果人工通过。
- Python JSON不接受所有NumPy标量，落盘前转Python float/bool；保留失败原因，不能失败后填0。
- 不把所有视频的规范化无损帧长期落盘，逐条处理即可。更改GT、输出、采样或参数后，旧分数不能直接复用。
- 检查完整期望键、逐帧数、无限PSNR标记及逐视频/模型均值。核心测试包括恒等SSIM=1、LPIPS≈0、PSNR∞、独立PSNR公式、24fps取偶数帧、GT合法终点、生成结果不足和错误画幅拒绝。

以下仅是过去一次4–10秒、12fps、有效720p混合视频的实测；**不是纯10秒UE或你的机器的保证**：

| 阶段 | 当时资源/条件 | 参考速度 |
|---|---|---:|
| SSIM＋PSNR | 4个CPU工作线程 | 约250对/小时 |
| SSIM＋PSNR | 16 CPU可用、12个工作线程 | 约400对/小时 |
| LPIPS | 单H200、16 CPU、128GB、FP32、batch4；1200对用1小时14分08秒 | 约970对/小时 |

当时两处CPU分片的整体CPU阶段约500对/小时；三项先CPU后GPU，首条CPU结果到GPU结束约3小时36分钟，1200对约330对/小时。模型分片、视频长度、IO和尾部等待均有差异，不把两处CPU速度相加为持续吞吐。本指南不要求这些机器、原数据、任何账号或私有脚本。请用自己的10秒UE小测和生成的实际用时报告做预算。
