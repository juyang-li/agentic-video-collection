# TripVVT 纯 CPU 相似度校准与基准

程序保存在上级目录 score_pair_similarity.py。已保存原始测算 initial_benchmark.json、可复跑的 input_manifest.csv、完整 75 对 pair_similarity.csv / pair_similarity.json 及 replay.log。75 对包括新抽查 50 对及此前换衣 25 对，所有指标在独立脚本复跑后与原测算一致（绝对误差 <1e-9），最低变化前 10 名完全相同。

## 复跑

在本目录运行：

```bash
python ../score_pair_similarity.py --manifest input_manifest.csv --out rerun --workers 4 --benchmark
```

输入清单字段：sample_id、category（upper_body/lower_body/dresses）、source、target；reference 和 cohort 可选。路径相对输入清单所在目录。已缓存 source_frames/target_frames/source_fps/target_fps 时无需额外 ffprobe；缺失时脚本自动读取。只输出分数，不改源文件，不自动排除样本。

依赖 Python 3.11+、numpy、Pillow、FFmpeg/ffprobe，当前在 Linux 纯 CPU 上运行。--ffmpeg /path/to/ffmpeg 可覆盖默认路径，--ffprobe 可单独指定。--workers 控制并发。--benchmark 对首 10 对串行计时，再对全部样本并发计时；--serial-samples 可修改串行样本数。不要将不同样本数的串行/并行记录解释为严格加速比。

## 参数

在 source/target 共同最后有效帧时刻的 10/30/50/70/90% 位置采样。本批均为 121 帧、24fps，帧号为 12、36、60、84、108。FFmpeg area 缩放到 144×256 RGB24，每个进程固定解码和滤镜线程为 1。

计算：63 位 pHash（32×32 灰度 DCT，取左上 8×8 并去 DC）、8×8 非重叠分块灰度 SSIM（不是 skimage 默认滑窗实现）、RGB MAE 除以 255。类别区域横向 15–85%，纵向上衣 15–60%、下装 45–95%、dresses 15–95%；边界向下取整。每个指标按 5 帧取中位数。

区域为固定位置，不是服装分割；颜色差异与结构相似度不能证明语义 no-op。pHash 可能忽略颜色变化，全图相似度可能受保持良好的背景主导。本轮只选最低变化的 10 对供人工确定阈值，不对正式数据作过滤。

## 实测速度

| 测算 | 串行首10对 | 4并发全部75对 | 4并发1K对线性估计 |
|---|---:|---:|---:|
| 原始测算 | 22.52s | 38.64s | 8.59min |
| 独立脚本复跑 | 54.10s | 47.82s | 10.63min |

实测基于已下载素材、可用 CPU affinity 为 64，实际仅开 4 个并发任务，无 GPU、无模型调用。不含下载时间，操作系统缓存和共享存储/CPU 负载会影响速度。对相近 720×1280、121 帧的 1K 对预留约 10–15 分钟，以上是测算而非耗时保证。环境、逐对抽帧/指标耗时与参数完整保存在 pair_similarity.json。

最低变化审核包在同级 tripvvt_lowchange10_20261108/。完整 75 对原始媒体仍由输入清单引用；临时 NPZ 缩略帧不是复现依赖。后续目标为训练10,000＋验证500＋测试500，总计11,000条；相似度阈值等待人工审核后确定。
