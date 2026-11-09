# TripVVT 低变化 10 对：阈值校准审核

完整解压，用 Chrome/Edge 打开 index.html。按服装区域 RGB 差异中位数从低到高取 10 对；同分按 ID。来自 75 对本地样本：上衣 3、下装 5、dresses 2。

## 如何审核

三栏为完整 source、单张服装参考、真实 target。prompt 位于素材上方，官方 text 单独标注。相似度数值默认折叠。请重点填写“是否有实际换衣变化”和“是否匹配参考图”，可备注你希望保留或剔除的情况。

1=通过、0=不通过、2=存疑；默认 1、reviewed=false 不是人工通过。支持同步播放/暂停/跳转、类别筛选和 CSV/JSON 导入导出。阈值未冻结，正式数据池未筛除。

## 排名

| 排名 | ID | 类别 | ROI RGB 差异 | 分块 SSIM | 哈希距离 |
|---|---|---|---:|---:|---:|
| 1 | tripvvt-08199 | lower_body | 0.024879 | 0.946651 | 2 |
| 2 | tripvvt-03779 | upper_body | 0.025575 | 0.960610 | 2 |
| 3 | tripvvt-02066 | lower_body | 0.026292 | 0.965659 | 2 |
| 4 | cloth_011 | lower_body | 0.027327 | 0.930039 | 4 |
| 5 | tripvvt-01796 | upper_body | 0.031385 | 0.964696 | 0 |
| 6 | cloth_012 | lower_body | 0.031443 | 0.935454 | 8 |
| 7 | tripvvt-09542 | dresses | 0.032827 | 0.905915 | 4 |
| 8 | tripvvt-02559 | lower_body | 0.033339 | 0.953427 | 2 |
| 9 | tripvvt-09859 | upper_body | 0.033806 | 0.964921 | 2 |
| 10 | tripvvt-05691 | dresses | 0.034004 | 0.944722 | 8 |

## 计算方法与复跑

纯 CPU，无视觉模型。每侧取共同最后有效帧时刻的 10/30/50/70/90% 时间点，缩放至 144×256，计算 63 位感知哈希、8×8 分块 SSIM 和 RGB 差异。RGB 差异除以 255 后归一化，区域横向均为 15–85%；上衣纵向 15–60%、下装 45–95%、dresses 15–95%。这是固定区域，不是服装分割，指标不等同于语义 no-op。

脚本依赖 Python、numpy、Pillow、FFmpeg/ffprobe。在本目录运行：

```bash
python score_pair_similarity.py --manifest scoring_input10.csv --out local_rescore --workers 4
```

FFmpeg 未在 PATH 时可用 --ffmpeg /path/to/ffmpeg 指定。完整 75 对输入清单、原始实测和复跑结果另保存在项目的 tripvvt_similarity_calibration_20261108/；本 ZIP 只包含排名前 10 对素材。

## 速度

初次实测：串行 10 对 22.52 秒，4 并发 75 对 38.64 秒；1K 对线性估计 8.59 分钟。
脚本复跑：串行 10 对 54.10 秒，4 并发 75 对 47.82 秒；1K 对线性估计 10.63 分钟。
估计不含下载；本地 I/O、缓存和 CPU 负载会影响结果。详细环境及参数见 benchmark.json。

manifest.json 保存逐条来源和语义；inventory.csv 保存排名、指令和哈希；similarity_scores75.csv 为完整分布；audit.csv 为未审核初始表。原始 50 对和 25 对审核包保持原版本，标注通过 ID 对应，不自动覆盖已有结果。

后续池规模已确定为训练 10,000＋验证 500＋测试 500，共 11,000 条；本次仅校准 TripVVT 筛除规则。
