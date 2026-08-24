# ViViD 随机抽查

直接观察：

- `vivid_random_pairs_15_h264.zip`：下载到本地查看；
- `examples/comparison/`：15 个 `MASKED INPUT | GARMENT | TARGET` H.264 MP4；
- `random_pairs.csv`：固定随机样本；
- `vivid_audit.csv`：人工审核；
- `vivid_report.md`：结论。

抽样总体为官方三类 train/test 清单中的全部 9,700 对。seed `20260823`，无放回均匀随机 15 对，不使用官方展示样例。

它是 garment-reference 条件下的遮罩重建：masked input 由 target 自身生成，不是“人物穿 A → 同人物穿 B”的真实编辑 pair。
