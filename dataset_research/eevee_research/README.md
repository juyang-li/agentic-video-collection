# Eevee 随机抽查

直接观察：

- `eevee_random_pairs_15_h264.zip`：下载到本地查看；
- `examples/comparison/`：15 个 `MASKED INPUT | GARMENT | FULL SHOT | CLOSE-UP` H.264 MP4；
- `random_pairs.csv`：固定随机样本；
- `eevee_audit.csv`：人工审核；
- `eevee_report.md`：结论。

抽样总体为发布包实际存在的全部 9,363 个样本目录。seed `20260823`，无放回均匀随机 15 条，不使用官方展示样例。论文写 9,364 条，但发布仓库少 1 条，详见报告。

它是 garment-reference 条件下的遮罩重建。官方 test CSV 是 cross-garment 输入，但没有对应的新服装 ground-truth target；观察视频展示的是各目录自身的 matched raw record。
