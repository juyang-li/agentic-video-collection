# Señorita 随机 pair 抽查

本目录保留 Señorita `object_swap` 分支的一卷随机抽查证据。直接查看：

- `senorita_random_pairs_15_h264.zip`：适合下载到本地；
- `examples/comparison/`：15 个 H.264 左右对照 MP4 和同名指令；
- `random_pairs.csv`：可复现抽样记录；
- `senorita_audit.csv`：人工审核；
- `senorita_report.md`：结论。

抽样采用 seed `20260822`：先从完整的 177,336 条 object-swap metadata 中均匀抽一个 anchor，命中 shard 144；再从该 shard 的 1,000 对中无放回抽 15 对。因此 shard 是按样本量加权随机选出的，但 15 对是单卷 cluster sample，不等于全库 15 条独立随机样本。

压缩包内不放重复的 source/target 原文件，只保留可直接观察的 comparison MP4、指令、manifest 和 audit。

