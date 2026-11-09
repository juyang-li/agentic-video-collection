# InsAVE general_editing single-step replacement 抽查

本目录从 InsAVE-80K `general_editing` 的固定 `part_04.tar` 子总体中，筛出无显式音频要求、以 `Replace` 开头且没有明确第二操作的 forward instruction，再以 seed `20260826` 无放回随机抽10个完整视频 pair。

选择 `part_04` 是为了把本轮临时下载从约20 GB降到约4 GB，不依据视频质量；因此结果只代表该固定分片内的 replacement 候选，不能直接外推整个 `general_editing`。

官方 revision：`8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289`。

抽查已经完成：10 对中 instruction 严格通过 0、失败 8、存疑 2，因此这套简单文本子总体不能直接作为 replacement baseline。完整结论已合并进 [`../insave_report.md`](../insave_report.md)。

文件：

- `random_pairs.csv`：随机10条 manifest；
- `general_replacement_inventory.csv`：客观规格与文件哈希；
- `general_replacement_audit.csv`：逐项人工审核结果；
- `examples/comparison/`：左右 H.264 对照与 instruction；
- `examples/contact_sheets/`：六帧同步 contact sheet。
- `insave_general_replacement_random_pairs_10_h264.zip`：主报告、表格、对照视频和 contact sheet 的可下载复核包。

最终打包校验后已删除 20 个独立 source/target MP4、临时审阅帧、完整 metadata 缓存和下载分片；它们都可以由固定 revision、manifest 和脚本重新获取。comparison 与 contact sheet 保留为无需解包即可查看的审核材料。
