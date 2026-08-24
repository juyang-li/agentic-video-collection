# CoinVE-200K 数据笔记

- 论文：https://arxiv.org/abs/2608.17566
- 数据集：https://huggingface.co/datasets/FireCRT/CoinVE-200K
- 许可证：CC BY-NC-SA 4.0。
- 数据包含 200,916 个 1080p compositional editing 样本，每条有 2–5 个原子指令；完整发布约 2.8 TB。
- source/target 位于独立未压缩 tar，可以通过 HTTP Range 定位并只下载单个 member。

本轮固定官方 revision，从完整 metadata 均匀随机抽 15 对。抽取出的 30 个原始 MP4 全部为 1920×1080且可解码；10 对规格一致，5 对 source 长于 target，target 均截到 201 帧。它们在 comparison 校验完成后已作为重复中间文件删除。

15 个 comparison 均为 H.264 `avc1`、1920×540，只展示 source/target 的共同帧区间。
