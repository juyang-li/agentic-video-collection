# Ditto 数据笔记

- 官方数据集：https://huggingface.co/datasets/QingyanBai/Ditto-1M
- 许可证：CC BY-NC-SA 4.0。
- local 数据约 530 GB，以一个 tar.gz 流拆成 `.01`–`.27`；分卷不能独立解压，也不能按 tar member 随机跳转。
- 官网 `local` 与 `more-local` 都是 `local editing` 展示区，官方没有声明展示案例为随机抽样，因此不用于统计。

为避免扫描完整 local，本轮在归档开头目录 `local/0130` 内抽样。该目录对应 472 个 metadata triplet；固定 seed 随机抽 15 对后，从 `.01` 开头扫描 485 个 MP4、约 785 MB 压缩数据，保存 30 个 pair 文件。

抽取出的 30 个原始 MP4 均为 1280×720、101 帧、20 fps，全部可解码且每对规格一致。15 个 comparison 均为 H.264 `avc1`、1920×540、101 帧；独立原片在 comparison 校验完成后已作为重复中间文件删除。
