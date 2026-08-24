# Ditto 拉取记录

> 状态：15 个 `local/0130` 随机 pair 已完成；不继续下载全量。

1. 从官方 `training_metadata/local.json` 中筛出 `local/0130` 的 472 个完整 triplet。
2. seed `20260821` 无放回均匀随机抽取 15 条。
3. 从 `local.tar.gz.01` 起顺序解压，扫描约 785 MB 后找齐 30 个 member。
4. 只保存命中 MP4，不保存 21.5 GB 分卷或中间 gzip。
5. 生成 H.264 comparison、instruction、contact sheet、inventory、audit 和下载 ZIP。

停止原因：15 条仅 1 条涉及人物配饰，且多条存在素材叠加感、修改范围扩大或非编辑背景变化。继续盲扫 `local/0130` 的人物属性收益较低；全量随机访问又可能要求读取约 530 GB 连续压缩流。

