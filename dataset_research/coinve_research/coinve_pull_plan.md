# CoinVE 拉取记录

> 状态：15 个完整随机 pair 已完成；不继续下载全量。

1. 固定官方 metadata 和 revision，从 200,916 行中以 seed `20260821` 无放回均匀抽 15 条。
2. 根据 manifest 定位 source/target tar 和实际 member basename。
3. 通过稀疏 HTTP Range 只提取命中的 30 个原始 MP4，不保存完整 tar。
4. 生成 H.264 comparison、instruction、contact sheet、inventory、audit 和下载 ZIP。

停止原因：15 条已经足以显示人物属性覆盖与主要故障模式；完整发布约 2.8 TB，没有必要盲目扩量。若后续采用 CoinVE，应先在 metadata 中筛人物属性候选，再只拉命中视频和 mask。

