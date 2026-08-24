# TripVVT-10K 随机 triplet 抽查

直接观察：

- `tripvvt_random_pairs_15_h264.zip`：适合下载到本地；
- `examples/comparison/`：15 个 `SOURCE | GARMENT | TARGET` H.264 MP4；
- `random_pairs.csv`：完整随机抽样记录；
- `tripvvt_audit.csv`：人工审核；
- `tripvvt_report.md`：结论。

抽样 population 是官方 train 9,931 加 test 100 的全部 10,031 个 triplet。seed `20260822`，无放回均匀随机 15 条；命中 14 条 train、1 条 test，15 个 target 均不重复。

这里不是纯文本视频编辑。输入由 source video、garment reference 以及可选的 human mask/pose 组成；comparison 把 garment 放在中间，便于判断 target 是否正确对应参考服装。

压缩包只保留观察所需的 comparison、说明、manifest 和 audit；重复的原始 role 视频与 garment PNG 在产物验证后删除。

