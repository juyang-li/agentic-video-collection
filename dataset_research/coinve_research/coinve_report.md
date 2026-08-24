# CoinVE-200K 随机 pair 阶段结论

## 结论

CoinVE 值得保留为合成人物编辑监督候选，但不能替代真实人物 raw data，也不建议下载约 2.8 TB 全量。

完整 metadata 随机 15 条中，9 条属于当前人物属性编辑；其中 1 条 source 是 3D 动画，所以真人属性候选为 8/15。覆盖以服装和配饰为主，没有身份替换、首饰或表情编辑。

## 方法

- population：官方 metadata 的 200,916 行。
- sampling：seed `20260821`，无放回均匀随机 15 条。
- acquisition：从独立未压缩 tar 中按 Range 只提取 30 个命中 MP4。
- review：H.264 左右 comparison、编号 instruction 和六帧 contact sheet。

官网展示案例没有公开随机选择过程，本报告未使用它们。

## 覆盖

- 人物属性相关：9/15；去掉 1 条 3D 动画后，真人候选 8/15。
- 服装相关 5 条、配饰相关 4 条、发型/发色 1 条、面部毛发 1 条；类别允许重叠。
- 身份、首饰、表情均为 0。
- 3 条 style-only 和 3 条纯物体/背景编辑不计入人物属性。

## 审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 9 | 6 | 0 |
| instruction correctness | 13 | 0 | 2 |
| logical correctness | 14 | 1 | 0 |
| content preservation | 14 | 1 | 0 |
| temporal consistency | 14 | 1 | 0 |
| visual quality | 14 | 0 | 1 |
| technical quality | 10 | 0 | 5 |

主要问题包括：style 只执行到部分身体；非编辑背景被重构；方向盘删除后双手仍保持操控姿势；太阳镜跨帧改变大小；5 对 source 长于 target 的 201 帧上限。

## 判断

- 优点：人物属性密度较高，1080p、结构化多指令、operation/object 和 mask 对编辑监督有价值。
- 缺点：仍是生成 target，存在内容保持和时间一致性问题；随机样本也包含非真人内容。
- 决策：保留 15 对作为随机观察证据。若后续扩展，先按人物属性语义筛 metadata，再拉视频和 mask；不直接下载全量。

