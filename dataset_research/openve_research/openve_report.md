# OpenVE-3M 两批随机 pair 阶段结论

## 结论

OpenVE 的 `local_change` 值得保留为人物属性编辑候选，但不能未经清洗直接扩量。两批合计随机抽取 30 条，其中 19 条人物属性相关；人物相关记录中只有 10 条指令通过、8 条失败、1 条存疑。再排除第一批 1 条视觉上疑似生成素材后，当前严格真人人物属性候选约为 9/30。

文件技术质量整齐，主要风险是语义错误以及同一视频 pair 的重复、反向和改写记录。

## 方法

- base population：`local_change.csv` 中 source/target 都属于 local-change 的 230,808 条记录。
- downloadable population：固定 2 GiB 压缩流前缀内，两个视频都完整出现的 39 条记录。
- batch 1：seed `20260821`，从 39 条中按 metadata 行随机抽 15 条。
- batch 2：排除 batch 1 的 15 条 metadata，seed `20260822`，从剩余 24 条随机抽 15 条。
- review：两批均生成 H.264 左右 comparison、instruction 文本和六帧 contact sheet，并逐条人工审核。

两批合并相当于在该 39 条固定前缀子总体中顺序无放回抽取 30 条，但不能代表完整 `local_change` 或 OpenVE-3M。官方展示案例没有公开随机选择过程，未纳入审核。

## 分批结果

| 指标 | batch 1 | batch 2 | 合计 |
| --- | ---: | ---: | ---: |
| metadata rows | 15 | 15 | 30 |
| person relevance | 10 | 9 | 19 |
| instruction pass | 10 | 10 | 20 |
| instruction fail | 5 | 4 | 9 |
| instruction doubtful | 0 | 1 | 1 |
| unique directed pairs | 14 | 15 | 29 |
| unique unordered pairs | 12 | 12 | 19 |
| unique MP4 | 24 | 24 | 38 |

第二批人物相关记录为 9/15：5 条指令通过、3 条失败、1 条存疑。主要失败是未出现珍珠项链、不可见的帽子移除伴随服装整体改变，以及 source 本身没有 prompt 所称的商务西装。另有一条确实移除了太阳镜，但 prompt 把男性误写为女性。

## 合并审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 19 | 11 | 0 |
| instruction correctness | 20 | 9 | 1 |
| logical correctness | 30 | 0 | 0 |
| content preservation | 29 | 1 | 0 |
| temporal consistency | 30 | 0 | 0 |
| visual quality | 30 | 0 | 0 |
| technical quality | 30 | 0 | 0 |

60 个角色文件全部可解码。每批均为 13 对 1280×720、2 对 720×1280；每对 source/target 的分辨率、帧数和 fps 完全一致。视觉通过只表示画面本身清晰稳定，不表示 instruction 一定正确。

## 重复风险

- batch 1 和 batch 2 各有 12 个唯一无向 pair，两批之间又重合 5 个无向 pair。
- 合计 30 条 metadata 最终只有 19 个独立无向视频 pair。
- 同一 pair 可能对应正反两个方向，也可能只改写 prompt 的主体或操作描述；按 metadata 行抽样会放大这些视觉案例的权重。

## 判断

- 优点：人物属性自然覆盖率为 19/30；原始 pair 规格一致；服装、头发、头饰、眼镜、首饰和身份外观均有覆盖。
- 缺点：人物相关指令严格通过率只有 10/19；30 条记录压缩后只有 19 个无向 pair；部分 prompt 的前提、主体或目标物与画面不符。
- 决策：保留两批共 30 条作为随机观察证据。若继续收集，应先按无向视频 pair 去重，再做人物语义、真人和指令正确性筛选；不建议继续从这个仅剩 9 条 metadata 的前缀总体盲目抽样，也不建议直接下载约 151.47 GiB 的完整 local-change 视频。
