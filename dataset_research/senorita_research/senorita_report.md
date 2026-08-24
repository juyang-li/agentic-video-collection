# Señorita object-swap 随机 pair 结论

## 结论

不建议把 Señorita 作为当前人物单任务编辑的主数据集。它的 `object_swap` 分支在数据组织上确实是原子任务、可按小卷获取，但本次随机一卷中的 15 对全部只有 592×336、约 4.1 秒，人物相关仅 2/15，严格指令通过 7/15。

## 方法

- population：完整 object-swap metadata 的 177,336 对；
- sampling：seed `20260822`，先做按样本量加权的随机选卷，再在 shard 144 的 1,000 对中无放回随机 15 对；
- acquisition：只下载命中的 458.8 MB 压缩卷并提取 30 个目标成员；
- review：H.264 左右 comparison、编号 instruction 和六帧 contact sheet。

这是一轮 cluster sample；结论能判断该卷和发现结构性问题，但不能把 15 对比例直接当成全库总体比例。

## 审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 2 | 13 | 0 |
| instruction correctness | 7 | 5 | 3 |
| logical correctness | 8 | 5 | 2 |
| content preservation | 12 | 1 | 2 |
| temporal consistency | 15 | 0 | 0 |
| visual quality | 15 | 0 | 0 |
| technical quality | 15 | 0 | 0 |

严格候选按 instruction、logical、content、temporal、visual、technical 全部通过计算，共 6/15；其中人物相关严格候选只有 chef→police officer 这 1 对。

## 主要问题

- 任务前提错误：robot→woman 的 source 已经是 woman；mountains→field 的 source 并非清楚的 mountains；
- 指令损坏或不可判定：`Replace him with he.`，以及 pipe/cylinder、rapids/waterfall 的差别不清；
- 局部保持失效：smoke→cloud 同时把暗背景整体换成天空；
- 人物密度低：只有 2/15 出现人物编辑，而且属于整个人/职业替换，不是脸、发型、首饰等属性修改。

## 决策

保留本轮 15 对作为证据，不继续下载 Señorita。若将来需要通用 object swap，可对 metadata 先做语义筛选，再按卷下载并执行严格审核；当前人物属性路线不值得为它扩大成本。
