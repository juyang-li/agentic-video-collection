# GOKU-2M `remove` 随机 pair 结论

## 结论

任务形态是正确的：每条都是独立的 `remove`，不是 CoinVE 式 2–5 个操作混合。随机 15 条中 13 条完成指令，9 条通过 instruction、逻辑、保持、时序、视觉和技术六项严格门槛，明显好于此前 GOKU `swap_alter` 的 6/15 指令通过率。

但它仍不如 OpenVE 那样可以直接按任务类别扩大：7/15 属于目标人物属性或人物所持/所穿物体，其中 6 条完成指令；手机、笔记本被移除后仍保留使用姿势，T 恤移除的人体重建偏软。其余多为背景植物、杯子、墙板等一般物体。

## 抽样

- shard sampling：32 个 `remove` 分卷中固定 seed `20260823`，命中 `part32`。
- population：卷内有 5,459 个 source/target 完整 case，5,438 个同时有可用 annotation，后者为抽样总体。
- pair sampling：同一 seed 从 5,438 条中无放回均匀抽 15 条。
- 所有 source/target 都是 1280×720、202 帧、25 fps、8.08 秒，规格逐对完全一致并可完整解码。

## 审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 7 | 8 | 0 |
| instruction correctness | 13 | 2 | 0 |
| logical correctness | 12 | 3 | 0 |
| content preservation | 15 | 0 | 0 |
| temporal consistency | 15 | 0 | 0 |
| visual quality | 13 | 2 | 0 |
| technical quality | 15 | 0 | 0 |

失败的两条是“移除男士礼帽”但 source 中没有男士或礼帽，以及 target 仍完整保留衣架。另有三类严格问题：移除手持物后交互姿态不合理、移除 T 恤后人体区域偏糊、移除游乐场立柱后绳索连接件悬空。

## 决策

- 保留为 single-task `remove` 候选，质量等级低于 ReCo/OpenVE、高于本次 GOKU `swap_alter`。
- 不把 GOKU 各任务合并后再随机抽；训练前必须保留官方 task 字段，并对人物相关性、交互逻辑和指令完成度做过滤。
- 下一步优先补 OpenVE `local_add` 和 `local_remove` 的随机样本；随后再决定是否值得抽 GOKU `add`。
