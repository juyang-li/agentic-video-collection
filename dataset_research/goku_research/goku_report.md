# GOKU-2M 随机 pair 阶段结论

## 结论

这批 `swap_alter` 自然随机样本不适合作为当前人物属性数据源继续盲目扩量。15 条中人物属性相关为 0；唯一涉及生物外观的是犬种替换，仍不属于人物属性。

编辑执行质量也不稳定：指令严格通过 6/15、失败 8/15、存疑 1/15。文件本身则全部可解码且 source/target 规格一致，问题主要来自编辑语义和局部合成质量，而非封装或编码。

## 方法

- task：`swap_alter`；style 未进入本轮。
- shard sampling：seed `20260821`，从 32 个视频分卷均匀随机选中 `part05`。
- population：该分卷 8,260 个 annotation 有效、source/target 完整的 case。
- pair sampling：同一 seed，从 8,260 条中无放回随机抽取 15 条。
- review：H.264 左右 comparison、编号 instruction、六帧 contact sheet 和完整解码检查。

该设计代表 `part05` 的完整 case，但不能直接外推整个 `swap_alter`，更不能代表 GOKU-2M 的其他任务。官方展示案例没有公开随机选择过程，本报告未使用。

## 审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 0 | 15 | 0 |
| instruction correctness | 6 | 8 | 1 |
| logical correctness | 13 | 2 | 0 |
| content preservation | 15 | 0 | 0 |
| temporal consistency | 15 | 0 | 0 |
| visual quality | 13 | 2 | 0 |
| technical quality | 15 | 0 | 0 |

通过的编辑包括道路变河流、犬种替换、漂浮花朵变蜡烛、树冠变积云、森林变岩石和白昼变日落；其中道路变河流仍导致车辆在水面行驶，树冠变云也有明显遮罩残留。失败案例包括未形成溜冰场、尾迹、极光、蜿蜒河流、木栈道、河獭、繁星夜空和积雪河岸。河流变海洋只改变了水色，判为存疑。

## 判断

- 优点：官方组织结构清楚；单卷可独立获取；全部 pair 技术规格一致；可按 case 精确连接 metadata 和视频。
- 缺点：自然随机的 `swap_alter` 未覆盖人物属性，且严格指令通过率仅 40%；个别成功编辑仍存在场景逻辑或遮罩问题。
- 决策：保留这 15 条作为随机观察证据，不继续从该分卷盲目扩量。若以后专门收集 GOKU 人物样本，应先在全任务 metadata 中做人物语义筛选，再单独标注为“定向筛选”，不可混写为随机抽查。
