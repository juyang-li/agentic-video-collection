# Stage 1 正式筛选报告

日期：2026-09-05  
固定数据选择 seed：`20260904`  
规则版本：`general_edit_stage1_quality_v2`  
策略修订：`v2_reverse_removal_generated_side_20260905`

## 结论

本轮已按最新版 metadata + specialist 规则完成实际 Stage 2 所需候选的一级筛选。实际送入第二级的候选共 **7,185 对**：基础池 3,712、GOKU 扩展 2,256、EgoEdit 扩展 697、ReCo remove 扩展 520。Stage 1 与 Stage 2 之间没有人工审核或人工改写结论。

GOKU add 由 removal 生成结果反向组织成训练 pair，因此严格模糊/低细节门作用于 generated source；常规 GOKU remove 则作用于 generated target。ReCo 缺少逐 pair 原始/生成 lineage，继续检查两侧。

## 基础 5K 重算

| 环节 | 输入 | accepted | uncertain | rejected | survivor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1.0 metadata | 5,000 | 4,050 | — | 950 | 4,050 |
| 非 ReCo specialist（GOKU + EgoEdit） | 1,400 | 1,192 | 169 | 39 | 1,361 |
| ReCo specialist | 2,650 | 2,257 | 94 | 299 | 2,351 |
| **基础池合计** | **4,050** | **3,449** | **263** | **338** | **3,712** |

Stage 1.0 的 950 条拒绝包括：InsAVE person remove 整支 623、任务范围排除 262、衣物/鞋裤 add-remove 排除 179、手持物 remove 排除 60；同一样本可命中多个原因。修订后的 generated-side 规则比旧非 ReCo 输出额外淘汰 14 条 GOKU add，其中 3 条原先曾被 Stage 2 接受；最终池只使用修订后的 1,361 条结果。

## 缺口扩展

| 扩展来源 | Stage 1 输入 | accepted | uncertain | rejected | survivor | 实际送入 Stage 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GOKU remove + add | 2,400 | 2,205 | 51 | 144 | 2,256 | 2,256 |
| EgoEdit replace | 700 | 511 | 186 | 3 | 697 | 697 |
| ReCo remove | 900 | 740 | 15 | 145 | 755 | 520 |
| **扩展合计** | **4,000** | **3,456** | **252** | **292** | **3,708** | **3,473** |

ReCo remove 按固定 seed 从 755 个 survivor 中先送 520 条进入付费复核；其余 235 条保留为无 API 成本的候补。GOKU 又在相同两个已下载官方 tar 内把候选并集扩至 2,750 对，完整 Stage 1 后保留 2,583 对；扣除主批已有 survivor，得到 441 个不重复候补（remove 217、add 224）。这些候补尚未调用 API，只有主批最终配额不足才按精确缺口启用。

## 主要一级淘汰信号

- GOKU 扩展：`global_blur_low_detail` 80、`global_low_visual_quality` 41、`sustained_near_freeze` 39。
- ReCo remove 扩展：`sustained_near_freeze` 78、`global_low_visual_quality` 75、`global_blur_low_detail` 53、`extreme_freeze_repeat` 18。
- EgoEdit 扩展：只因 `sustained_near_freeze` 淘汰 3 条；MUSIQ/DINOv2 单项边界仍进入 uncertain，而非硬拒绝。

## 产物

逐样本结果、视频级分数、survivor manifest 和 JSON summary 位于 `artifacts/filter_run_v3/`。正式第二级入口分别为：

- `nonreco_stage1_survivors_corrected.csv`
- `reco_stage1_survivors.csv`
- `expansion_round1_goku_buffer_stage1_survivors.csv`
- `expansion_round2_ego_stage1_survivors.csv`
- `expansion_round3_reco_remove_buffer_stage1_survivors.csv`（其中固定抽取 520 条）

本报告只总结第一级生成质量门；instruction following、局部遮罩/残影和内容保持由冻结的 Stage 2 rescue cascade 决定。

## Round 4 严格配额补池（2026-09-05）

Gemini 全量复核使 EgoEdit replace、ReCo add、ReCo replace 分别缺 41、60、92 条。使用相同 seed `20260904`、相同 metadata 与 specialist 门槛从未使用候选中补取 390 条：EgoEdit replace 80、ReCo add 140、ReCo replace 170；与此前各轮 pair 全部去重。

780 个 source/target 视频全部到齐并完成技术检查，均满足 3 秒与可解码要求；743 个按 provenance 规则需要完整质量评分的视频全部取得 MUSIQ、sharpness、frame diff 与 DINOv2 指标。Stage 1.1 得到 340 accepted、33 uncertain、17 rejected，accepted+uncertain 共 373 条进入 Gemini，没有人工审核口，也没有放宽阈值。

| 分支 | 输入 | accepted | uncertain | rejected | Stage-2 survivor |
| --- | ---: | ---: | ---: | ---: | ---: |
| EgoEdit replace | 80 | 57 | 22 | 1 | 79 |
| ReCo add | 140 | 122 | 5 | 13 | 127 |
| ReCo replace | 170 | 161 | 6 | 3 | 167 |
| **合计** | **390** | **340** | **33** | **17** | **373** |

权威入口为 `artifacts/filter_run_v3/expansion_round4_refill_stage1_survivors.csv`，逐 pair 决策和汇总分别为 `expansion_round4_refill_stage1_1_pair_decisions.csv` 与 `expansion_round4_refill_stage1_1_summary.json`。
