# Gemini 全量复核与最终 4K

更新日期：2026-09-05  
固定数据选择 seed：`20260904`

## 结论

第二级筛选现已统一由 `google/gemini-3.8-flash` 覆盖，不再让 1,882 条 Qwen 低风险通过项缺少 Gemini 判断。完整补录后，原 7,185 条 Stage-2 候选降为 3,863 accepted，严格任务配额缺 193 条；随后从上游补取 390 条、完整重跑 Stage 1，并把 373 条 survivor 直接交给同一 Gemini。最终共有 4,141 条 accepted，可按原配比冻结为严格 4,000 条。

最终 4K 位于：

`artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv`

## 筛选流水

| 环节 | 输入 | accepted | rejected | quarantine/错误 |
| --- | ---: | ---: | ---: | ---: |
| 原 Stage-2 rescue cascade | 7,185 | 4,304 | 2,868 | 13 |
| 其中 Qwen 低风险直过的 Gemini 补录 | 1,882 | 1,441 | 439 | 2 provider-blocked |
| 统一重合成后的原池 | 7,185 | 3,863 | 3,307 | 15 |
| Round 4 Stage 1 | 390 | 340 hard accepted + 33 uncertain | 17 | 0 |
| Round 4 送入 Gemini | 373 | 278 | 95 | 0 |
| **合并后的 Stage-2 候选** | **7,558** | **4,141** | **3,402** | **15** |

Round 4 的 390 条原始候选由同一 seed 抽取，并与此前各轮去重：EgoEdit replace 80、ReCo add 140、ReCo replace 170。全部 780 个视频到齐；Stage 1.0 全过，Stage 1.1 淘汰 17 条，保留 373 条。Stage-1 uncertain 只表示 specialist 边界，仍由 Gemini 做最终决定。

## 最终分支配比

| 分支 | 全部 accepted 可用 | 冻结 4K | accepted reserve |
| --- | ---: | ---: | ---: |
| GOKU remove | 602 | 600 | 2 |
| GOKU add | 420 | 400 | 20 |
| EgoEdit replace | 1,026 | 1,000 | 26 |
| ReCo add | 622 | 600 | 22 |
| ReCo remove | 634 | 600 | 34 |
| ReCo replace | 837 | 800 | 37 |
| **合计** | **4,141** | **4,000** | **141** |

共享 source/target 媒体先组成连通组，再按 seed 整组划分，最终为 train 3,601、validation 195、test 204；不存在同一媒体跨 split 泄漏。

## 固定 Gemini 配置

- 模型：`google/gemini-3.8-flash`，provider 为 Google AI Studio，经 OpenRouter 调用。
- 思考等级：`reasoning_effort=minimal`；`max_tokens=2048`。
- 输入：避开首帧的 5 个对齐时刻；每个时刻提供 source/target 全图对和收紧的同坐标 ROI 对。
- prompt SHA256：`32e0d04ebc02bd292725425f144e6eab9d36b7915a39a4da2b5511176394212c`。
- 判定项：instruction following、visible artifact、content preservation；provenance 明确时只把生成侧作为主要质量对象，ReCo lineage 不明时按既定双侧规则。

1,882 条补录最终使用 12,671,880 input tokens、338,689 output tokens，结果文件记录费用 `$10.7689461`。Round 4 的 373 条最终有效结果使用 2,508,670 input tokens、67,289 output tokens，最终结果文件记录费用 `$2.13383625`；期间一条无 JSON 返回被重试，若计入已被覆盖的失败请求，本轮实际请求费用约 `$2.14649`。

## 审阅与使用边界

最终审阅包固定 100 对，分层包含：最终 4K 20、accepted reserve 10、Gemini 推翻旧低风险通过 15、原高风险 Gemini reject 15、补池通过/淘汰各 10、Round-4 Stage-1 淘汰 10、quarantine 10。所有六个分支都有覆盖；这不是纯随机样本。

模型筛选结果不是新增人工真值。审阅包用于发现系统性误杀/漏放和决定下一版规则，不回写本轮冻结模型结果；训练消融仍必须使用相同 seed 和同一最终 manifest。

## 权威文件

- 原池完整 Gemini 结果：`artifacts/gemini_full_review_v1/gemini38_supplement_scores.csv`
- 原池统一重合成：`artifacts/gemini_full_review_v1/cascade_rows_gemini_full.csv`
- Round-4 Stage 1：`../1_specialist_filtering/artifacts/filter_run_v3/expansion_round4_refill_stage1_1_summary.json`
- Round-4 Gemini：`artifacts/gemini_refill_round4_v1/gemini38_scores.csv`
- 合并后的全量决策：`artifacts/gemini_refill_round4_v1/cascade_rows_gemini_full_plus_refill.csv`
- 最终 4K 与 reserve：`artifacts/final_edit_pool_gemini_full_refill_v1/`
- 最终审阅包：`artifacts/gemini_full_final_review_100_browser_v1/`
