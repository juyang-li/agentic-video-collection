# Stage 2 正式筛选报告

日期：2026-09-05  
固定数据选择 seed：`20260904`

## 最终结果

Stage 2 已完成。最终规则是：所有进入 Stage 2 的样本都必须有 Gemini 3.8 Flash 判断；Qwen/rescue cascade 只保留为历史 benchmark 与诊断字段，不再拥有 canonical 最终决定权。

原池 7,185 条在全量 Gemini 复核后得到 3,863 accepted，三个低风险分支出现 193 条严格配额缺口。Round 4 从上游补取 390 条并完整运行 Stage 1，373 条 survivor 经 Gemini 直筛后新增 278 accepted。最终合并候选为 7,558 条，其中 4,141 accepted、3,402 rejected、15 quarantine；固定 seed 从 accepted 中选出严格 4,000 条。

| 分支 | accepted 可用 | 最终选择 | reserve |
| --- | ---: | ---: | ---: |
| GOKU remove | 602 | 600 | 2 |
| GOKU add | 420 | 400 | 20 |
| EgoEdit replace | 1,026 | 1,000 | 26 |
| ReCo add | 622 | 600 | 22 |
| ReCo remove | 634 | 600 | 34 |
| ReCo replace | 837 | 800 | 37 |
| **合计** | **4,141** | **4,000** | **141** |

共享媒体按连通组防泄漏划分，得到 train 3,601、validation 195、test 204。

## 运行阶段

| 阶段 | 输入 | 输出摘要 |
| --- | ---: | --- |
| 历史 rescue cascade | 7,185 | 4,304 accepted / 2,868 rejected / 13 quarantine |
| 1,882 条 Qwen 低风险直过的 Gemini 补录 | 1,882 | 1,441 accepted / 439 rejected / 2 provider-blocked |
| 原池统一重合成 | 7,185 | 3,863 accepted / 3,307 rejected / 15 quarantine |
| Round-4 Stage 1 | 390 | 340 accepted / 33 uncertain / 17 rejected；373 survivor |
| Round-4 Gemini 直筛 | 373 | 278 accepted / 95 rejected / 0 error |
| 最终合并 | 7,558 | 4,141 accepted / 3,402 rejected / 15 quarantine |

Round 4 只扩充缺口分支：EgoEdit replace 80、ReCo add 140、ReCo replace 170。所有媒体、技术指标、MUSIQ 和 DINOv2 分数完整；未跳过 Stage 1，也未为补齐配额降低阈值。Stage-1 uncertain 仍进入 Gemini，由 Gemini 作最终二分类。

## 模型与成本

Gemini 固定为 `google/gemini-3.8-flash`、`reasoning_effort=minimal`、`max_tokens=2048`，使用 5 个避开首帧的对齐时刻，每个时刻包含全图 pair 和 tight ROI pair。prompt SHA256 为 `32e0d04ebc02bd292725425f144e6eab9d36b7915a39a4da2b5511176394212c`。

- 历史 Stage-2 rescue cascade 正式调用记录：约 `$59.1549`。
- 1,882 条全量补录：`$10.7689461`。
- Round 4 最终结果 CSV：`$2.13383625`；加上被覆盖的一次无 JSON 失败请求后，实际本轮请求约 `$2.14649`。
- 三部分合计约 `$72.07`。账户总用量还包含 benchmark 与提示词实验，不能与该生产和补录合计直接等同。

## 人工抽查包

最终包为黑色单 pair 翻页审核器，source/target 同屏同步播放，顶部显示 dataset、branch、provenance、generated 侧、最终状态、路由、是否进入 4K 和抽查类别；可以不打分直接翻页。

100 条不是纯随机，而是固定配额分层：最终 4K 20、accepted reserve 10、Gemini 推翻旧低风险通过 15、原高风险 Gemini reject 15、Round-4 Gemini accept/reject 各 10、Round-4 Stage-1 reject 10、quarantine 10。六个任务分支全部覆盖。

## Canonical 文件

- 最终 4K：`artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv`
- accepted reserve 与全部未通过项：`artifacts/final_edit_pool_gemini_full_refill_v1/`
- 全量合并决定：`artifacts/gemini_refill_round4_v1/cascade_rows_gemini_full_plus_refill.csv`
- Gemini 全量补录：`artifacts/gemini_full_review_v1/gemini38_supplement_scores.csv`
- Round-4 Gemini：`artifacts/gemini_refill_round4_v1/gemini38_scores.csv`
- Round-4 Stage 1：`../1_specialist_filtering/artifacts/filter_run_v3/expansion_round4_refill_stage1_1_summary.json`
- 最终审阅包：`artifacts/gemini_full_final_review_100_browser_v1/`

百度网盘已上传：

- `/apps/bypy/lance_review/gemini_full_final_review_100_browser_v1.tar`：399,984,640 bytes，SHA256 `dfdc09194e297e3e87ea4c162cea3c13e4434a2b8a03ca999dfcc110741feaa2`。
- `/apps/bypy/lance_review/gemini_stage2_final_results_v1.tar`：42,977,280 bytes，SHA256 `0da7e017f57a2fdd4790b70e9393c4c4a33e4f18afef417ad531df3c33c91650`。

更详细的 token、prompt、缺口回填和安全拦截说明见 `GEMINI_FULL_REVIEW_STATUS.md`。
