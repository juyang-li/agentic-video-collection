# Gemini 全量补录状态

更新日期：2026-09-05

## 结论

当前正式 Stage 2 共处理 7,185 条 Stage-1 survivor，得到 4,304 条 accepted、2,868 条 rejected 和 13 条 quarantine，并从 4,304 条 accepted 中按固定 seed `20260904` 和冻结分支配额选出 4,000 条编辑池。

7,185 条中已有 5,303 条在正式级联里经过 Gemini；剩余 1,882 条走的是 `qwen_accept_low_risk`，此前只由 Qwen 直接放行。因此“Gemini 全量补录”只需处理这 1,882 条，而不是重新处理全部 7,185 条。

本次因余额限制先处理 700 条，得到 698 个有效 Gemini 判断：533 accepted、165 rejected；另有 2 条被 Gemini 内容安全策略拦截，保留为 quarantine，不伪装成模型 reject。尚余 1,182 条未调用。

```text
Stage-2 全量 7,185
├── 已经 Gemini 判断 5,303
└── Qwen 低风险直过 1,882  ← 补录全集
    ├── 本次已调用 700
    │   ├── Gemini accepted 533
    │   ├── Gemini rejected 165
    │   └── provider-blocked / quarantine 2
    └── 尚未调用 1,182
```

补录结果目前是附加评估层，**尚未改写** canonical Stage-2 决策、4,304 accepted 池或冻结 4K manifest。等 1,882 条覆盖完整后再统一重合成和重新按配额选池，避免产生一半新规则、一半旧规则的训练清单。

## 1. Stage 0 与 Stage 1

第零级 metadata 入选总体约 89 万条，来自 GOKU add/remove、EgoEdit replace 和 ReCo add/remove/replace。InsAVE person remove 因人工抽查质量低而整支停用；GOKU/InsAVE 衣物 add/remove、手持物 remove、style、camera 和 multi-step 等已知不适用任务在 metadata 层排除。89 万是上游可选总体，不是已经下载或逐条跑模型的数量。

当前实际进入 pilot 与扩容 manifest 的候选共 9,000 条：基础 5,000、GOKU 扩容 2,400、EgoEdit 扩容 700、ReCo remove 扩容 buffer 900。

| 步骤 | 输入 | 保留/进入下一候选层 | 淘汰或未进入 |
| --- | ---: | ---: | ---: |
| Stage 1.0/1.1 合计 | 9,000 | 7,420 survivor | 1,580 rejected |
| Stage 2 实际入口 | 7,420 survivor | 7,185 | 235 条 ReCo remove survivor 留作未调用 reserve |

其中 1,580 条 Stage-1 reject 包含基础 5K 的 950 条 metadata/任务硬排除，以及各轮共 630 条 specialist 质量淘汰。Stage 1.1 主要检查生成/编辑侧的近冻结、低 MUSIQ、低 sharpness、全局模糊/低细节与 DINOv2 异常跳变；ReCo provenance 无法逐条确定生成侧，因此两侧都查。Stage-1 uncertain 仍作为 survivor 进入 Stage 2，由 VLM 最终判断，不等同于自动通过。

基础 5K 的 950 条硬排除理由可重叠，主要包括：InsAVE 分支 623、任务范围不符 262、衣物 add/remove 179、手持物 remove 60，以及少量 style/camera/旧 target scope 问题。扩容 manifest 在抽样前已经应用相同 metadata 规则，所以其 Stage 1.0 显示全部通过，并不表示原始总体没有被 metadata 排除的样本。

## 2. 正式 Stage 2 级联

正式级联的最终决定路径如下：

| 路径 | 数量 | 最终决定者 |
| --- | ---: | --- |
| `gemini_direct_preclassified_risk` | 2,963 | 已知高风险，跳过 Qwen，Gemini 决定 |
| `gemini_confirm_qwen_accept_high_risk` | 1,278 | Qwen 接受但属于高风险，Gemini 决定 |
| `gemini_decides_qwen_rejected` | 1,062 | Qwen 拒绝，Gemini 复核并可救回 |
| `qwen_accept_low_risk` | 1,882 | Qwen 直接放行，此前没有 Gemini 判断 |
| **总计** | **7,185** |  |

前三条路径合计 5,303 条，全部已有 Gemini 判断。最后 1,882 条就是本轮补录全集。

Stage-2 正式结果：

| 分支 | Stage-2 输入 | accepted | rejected | quarantine | accepted 率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GOKU remove | 1,773 | 602 | 1,166 | 5 | 34.0% |
| GOKU add | 1,251 | 420 | 827 | 4 | 33.6% |
| EgoEdit replace | 1,290 | 1,108 | 181 | 1 | 85.9% |
| ReCo add | 740 | 698 | 41 | 1 | 94.3% |
| ReCo remove | 1,237 | 634 | 601 | 2 | 51.3% |
| ReCo replace | 894 | 842 | 52 | 0 | 94.2% |
| **合计** | **7,185** | **4,304** | **2,868** | **13** | **59.9%** |

remove 与反向构造的 GOKU add 被视为高风险，已经全部进入 Gemini，因此不在 1,882 条补录池中。补录只涉及普通低风险的 EgoEdit replace、ReCo add 和 ReCo replace。

## 3. 4,304 accepted 与冻结 4K

4,304 条 accepted 不是全都用于训练。固定 seed `20260904` 按冻结配额选择 4,000 条，304 条为 accepted reserve：

| 分支 | accepted 可用 | 冻结 4K | reserve |
| --- | ---: | ---: | ---: |
| GOKU remove | 602 | 600 | 2 |
| GOKU add | 420 | 400 | 20 |
| EgoEdit replace | 1,108 | 1,000 | 108 |
| ReCo add | 698 | 600 | 98 |
| ReCo remove | 634 | 600 | 34 |
| ReCo replace | 842 | 800 | 42 |
| **合计** | **4,304** | **4,000** | **304** |

冻结 4K 当前 split 为 train 3,615、validation 189、test 196；共享 source/target 媒体的样本被放在同一个 split。

## 4. 1,882、700 与 1,182 的关系

1,882 条补录全集构成为：EgoEdit replace 694、ReCo add 588、ReCo replace 600。其中 1,714 条属于当前冻结 4K，168 条属于 accepted reserve。

为了在有限余额下先均衡观察三个分支，1,882 条按固定 seed 和分支交错排序。本次取前 700 条：EgoEdit replace 234、ReCo add 233、ReCo replace 233。它们与冻结 4K 的关系为：

| 补录范围 | 总数 | 当前冻结 4K 内 | accepted reserve 内 |
| --- | ---: | ---: | ---: |
| 已跑首批 | 700 | 532 | 168 |
| 尚未运行 | 1,182 | 1,182 | 0 |
| **合计** | **1,882** | **1,714** | **168** |

这不是质量优先顺序。补录排序和冻结 4K 选择恰好使用同一 seed 的相反哈希端，因此 168 条 reserve 全部落在首批 700 内，剩余 1,182 条全部属于当前冻结 4K。哈希与质量无关，所以首批仍可用于估计三个分支的模型拒绝倾向；但如果目标是优先保护实际训练清单，接下来应直接续跑这 1,182 条，不能再从别处抽样替代。

## 5. 首批 700 条结果

| 分支 | 调用数 | Gemini accepted | Gemini rejected | provider blocked | 有效判断拒绝率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EgoEdit replace | 234 | 181 | 53 | 0 | 22.6% |
| ReCo add | 233 | 170 | 62 | 1 | 26.7% |
| ReCo replace | 233 | 182 | 50 | 1 | 21.6% |
| **合计** | **700** | **533** | **165** | **2** | **23.6%** |

在 698 个有效判断中：instruction following fail 84、visible artifact fail 104、content preservation fail 67；多个失败项可以落在同一个样本上。两条 provider blocked 均为 Gemini `PROHIBITED_CONTENT`，没有模型判断，最终合成时应进入 quarantine。

首批 700 中属于冻结 4K 的 532 条得到：405 accepted、125 rejected、2 blocked。若现在只在原 4K 上直接删除，不做候补回填，会暂时剩 3,873 条；若从全部 4,304 accepted 中按原配额重新选择，当前总量仍有 4,137 条，但 ReCo replace 只剩 791 条，较固定 800 配额缺 9 条，因此严格配额池只能形成 3,991 条。这只是首批覆盖的中间推演，不写回正式 manifest。

若首批 23.6% 的有效拒绝率粗略延续到剩余 1,182 条，预计还会新增约 279 条 rejected，最终 accepted 池可能降到约 3,850–3,870 条。该数字只是无人工标签的外推，不是最终结果；完整补录后需要补充上游候选或重新决定任务配额，不能为了维持 4K 而放宽质量门。

## 6. 为什么发起全量 Gemini 补录

冻结 230 条人工 benchmark 上，Gemini-only 比当前 rescue cascade 更重视坏样本召回，但会多误杀一些好样本：

| 方案 | 范围 | rejected recall | accuracy | accepted false-reject rate |
| --- | --- | ---: | ---: | ---: |
| Gemini-only | 全 230 | 81.5% | 77.4% | 23.9% |
| rescue cascade | 全 230 | 75.9% | 80.4% | 18.2% |
| Gemini-only | test 91 | 84.6% | 82.4% | 18.5% |
| rescue cascade | test 91 | 80.8% | 86.8% | 10.8% |

用户明确表示坏样本召回更重要，因此补录 1,882 条 Qwen 直过样本，用同一 Gemini 判断覆盖全部 Stage 2，而不是只让高风险样本进 Gemini。补录本身没有新的人工真值，首批 165 个 reject 仍需要抽查后才能判断新增召回与新增误杀的真实比例。

## 7. 调用成本与后续动作

首批 700 条的最终结果文件记录 Gemini 成本 `$3.9915`，平均输入 6,737 tokens/对；总输入 4,702,684 tokens、总输出 125,214 tokens。OpenRouter 账户查询暂显示从约 `$4.7906` 降至 `$0.9733`，与逐条账单存在少量结算时差。

剩余 1,182 条按当前均价预计约 `$6.74`。扣除现有约 `$0.97` 后，净缺口约 `$5.77`；充值 `$7` 可覆盖价格波动和少量重试。续跑按 sample ID 恢复，698 条已有有效结果不会重复请求。

后续顺序：

1. 续跑剩余 1,182 条；
2. 将持续安全拦截项置为 quarantine；
3. 用 1,882 条完整 Gemini 结果统一重合成 7,185 条 Stage-2 决策；
4. 重新统计各分支 accepted 与固定配额缺口；
5. 不足的分支从 Stage-1 reserve 或上游 metadata 池补候选，并走相同筛选；
6. 重新冻结 4K 和 leakage-safe split；
7. 对新增 reject、原 cascade accept/Gemini reject 分歧及最终 accepted 做分层人工抽查。

## 8. 权威文件

- 正式 Stage-2 报告：`STAGE2_PRODUCTION_REPORT.md`
- 当前冻结 4K：`artifacts/final_edit_pool_4k_v1/final_edit_pool_4k.csv`
- 补录全集与选择摘要：`artifacts/gemini_full_review_v1/supplement_input.csv`、`supplement_input_summary.json`
- 首批逐样本结果与调用摘要：`artifacts/gemini_full_review_v1/gemini38_supplement_scores.csv`、`gemini38_supplement_summary.json`

