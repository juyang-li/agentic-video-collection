# 视频编辑二分类人工数据集与强模型评测

> 数据版本：`explicit_binary_dataset_20260904_partial_v1`  
> 划分版本：`explicit_binary_prompt_eval_split_v1`  
> 当前状态：230 条明确二分类数据、冻结划分与 Gemini 评测均已完成。

## 1. 数据集范围

当前合并集只收录人工明确标为 `accepted` 或 `rejected` 的 pair。历史和本轮审核中的 `uncertain` 均不根据备注二次猜标签；未完成项也不进入数据集。

| 来源 | accepted | rejected | 合计 | 用途 |
| --- | ---: | ---: | ---: | --- |
| 此前明确人工标签 | 111 | 28 | 139 | prompt train/validation |
| 本轮 systematic 审核已完成的明确标签 | 65 | 26 | 91 | 最终 test |
| **合计** | **176** | **54** | **230** | - |

本轮 300 条审核目前另有 75 条 `uncertain` 和 134 条未完成，均未并入。因此这里的 91 条 test 是冻结 systematic 集的“当前已完成明确子集”，不是完整 300 条测试集；继续完成人工审核后应扩展 test，而不是重抽或改变现有划分。

## 2. 构成

| 数据集 | accepted | rejected | 合计 |
| --- | ---: | ---: | ---: |
| EgoEdit | 22 | 6 | 28 |
| GOKU | 60 | 28 | 88 |
| ReCo | 94 | 20 | 114 |

| 实际审核任务 | accepted | rejected | 合计 |
| --- | ---: | ---: | ---: |
| remove | 99 | 39 | 138 |
| replace | 52 | 12 | 64 |
| add | 25 | 3 | 28 |

remove 占 `138/230 = 60.0%`，rejected 中 remove 占 `39/54 = 72.2%`，符合当前重点检查 removal 缺陷的目标。provenance 路由为：已知 forward 70、inverse 41、双侧生成 5、`generation_side_unresolved` 114。后者全部来自 ReCo：instruction following 和 content preservation 按 `SRC→TARGET`，visible artifact 检查两侧。

## 3. 冻结划分

旧 139 条 development 按 `branch × human label`、固定种子 `20260904` 分层为 train/validation；本轮新审核的 91 条不参与 prompt 修改，整体作为 test。

| split | accepted | rejected | 合计 | 用途 |
| --- | ---: | ---: | ---: | --- |
| train | 76 | 18 | 94 | 分析错误并修改 prompt |
| validation | 35 | 10 | 45 | 选择 prompt，不再据此修改入选版本 |
| test | 65 | 26 | 91 | prompt 冻结后只运行一次 |

test 的 remove/replace/add 为 59/18/14，数据集 GOKU/ReCo/EgoEdit 为 31/56/4。三个 split 的 sample ID 及 SHA256 固定在 `artifacts/systematic_binary_benchmark_v1/labels/splits/split_summary.json`。

必须注意：train/validation 中的旧标签曾参与此前 Qwen/v12 规则研究，因此不能拿它们报告新管线的最终泛化能力；Gemini 本次从未看过的 91 条 test 才是主要结果。由于人工只按页面顺序完成了部分 systematic 样本，test 仍可能存在完成顺序带来的选择偏差。

## 4. 强模型与 prompt 调优协议

强模型固定为 OpenRouter `google/gemini-3.8-flash`，不使用会随时间重定向的 `flash-latest`。该模型支持图像输入和 JSON schema；调用继续使用五张对齐 FULL+TIGHT_ROI 时间卡、首卡避开 conditioning 首帧，并禁止 provider fallback。

调优步骤固定为：

1. 用当前 provenance-aware v12 prompt 在 train 上得到基线。
2. 只根据 train 的 false negative/false positive、人工备注和模型 evidence 产生一个精简候选 prompt；不查看 test 预测。
3. 基线和候选都在 validation 上运行。先比较 rejected recall，再比较 balanced accuracy、accuracy 和 accepted false-reject rate；只冻结一个版本。
4. 入选 prompt 只在 test 上运行一次。test 结果出来后不再修改该版本；若继续调 prompt，必须另建下一版 test。

人工标签中的 `uncertain` 无论备注偏正面还是偏负面都直接排除，绝不转成 `rejected`。参与本次计分的模型则必须输出二分类；若模型响应无法解析，不把它伪装成正确拒绝，而是重试并单列解析失败。主要指标定义为：

- rejected recall：人工 rejected 中被模型拒绝的比例；
- accuracy：全部明确人工标签中模型判断正确的比例；
- accepted false-reject rate：人工 accepted 中被模型拒绝的比例；
- reject precision：模型拒绝中真正人工 rejected 的比例；
- balanced accuracy：accepted recall 与 rejected recall 的平均。

最终同时报告总体、dataset、branch、audit operation、provenance relation，并将 train/validation 与 test 分开，不把三部分混成一个泛化数字。

## 5. 持久化文件

- `artifacts/systematic_binary_benchmark_v1/labels/explicit_binary_dataset.csv`：230 条规范人工标签；
- `artifacts/systematic_binary_benchmark_v1/labels/explicit_binary_v12_input.csv`：模型输入清单；
- `artifacts/systematic_binary_benchmark_v1/labels/explicit_binary_dataset_summary.json`：完整构成；
- `artifacts/systematic_binary_benchmark_v1/labels/splits/`：冻结 train/validation/test。

## 6. 历史 Qwen72B v12 的可比范围

历史 `general_edit_structural_v8_plus_eyewear_v9_filter_v12` 只覆盖当前二分类集中的 42 条，不能据此声称测试了全部 230 条。这 42 条还来自过去的定向高风险集和 holdout，且已参与系统开发，不是本轮 blind test。

| 模型/系统 | n | rejected recall | accuracy | accepted false-reject | balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 历史 Qwen2.5-VL-72B v12 可比交集 | 42 | 90.0% | 78.6% | 25.0% | 82.5% |

对应的 42 条标签、历史分数和重算结果位于 `artifacts/systematic_binary_benchmark_v1/qwen72b_v12_existing_overlap/`。这个数字只作为旧系统参考，不能和下面 91 条 blind test 的 Gemini 数字直接排名。

## 7. Gemini 3.8 Flash 的 prompt 选择

模型固定为 `google/gemini-3.8-flash`。v1 直接采用 provenance-aware v12 prompt，SHA256 为 `32e0d04ebc02bd292725425f144e6eab9d36b7915a39a4da2b5511176394212c`；v2 只依据 train 错误减少轻微伪影和属性细节带来的误杀，SHA256 为 `1db59793f37316af9ab6d65b1a6f8546b60ca70d35715a5511967ab6b4e5939e`。

| split | prompt | n | rejected recall | accuracy | accepted false-reject | balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | v1 / v12 | 94 | 83.3% | 74.5% | 27.6% | 77.9% |
| train | v2 | 94 | 77.8% | 79.8% | 19.7% | 79.0% |
| validation | v1 / v12 | 45 | **70.0%** | 73.3% | 25.7% | **72.1%** |
| validation | v2 | 45 | 60.0% | **77.8%** | **17.1%** | 71.4% |

本项目把坏样本召回放在第一优先级，同时把误杀看作泛化不足的风险信号。因此 validation 上冻结 v1：它多抓到 1/10 个坏样本，balanced accuracy 也略高；代价是多误杀 3/35 个好样本。test 结果出来后未再修改 prompt。

## 8. Gemini 最终 blind test

91/91 请求成功，解析失败为 0。总体混淆矩阵为：22 个坏样本正确拒绝、4 个坏样本漏放、12 个好样本误杀、53 个好样本正确保留。

| 范围 | n（人工 rejected） | rejected recall | accuracy | accepted false-reject | reject precision | balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **总体** | **91（26）** | **84.6%** | **82.4%** | **18.5%** | **64.7%** | **83.1%** |
| remove | 59（21） | 90.5% | 86.4% | 15.8% | 76.0% | 87.3% |
| replace | 18（4） | 50.0% | 72.2% | 21.4% | 40.0% | 64.3% |
| add | 14（1） | 100.0% | 78.6% | 23.1% | 25.0% | 88.5% |

| 数据集 | n（人工 rejected） | rejected recall | accuracy | accepted false-reject |
| --- | ---: | ---: | ---: | ---: |
| GOKU | 31（16） | 93.8% | 87.1% | 20.0% |
| ReCo | 56（9） | 66.7% | 80.4% | 17.0% |
| EgoEdit | 4（1） | 100.0% | 75.0% | 33.3% |

最可靠的结论是 removal 已经具备较强筛坏能力，尤其 GOKU remove 为 8/8、GOKU 反向 add-remove 为 7/8。主要短板是 ReCo：generation side 未公开，只能按 `SRC→TARGET` 检查语义并对两侧查伪影；其 rejected recall 为 6/9。replace 只有 4 个坏样本，其中 2 个 ReCo replacement 被漏放，因此 50% 只能作为需要补充 replacement 坏例的告警，不能视为稳定估计。

本轮 4 个漏放为 2 条 ReCo replace、1 条 ReCo remove、1 条反向 GOKU add-remove；后者的人工拒绝理由是 content preservation。12 个误杀中包含模型认为存在移除后模糊/残留、低质 3D 融合、姿态或非编辑内容变化的样本，说明误杀不全是随机噪声，也确实反映了当前 rubric 比部分人工接受标准更严格。

## 9. 费用与产物

最终成功行记录的 train/validation/test 费用合计 `$2.1068`：train 两版 `$1.0762`、validation 两版 `$0.5110`、最终 test `$0.5196`。另有一次已计费但响应截断的 validation 请求 `$0.0089`，重试成功后才写入最终成绩，因此本轮正式评测实际已知费用至少 `$2.1157`。所有计分行均由 Google AI Studio provider 返回；train、validation 最终文件和 test 均无解析失败。费用不含少量前置 smoke；两次错误环境尝试都在本地解码阶段失败，API 调用和费用均为 0。

主要结果文件位于 `artifacts/systematic_binary_benchmark_v1/gemini_flash_eval/`：

- `train_comparison.json` 与 `validation_comparison.json`：prompt 选择证据；
- `prompt_selection.json`：在运行 test 前冻结的入选版本、SHA256 和选择理由；
- `test_selected_v1_scores.csv`：91 条逐样本回答、理由与费用；
- `test_selected_v1_evaluation.json`：总体及 dataset/branch/task/provenance 分组指标；
- `test_selected_v1_summary.json`：调用完整性、token 和费用统计。

当前结论只针对已完成的 91 条明确 systematic 子集。若继续完成剩余人工审核，应把新增明确标签作为新的 test 扩展单独报告；不能把 75 条人工 uncertain 按负面备注并入 rejected，也不应据新增 test 错误回头修改本次冻结 prompt。
