# 最终实验结果：冻结300与V4自建116

更新：2026-11-16。仅摘录通过最终验收的全量结果；第二套数据已确认是116条。所有均值按视频等权，样本数为每模型、每指标的有效数量。

SSIM／PSNR／LPIPS仅与GT比较（↑／↑／↓，PSNR单位dB）；VLM的IF为编辑达成、CP为非目标内容保持，均1–5分、越高越好。VBench六项均展示为×100、越高越好；Temporal Flickering使用全量原始分数，受正常运动影响，不采用静态筛选子集，不计算Total Score。

## 1. 冻结300：六模型

UE200＋CoinVE75＋TripVVT25；下列各项每模型 **n=300**。

### GT保真度

| 模型 | SSIM ↑ | PSNR ↑（dB） | LPIPS ↓ |
|---|---:|---:|---:|
| H3 PDD8 | 0.51807 | 14.697 | 0.43986 |
| VACE LightX2V4 | 0.63684 | 17.846 | 0.31072 |
| LTX2.5 Base（distilled） | 0.44589 | 12.803 | 0.52000 |
| A849（dev） | 0.61716 | 18.010 | 0.30527 |
| 仅合成 step600 | 0.60154 | 17.674 | 0.33586 |
| 仅真实 step232 | 0.46730 | 13.467 | 0.46651 |

### VLM

| 模型 | IF ↑ | CP ↑ |
|---|---:|---:|
| H3 PDD8 | 3.853 | 3.673 |
| VACE LightX2V4 | 3.013 | 4.193 |
| LTX2.5 Base（distilled） | 2.213 | 2.477 |
| A849（dev） | 4.117 | 4.183 |
| 仅合成 step600 | 3.810 | 3.777 |
| 仅真实 step232 | 2.940 | 2.967 |

### VBench（全部×100）

| 模型 | 主体一致性 ↑ | 背景一致性 ↑ | 美学 ↑ | 成像质量 ↑ | 原始闪烁 ↑ | 运动平滑度 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| H3 PDD8 | 91.430 | 92.831 | 61.928 | 72.738 | 96.325 | 98.383 |
| VACE LightX2V4 | 94.246 | 94.998 | 60.146 | 72.282 | 96.920 | 98.303 |
| LTX2.5 Base（distilled） | 89.789 | 91.925 | 58.806 | 70.864 | 97.178 | 98.408 |
| A849（dev） | 94.046 | 94.615 | 58.527 | 72.302 | 96.951 | 98.389 |
| 仅合成 step600 | 93.758 | 94.508 | 59.252 | 72.752 | 97.135 | 98.473 |
| 仅真实 step232 | 94.812 | 94.718 | 59.189 | 72.864 | 97.312 | 98.553 |

模型口径：H3为PDD8；VACE为LightX2V4四步。LTX基座及两个消融模型使用LTX2.5 distilled＋V7、8步；A849使用dev＋V7、40步。两个消融分别加载仅合成step600、仅真实step232 LoRA；232／600是checkpoint步数，不是训练样本数。推理基座／步数及训练步数不同，不能把全部差异归因于训练数据。

## 2. V4自建116：四模型

发型24＋发色24＋背景24＋TripVVT24＋通用编辑20；共同评分内容为0–4秒。采用最终 **all116**，包含保留92条与新增24条，下列各项每模型 **n=116**。

### VLM

| 模型 | IF ↑ | CP ↑ |
|---|---:|---:|
| H3 PDD8 | 3.922 | 4.474 |
| VACE | 3.224 | 4.474 |
| LTX2.5 Base（dev） | 1.431 | 2.940 |
| A849（dev） | 4.129 | 4.431 |

### VBench（全部×100）

| 模型 | 主体一致性 ↑ | 背景一致性 ↑ | 美学 ↑ | 成像质量 ↑ | 原始闪烁 ↑ | 运动平滑度 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| H3 PDD8 | 94.548 | 94.364 | 56.544 | 72.698 | 96.577 | 98.511 |
| VACE | 94.861 | 95.035 | 54.878 | 70.596 | 96.827 | 98.418 |
| LTX2.5 Base（dev） | 90.601 | 92.209 | 52.593 | 67.250 | 96.634 | 98.231 |
| A849（dev） | 95.263 | 95.396 | 54.467 | 71.384 | 96.754 | 98.496 |

模型口径：LTX基座与A849均为dev V7、40步，H3为PDD8；VACE历史输出包含100条40步和16条蒸馏4步。116集部分TripVVT经过看输出后选样，存在选择偏差。两套测评集的输入和推理配置不同，分别展示，不合并排名。

## 3. VLM最终版本与解读

正式Judge为OpenRouter转发的 `deepseek/deepseek-v4.1-flash-1017`（Wafer／mxfp4），采用用户锁定的最终Prompt及镜头CP v3。Judge不看GT；CP不看参考像素。排除早期多裁判试评、旧镜头CP及检查点快照，重复评分不混入总体均值。

上表是含校准样本的全量描述性结果：300集含72条校准任务，116集含28条。未参与本次Prompt调优的228／88条分表见[未参与调优汇总](ds_full_eval_20261115/report/full/not_used_for_prompt_tuning_summary.csv)；校准分表见[校准汇总](ds_full_eval_20261115/report/full/calibration_summary.csv)。GT保真度、VLM与VBench衡量不同方面，均不能替代人工编辑效果审核。

## 4. 最终数据与验收入口

以下均为本次表格引用的最终文件；若随ZIP交付，解压后可按相对链接查看。

| 数据 | 最终汇总 | 逐视频／逐项数据 | 最终验收 |
|---|---|---|---|
| 300：原四模型GT | [汇总](frozen300_gt_metrics_20261115/report/summary.csv) | [明细](frozen300_gt_metrics_20261115/report/videos.csv) | [验收通过](frozen300_gt_metrics_20261115/metadata/completion_audit.json) |
| 300：原四模型VBench | [汇总](frozen300_vbench_20261115/report/summary.csv) | [明细](frozen300_vbench_20261115/report/per_video_metrics.csv) | [验收通过](frozen300_vbench_20261115/metadata/final_validation.json) |
| 300：仅合成／仅真实GT＋VBench | [汇总](frozen300_data_ablation_20261115/report/summary.csv) | [明细](frozen300_data_ablation_20261115/report/per_video_metrics.csv) | [验收通过](frozen300_data_ablation_20261115/metadata/final_validation.json) |
| 300＋116：正式VLM | [汇总](ds_full_eval_20261115/report/full/summary.csv) | [明细](ds_full_eval_20261115/report/full/scores.csv) | [验收通过](ds_full_eval_20261115/metadata/final_completion_audit.json) |
| 116：V4全量VBench | [汇总](a849_external_vbench_v4_116/metadata/group_metrics.json) | [明细](a849_external_vbench_v4_116/metadata/per_video_metrics.csv) | [验收通过](a849_external_vbench_v4_116/metadata/final_validation.json) |

98个展示均值均已对照最终逐视频／逐项数据复核；本次只整理结果，没有重新推理或评分。
