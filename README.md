# Video Editing 数据与模型实验

当前入口：

- **已完成 CoinVE＋TripVVT 本地 11K 数据与使用说明**：[`dataset_research/coinve_tripvvt_11k_20261108/README.md`](./dataset_research/coinve_tripvvt_11k_20261108/README.md)
- **审核包规范（默认无评分、连续滚动）**：[`DATASET_AUDIT_GUIDE.md`](./DATASET_AUDIT_GUIDE.md)

- **当前任务：通用编辑数据整合与 Lance 微调**：[`lance_general_edit_finetune/README.md`](./lance_general_edit_finetune/README.md)
- **当前 4K 数据池与 Gemini-only 筛选摘要**：[`lance_general_edit_finetune/3_lance_lora_finetuning/CURRENT_DATASET_STATUS.md`](./lance_general_edit_finetune/3_lance_lora_finetuning/CURRENT_DATASET_STATUS.md)
- 历史 removal 500-step 消融（旧 15 条 base 依赖已删除，仅保留归档）：[`model_evaluation/lance_remove_ablation/README.md`](./model_evaluation/lance_remove_ablation/README.md)
- **Lance 基模 OmniEdit Spatial 240 评测**：[`model_evaluation/lance_base/README.md`](./model_evaluation/lance_base/README.md)
- 开源数据集调查：[`dataset_research/README.md`](./dataset_research/README.md)
- 数据集推荐排序：[`dataset_research/dataset_recommendations.md`](./dataset_research/dataset_recommendations.md)

目录约定：数据集调查统一放在 `dataset_research/`，通用编辑整合与训练工作放在 `lance_general_edit_finetune/`，模型部署与评测统一放在 `model_evaluation/`。视频、压缩包、模型、输出和大型官方 metadata 是本地可重建产物，不进入 Git。
