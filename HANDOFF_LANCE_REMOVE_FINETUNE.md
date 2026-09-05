# Lance removal pilot（已归档）

> 2026-09-05：旧 `lance_remove_pilot/` 训练目录、checkpoint、媒体和 runtime 副本已删除，不再作为当前执行入口。删除前的轻量 README、脚本、配置和 manifest 保存在 Git commit `69735dc`；未进入 Git 的大文件不可恢复。

该 pilot 的历史结论是：两个 500-step 模型均未稳定提升 removal，且出现非 removal 能力与时序退化，因此没有扩到 1K。报告及已有评测产物保留在 [`model_evaluation/lance_remove_ablation/README.md`](./model_evaluation/lance_remove_ablation/README.md)。

当前唯一执行入口为 [`lance_general_edit_finetune/README.md`](./lance_general_edit_finetune/README.md)。
