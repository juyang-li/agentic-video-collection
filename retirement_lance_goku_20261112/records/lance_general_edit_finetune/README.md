# Lance 通用视频编辑微调

本目录是通用 single-step 视频编辑数据整合、分级筛选、抽样验证和 Lance 微调的当前工作入口。各数据集的原始调研与审核材料仍放在 `dataset_research/`，这里不重复保存数据集研究笔记。

## 目录结构

```text
lance_general_edit_finetune/
├── README.md
├── 0_metadata_filtering.md
├── 1_specialist_filtering/
├── 2_vlm_filtering/
└── 3_lance_lora_finetuning/
    ├── CURRENT_DATASET_STATUS.md
    └── README.md
```

- [`0_metadata_filtering.md`](./0_metadata_filtering.md)：第零级筛选；定义任务格，根据 metadata、官方分支、编辑方向和已有抽查决定哪些数据进入视频级筛选。
- [`1_specialist_filtering/README.md`](./1_specialist_filtering/README.md)：第一级筛选入口；汇总原四分支与 ReCo add/remove/replace 的人工审核，并定义通用质量、机器人/CG、人物/服装和 pair 检查的顺序。
- [`2_vlm_filtering/README.md`](./2_vlm_filtering/README.md)：第二级综合 pair judge 与系统人工基准；已知生成侧按 reference→generated，ReCo 等逐 pair lineage 未发布的数据按 `generation_side_unresolved` 路由。
- [`3_lance_lora_finetuning/CURRENT_DATASET_STATUS.md`](./3_lance_lora_finetuning/CURRENT_DATASET_STATUS.md)：**当前数据情况的权威入口**；记录 4K 冻结池、数据集/任务配比、排除项、Gemini-only 筛选和 OpenVid 决策。
- [`3_lance_lora_finetuning/README.md`](./3_lance_lora_finetuning/README.md)：第三阶段 LoRA 参数、固定 seed、训练执行与评测约定。

## 数据流

```text
官方 metadata / archive
        ↓
0_metadata_filtering
        ↓ 约 89 万 metadata 候选（已排除 InsAVE person remove）
1_specialist_filtering（单视频质量 + ReCo robot/CG）
        ↓
2_vlm_filtering（完整 pair 编辑成功判定）
        ↓
3_lance_lora_finetuning
        ↓
冻结 manifest → Lance Parquet → LoRA 微调 → 评测 → 扩量
```

当前状态（2026-09-05）：7,558 条 Stage-2 候选已统一由 Gemini 判定，得到 4,141 accepted、3,402 rejected 和 15 quarantine；已用固定 seed `20260904` 冻结严格配额的 4,000 对编辑池，split 为 train 3,601 / validation 195 / test 204。首轮 LoRA 只用编辑 train split，不混合 OpenVid T2V；详见 [`CURRENT_DATASET_STATUS.md`](./3_lance_lora_finetuning/CURRENT_DATASET_STATUS.md)。
