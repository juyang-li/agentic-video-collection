# 师兄 gtfresh 训练数据抽样审核

本目录抽查师兄 `gtfresh-v1 step1150` 实际绑定的冻结训练数据：

`/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/10_research/model_experiments/lance/outputs/training_data/datamake_coinve4k_mixed_r5x3_seed20260911_v1`

审核包以 UE 合成数据为主，并保留少量 CoinVE 真实配对样本作对照：

- 20 条 UE 合成完整多轮链，共 81 个原子编辑轮次；
- 6 条 CoinVE 真实单轮复合编辑，Add/Remove/Replace 来源分支各 2 条；
- 合成链覆盖 `identity_id`、`outfit_id`、`action_id`、`camera_id` 四个实际训练分支；
- 同时覆盖 3/4/5 轮、progressive/freeform、重复轴、连续同轴、rollback 和 returns-to-start。

这里的“四分支”指实际进入该 checkpoint 训练集的四个 UE edit axis，不把尚未进入这份训练数据的 E1–E11 规划项写成已覆盖。

最终审核包解压后打开 `index.html`。所有审核字段默认是 `1`（通过）；失败改为 `0`，存疑改为 `2`，并填写备注。页面支持 CSV/JSON 导入导出且完全离线。

