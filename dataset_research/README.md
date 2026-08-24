# 开源视频编辑数据集调查

本目录统一保存开源数据集的可获取性核查、随机样本、人工审核表、报告和复现脚本。

- `dataset_scout_report.md`：所有候选的总览、筛选口径与下一步顺序。
- `dataset_recommendations.md`：按当前项目推荐程度排列的逐数据集简介。
- `dataset_integration_pilot_plan.md`：CoinVE + GOKU 起步整合、筛选、训练和 benchmark 计划，附新 session 实施提示词。
- `remote_zip_extract.py`：远程 ZIP 精确抽取的公共工具。
- `prepare_vvt_samples.py`：ViViD / Eevee 的公共样本整理工具。
- `*_research/`：每个数据集各自的调研材料，目录内部结构保持不变。

当前数据集目录：CoinVE、Ditto、Eevee、EgoEdit、GOKU、InsAVE、InsViE、OpenVE、ReCo、Señorita、TripVVT 和 ViViD。

项目任务定义、协作规则和审核手册仍放在上一级 `video_collection/`，作为所有数据集共用的规范。
