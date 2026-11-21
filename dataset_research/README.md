# 开源视频编辑数据集调查

本目录统一保存开源数据集的可获取性核查、随机样本、人工审核表、报告和复现脚本。

- `dataset_recommendations.md`：按当前项目推荐程度排列的逐数据集简介。
- `remote_zip_extract.py`：远程 ZIP 精确抽取的公共工具。
- `prepare_vvt_samples.py`：ViViD / Eevee 的公共样本整理工具。
- [`ref_edit_review_20261030/`](./ref_edit_review_20261030/README.md)：25 组换衣三元组与 25 组独立动作参考输入对的离线质量审核包；动作部分无编辑后 GT。
- [`recent_motion_review_20261030/`](./recent_motion_review_20261030/README.md)：替换旧动作选材的近期数据调研；X-Dance、RealisDance-Val、MoveBench 人物子集和 Wan 官方示例，共 50 条连续滚动浏览，无评分栏。
- [`datamake_v1_review_20261030/`](./datamake_v1_review_20261030/README.md)：师兄新版合成数据 80 条均衡审核，覆盖 8 个任务组，含全部参考素材、双语 prompt、多轮上下文及 prompt 风格报告，无评分栏。
- `*_research/`：每个数据集各自的调研材料，目录内部结构保持不变。

当前数据集目录：CoinVE、Ditto、Eevee、EgoEdit、InsAVE、InsViE、OpenVE、ReCo、Señorita、TripVVT 和 ViViD。GOKU 本地调研素材已随 Lance/GOKU 退役清理，最小记录见上级退役档案。

项目任务定义、协作规则和审核手册仍放在上一级 `video_collection/`，作为所有数据集共用的规范。
当前正式数据入口为 [`coinve_tripvvt_11k_20261108/`](./coinve_tripvvt_11k_20261108/README.md)；其所有 canonical 媒体路径均已内部化到本目录的 `media/`。
