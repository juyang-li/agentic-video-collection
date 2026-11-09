# DataMake dataset_package_v1 均衡审核

来源：`/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/00_engine/outputs/dataset_package_v1`，只读使用。

80 个焦点编辑，8 个任务组各 10 条：身份、单件服装、整套服装、动作、场景、发型、发色、镜头。64 条独立单轮 + 16 条多轮焦点轮，包含实际分叉与完整链上下文；男女各 40，横竖 41/39，train/test 56/24。

审核页含 source、target、当轮所有 ref image/ref video、完整原始中英文 prompt。支持连续下拉、分组跳转、从头同步播放，无评分控件。多轮上下文可展开，保留历史位置与真实分叉输入，不默认沿用上一轮。

Prompt 风格和全量文本统计见 [REPORT.md](./REPORT.md)。数据集总体 31,702 条单轮、14,400 条多轮链；本次读取全部业务 JSON 统计。原数据的7个编辑轴均覆盖，服装拆成两组，不虚构旧规划中尚无样本的编辑轴。

复现入口：

1. `inventory.py` 读取全量元数据缓存并统计任务分布。
2. `sample.py` 固定 seed=20261030，按任务/来源分层、兼顾属性多样性抽取，不依画面质量筛选。
3. `build.py` 核对所选源文件 SHA256，复用已完成的 CRF18 浏览副本，其余用原始码流无损封装；复制原 PNG、保留原 JSON，生成离线页面。原尺寸/帧率/时长均保留，逐文件策略写入 manifest。
4. `test_browser.py` 用现有 Playwright/Chrome 验证完整 prompt、80 条连续布局、媒体加载、代表性同步播放和多轮展开。
5. `build.py --finalize` 生成本地校验清单和 ZIP；`deliver.py` 沿用项目百度上传器，核对上传完成及远端列表，不整包回读。

媒体工具 Python：`/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python`。编码已切换至用户现有 allocation 179941 的16个 CPU，8个并行编码进程，不使用 GPU。源数据未改变，先前推理 bench 未修改。

交付 ZIP 名称：`datamake_v1_balanced80_scroll_20261030.zip`；远端目录 `/apps/bypy/lance_review/`。完成状态与精确大小/hash 以 `metadata/delivery_receipt.json` 为准。
