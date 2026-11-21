# CoinVE-200K 数据调研与审核

## 当前：30 对时长/帧数倍数阈值审核（2026-11-08）

[长度倍数审核包](./coinve_duration_ratio30_20261108/coinve_duration_ratio30_20261108.zip)按 1×、1.1×、1.25×、1.5×、2×五档各 6 对展示。非等长组各含阈值内外 3 对，每档均为 3 个两任务和 3 个三任务样本。全部无 Stylize、两侧至少 4 秒，采用本地已有原始媒体，保留完整分辨率、帧率、时长与全部 prompt。联动仅比较共同区间，较长 source 的无对应尾段通过独立完整视频链接查看。

倍数定义为帧数比、时长比的较大值，均按较长/较短计算。统计基于本地 4,240 对已探测数据中满足前述基础条件的 2,693 对：严格 1/1.1/1.25/1.5/2 倍分别筛掉 66.91% / 49.50% / 39.73% / 28.52% / 15.60%。严格 1×会计入细小帧率记录差异；若按等帧数、容许一帧以内时差定义，筛除率为 57.00%。这不是 20 万全库的精确比例；旧候选曾按文件名帧跨度≥151预筛，可能偏向较长片段。

201 帧在 24/25/30 fps 下分别约为 8.375 / 8.040 / 6.700 秒。完整统计及口径见 [REPORT.md](./coinve_duration_ratio30_20261108/coinve_duration_ratio30_20261108/REPORT.md)，包内另有完整 2,693 对测量 CSV 和阈值计数 CSV，可复算。生产入口为 `build_duration_review30.py`，默认人工标签为 1、reviewed=0，不算人工通过；此包用于选择门槛，未据此筛除正式数据。

30 对的 60 个视频均已完整解码，并通过离线 Chrome 的逐对播放、分组跳转、prompt/比值、标注导入导出检查。已上传 `/apps/bypy/lance_review/coinve_duration_ratio30_20261108.zip`（433,868,756 bytes）及 SHA256 sidecar，均核对远端大小，回执见该批次 `baidu_delivery_receipt.json`。统计所用历史探测时长与本次容器读数分列保存，以免严格 1× 比较混用微小测量差异。

## 当前：扩至 10k 前的 50 对任务均衡摸底（2026-11-08）

本轮为保留旧 CoinVE-4K、后续新增 6k 做准备，先交付 [50 对离线审核包](./coinve_balanced50_20261108/coinve_balanced50_20261108.zip)。人物局部属性、主体整体、物体、背景、风格化五个展示组各 10 对；保留全部复合指令，排除旧 4k 共享媒体。固定 seed `20261108`，按指令分类后在组内稳定抽取，仅因技术不合格补样，不按编辑效果换样。

最终包含 141 条子任务；每对 2/3/4/5 条指令分别为 21/18/10/1 对，18 对含 Stylize。详细分类、条数和逐条指令见 [TASK_REPORT.md](./coinve_balanced50_20261108/coinve_balanced50_20261108/TASK_REPORT.md)。均衡展示的占比不能外推全库；本批尚未计作正式新增训练池，后续质量门槛与 6k 配比等待审核后确定。

完整解压 ZIP 后打开 `index.html`，支持同步播放、备注、CSV/JSON 导入导出。100 个浏览器视频已完整解码，并通过离线 Chrome 验证。审核默认 `human_decision=1`、`reviewed=0`，不代表人工通过；模型 judge 调用为 0。原始视频另存本轮外层 `originals/`，来源、原始哈希与浏览器副本映射随清单保留。数据交付暂不绑定 Lance 或 LTX2.5 训练格式。

生产脚本为 `build_balanced_review.py`，抽样清单、技术补样记录及上传回执保存在 `coinve_balanced50_20261108/`；远端交付目录为 `/apps/bypy/lance_review/`。

## 历史：50 对直接入池审核（2026-09-06）

为判断 CoinVE 是否可以跳过模型质量筛选，新增固定的 50 对人工审核清单：从不含 `Stylize` 的 145,483 条中排除此前看过的 10 条，在剩余 145,473 条中以 seed `20260906` 无放回均匀随机抽样。未按人物、画质、时长或编辑效果筛选，未根据内容换样。2/3/4/5 个任务分别为 23/23/3/1 对。

本轮目录为 [`coinve_nostylize_random50_20260906/`](./coinve_nostylize_random50_20260906/)。冻结来源见 `sample_manifest.csv` 和 `sampling_summary.json`；审核包为同名 ZIP，解压后打开 `index.html`。采用黑色单 pair 页面，左右同步、完整子指令、可跳过、备注和 CSV/JSON 导入导出。审核项预填 `1`，但 `reviewed=0` 不计作人工通过；结论等待用户审核。

2026-09-06 已上传百度网盘：`/apps/bypy/lance_review/coinve_nostylize_random50_20260906.zip`，503,419,864 bytes；同名 `.sha256.json` 已一并上传。已核对两文件的远端 ID 与大小，交付记录见本轮目录的 `baidu_delivery_receipt.json`。

`build_random_review.py` 负责固定抽样和按 Range 获取媒体，`package_random_review.py` 负责保留完整分辨率/帧率/时长的浏览器副本及打包，`verify_random_review.py` 验证真实离线播放和标注往返。完成回执见 ZIP 同名 `.sha256.json`；浏览器检查和逐视频解码记录随包保留。

## 历史 15 对全库随机抽查

从官方完整 metadata 的 200,916 个样本中随机抽取 15 个 source–instructions–target pair，并整理为可直接观察的 H.264 MP4。历史 manifest、人工审核和视频均保持不变。

## 查看

- `examples/comparison/`：左右对照 H.264 MP4；同名 `.txt` 是编号 instruction。
- `examples/contact_sheets/`：每对均匀取 6 帧，左 source、右 target。
- `coinve_random_pairs_15_h264.zip`：15 个 comparison、instruction 和 manifest，可下载到本地查看。

抽取阶段使用的 30 个独立 source/target MP4 已在 comparison 完成并校验后删除，以免重复保存；需要时可重新运行下载和准备脚本。

## 抽样口径

- population：官方 metadata 的 200,916 行。
- method：`random.Random(20260821).sample(range(200916), 15)`，无放回均匀随机抽样。
- metadata SHA256：`d1f0b643d2c5cc1cfe855351d2642fb1b6c0cb403f5301002111ac8c2246a5a6`。
- manifest：`random_pairs.csv`。

## 文件

- `sample_random_pairs.py`：复现抽样。
- `download_random_pairs.py`：利用未压缩 tar 的 HTTP Range，只提取命中 MP4。
- `prepare_random_pairs.py`：生成 H.264 comparison、contact sheet、inventory 和 audit。
- `coinve_inventory.csv`：客观技术信息。
- `coinve_audit.csv`：人工审核；`0/1/2` 表示不通过/通过/存疑。
- `coinve_report.md`：阶段结论。

```bash
python sample_random_pairs.py
python download_random_pairs.py --workers 6
python prepare_random_pairs.py
```

prepare 默认不覆盖现有 audit；只有明确传入 `--reset-audit` 才会重建。
