# InsAVE-80K 原子 add / remove 随机抽查

当前保留两套各 15 组的 H.264 画面对照：

- `insave_remove_random_pairs_15_h264.zip`：`part_01` 的 9,994 个完整 pair 中均匀随机抽 15 个正向 remove。
- `insave_add_random_pairs_15_h264.zip`：从同一分片的反向 add 中，先排除语音标记和显式音频关键词，再在 97 个候选中均匀随机抽 15 个。

ZIP 中的 `comparison/*.mp4` 是左右对照、无音轨、H.264/AVC、`yuv420p`、fast-start；同名 `.txt` 是 instruction。若 VS Code 不能直接播放，下载 ZIP 后用系统播放器打开。

## 抽样

- 官方 revision：`8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289`。
- 两个官方训练分片用 seed `20260826` 随机选中 `part_01.tar`；该卷实测 9,994 个完整 pair。
- remove：seed `20260826`，不做人物或质量过滤。
- add：seed `20260827`；过滤条件只定义“文字上不要求音频”的反向 add 子总体，不按人物或画质挑选。
- 抽查恰好 30/30 都是整个人物 add/remove，但这不能外推为完整数据集的精确人物比例。

## 文件

- `random_pairs.csv` / `random_add_pairs.csv`：两套 manifest。
- `insave_inventory.csv` / `insave_add_inventory.csv`：客观视频规格。
- `insave_audit.csv` / `insave_add_audit.csv`：人工画面审核。
- `insave_audio_inventory.csv` / `insave_add_audio_inventory.csv`：16 kHz mono PCM 客观对比。
- `insave_report.md`：结论与适用边界。
- `sample_random_pairs.py`、`sample_random_add_pairs.py`、`extract_random_pairs.py`、`prepare_random_pairs.py`、`prepare_random_add_pairs.py`：复现脚本。

独立 source/target MP4 在 comparison 与 ZIP 校验后删除，避免重复占空间；官方原始分片也只作临时获取，不留在项目中。
