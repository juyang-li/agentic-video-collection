# GOKU-2M `remove` 随机抽查

从官方 `remove` 的 32 个视频分卷中按 seed `20260823` 随机选中 `part32`，再从该卷 5,438 个 metadata、source、target 齐全的 case 中无放回随机抽 15 组。没有按人物、关键词或质量筛选。

## 查看

- `examples/comparison/`：15 个 H.264 左右对照 MP4；同名 `.txt` 是 instruction。
- `examples/contact_sheets/`：每对均匀取 6 帧。
- `goku_remove_random_pairs_15_h264.zip`：方便下载到本地观察的最终包。
- `goku_remove_report.md`：抽查结论。

30 个独立 source/target 角色 MP4 只用于生成 comparison 和校验，最终 ZIP 通过检查后已删除；需要时可用同一 revision、seed 和脚本复现。21.4 GB 临时分卷也未长期保留。

固定 revision：`bigfacing/GOKU-2M@f289f4b245648dc57db04afaaeb68659440c0fa6`。随机选中的 `remove.videos.part32.tar` 为 21,421,004,800 bytes，SHA256 为 `67a6cf4c...0906`；metadata SHA256 为 `8f66847b...6e0c`。

这是随机分卷内随机样本，不是完整 `remove` 的全量均匀随机样本。
