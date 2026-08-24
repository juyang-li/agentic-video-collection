# CoinVE-200K 随机 pair 抽查

当前保留一套最终结果：从官方完整 metadata 的 200,916 个样本中随机抽取 15 个 source–instructions–target pair，并整理为可直接观察的 H.264 MP4。

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
