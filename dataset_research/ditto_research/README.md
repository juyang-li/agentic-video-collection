# Ditto 随机 pair 抽查

当前保留一套最终结果：从官方 metadata 的 `local/0130` 子总体中随机抽取 15 个 source–instruction–target pair，并整理为可直接观察的 H.264 MP4。

## 查看

- `random_pair_examples/comparison/`：左右对照 H.264 MP4；同名 `.txt` 是 instruction。
- `random_pair_examples/contact_sheets/`：每对均匀取 6 帧，左 source、右 target。
- `ditto_random_pairs_15_h264.zip`：15 个 comparison、instruction 和 manifest，可下载到本地查看。

抽取阶段使用的 30 个独立 source/target MP4 已在 comparison 完成并校验后删除，以免重复保存；需要时可重新运行下载和准备脚本。

## 抽样口径

- population：metadata 中 source 和 target 都位于 `local/0130/` 的 472 个 triplet。
- method：`random.Random(20260821).sample(range(472), 15)`，无放回均匀随机抽样。
- manifest：`random_pairs.csv`。
- 该结果对 `local/0130` 是随机的，但不是 Ditto 全量 local 的均匀随机样本。

## 文件

- `sample_random_pairs.py`：复现抽样。
- `download_random_pairs.py`：从 split gzip 起点顺序扫描，只保存命中的 pair。
- `prepare_random_pairs.py`：生成 H.264 comparison、contact sheet、inventory 和 audit。
- `ditto_inventory.csv`：客观技术信息。
- `ditto_audit.csv`：人工审核；`0/1/2` 表示不通过/通过/存疑。
- `ditto_report.md`：阶段结论。

```bash
python sample_random_pairs.py
python download_random_pairs.py
python prepare_random_pairs.py
```

prepare 默认不覆盖现有 audit；只有明确传入 `--reset-audit` 才会重建。
