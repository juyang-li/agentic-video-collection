# OpenVE-3M 随机 pair 抽查

当前保留两批最终结果：从 `local_change` 固定 2 GiB 归档前缀的 39 条可完整提取记录中，先随机抽 15 条，再从剩余 24 条中随机抽 15 条。两批均整理为可直接观察的 H.264 MP4，全程未按人物或关键词筛选。

## 查看

- `examples/comparison/`：左右对照 H.264 MP4；同名 `.txt` 是 instruction。
- `examples/contact_sheets/`：每对均匀取 6 帧，左 source、右 target。
- `openve_random_pairs_15_h264.zip`：15 个 comparison、instruction、manifest 和 audit，可下载到本地查看。
- `batch2/`：第二批 15 条的完整独立结果和下载 ZIP。

两批抽取阶段使用的 60 个 source/target 角色 MP4 已在 comparison 完成并校验后删除，以免重复保存；需要时可重新准备相同归档前缀并运行提取和准备脚本。

## 抽样口径

- revision：`Lewandofski/OpenVE-3M@d2c134f01c49d22efc2e0d9aefd798772f039a79`。
- population：2 GiB 固定压缩流前缀中，source 和 target 都已完整出现的 39 条 `local_change` metadata 记录。
- method：`random.Random(20260821).sample(range(39), 15)`，按 metadata 行无放回均匀抽样。
- batch 2：排除第一批 15 条 metadata 后，用 seed `20260822` 从剩余 24 条中无放回随机抽 15 条。
- metadata SHA256：`eaf9b18ab58ef2f91812a01ca5d5656d2b018b72670ed02d4a36c6366e6765bb`。
- prefix SHA256：`12cf8d8d7b924250b30cb2c6526373ea48e74dcca3c6d343f45194c73585ee1b`。
- 该结果是固定归档前缀子总体的随机样本，不是完整 `local_change` 或 OpenVE-3M 的均匀随机样本。

## 文件

- `sample_random_pairs.py`：扫描固定前缀并复现抽样。
- `extract_random_pairs.py`：从同一前缀提取命中 MP4。
- `prepare_random_pairs.py`：生成 H.264 comparison、contact sheet、inventory 和 audit。
- `prefix_members.txt`、`downloadable_population.csv`：固定前缀及抽样总体证据。
- `random_pairs.csv`、`batch2/random_pairs.csv`：两批样本 manifest。
- `openve_inventory.csv`：客观技术信息。
- `openve_audit.csv`：人工审核；`0/1/2` 表示不通过/通过/存疑。
- `openve_report.md`：阶段结论。

```bash
python sample_random_pairs.py --prefix /tmp/openve_local_change_prefix_2g.tar.gz
python extract_random_pairs.py --prefix /tmp/openve_local_change_prefix_2g.tar.gz
python prepare_random_pairs.py
```

前两个命令需要先按 `openve_pull_plan.md` 准备相同字节前缀。prepare 默认不覆盖现有 audit；只有明确传入 `--reset-audit` 才会重建。
