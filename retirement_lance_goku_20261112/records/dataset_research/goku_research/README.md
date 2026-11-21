# GOKU-2M 随机 pair 抽查

当前保留一套最终结果：在官方 `swap_alter` 的 32 个视频分卷中随机选中 `part05`，再从该卷 8,260 个完整 case 中随机抽取 15 个 source–instruction–target pair。样本已整理为可直接观察的 H.264 左右对照 MP4，全程未按人物、关键词或质量筛选。

## 查看

- `examples/comparison/`：左右对照 H.264 MP4；同名 `.txt` 是 instruction。
- `examples/contact_sheets/`：每对均匀取 6 帧，左 source、右 target。
- `goku_random_pairs_15_h264.zip`：15 个 comparison、instruction、manifest 和 audit，可下载到本地查看。

抽取阶段使用的 30 个独立 source/target MP4 已在 comparison 完成并校验后删除，以免重复保存。需要重建时重新取得官方分卷，再运行下方脚本。

## 抽样口径

- repo/revision：`bigfacing/GOKU-2M@f289f4b245648dc57db04afaaeb68659440c0fa6`。
- shard sampling：`random.Random(20260821).choice(part01...part32)`，结果为 `swap_alter.videos.part05.tar`。
- population：该卷内 source、edited、instruction 和有效 JSON 均齐全的 8,260 个 case。
- pair sampling：`random.Random(20260821).sample(range(8260), 15)`，无放回均匀随机抽样。
- metadata SHA256：`0e2acb13ce472578435851ed854a859c21906472a8a6752a7eabd11b582fdb70`。
- video shard SHA256：`42360d430b499ef69d378da6c94af07d6922a8f137be3be50c2096b9c6b40dae`。
- manifest：`random_pairs.csv`。

这是**随机分卷内随机样本**，不是完整 `swap_alter` 或 GOKU-2M 的均匀随机样本。

## 文件

- `sample_random_pairs.py`：扫描分卷、连接 annotation 并复现抽样。
- `extract_random_pairs.py`：从未压缩视频 tar 中只提取命中的 30 个角色文件。
- `prepare_random_pairs.py`：生成 H.264 comparison、contact sheet、inventory 和 audit。
- `shard_members.txt`、`shard_population.csv`：固定分卷总体证据。
- `goku_inventory.csv`：客观技术信息。
- `goku_audit.csv`：人工审核；`0/1/2` 表示不通过/通过/存疑。
- `goku_report.md`：阶段结论。

```bash
python sample_random_pairs.py
python extract_random_pairs.py
python prepare_random_pairs.py
```

前三步需要相同 revision 的 metadata tar 和 `part05` 视频 tar。prepare 默认不覆盖现有 audit；只有明确传入 `--reset-audit` 才会重建。
