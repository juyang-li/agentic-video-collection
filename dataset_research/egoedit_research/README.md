# EgoEditData 随机 replacement pair 抽查

当前结果是从已授权发布的完整 EgoEditData 编辑表中均匀随机抽取的 15 组 source–instruction–target pair。样本没有按人物、关键词或画质筛选，已整理为可直接观察的 H.264 左右对照 MP4。

## 查看

- `examples/comparison/`：15 个左右对照 H.264 MP4；同名 `.txt` 是 instruction。
- `examples/contact_sheets/`：每组均匀取 6 帧，左 source、右 target。
- `egoedit_replace_random_pairs_15_h264.zip`：可下载到本地查看的精简包。
- `egoedit_audit.csv`：待人工填写的审核表，`0/1/2` 表示不通过/通过/存疑。

本轮 30 个独立 source/target MP4 已在 comparison 和 ZIP 完整性验证后删除，避免重复保存约 154 MB；需要追查时可按下方脚本和固定 manifest 精确重新获取。

## 抽样口径

- 数据镜像与 revision：`liguang0115/EgoEdit@a9bd3faafe51a581b359acd7c6ad5c432508ff60`。
- `egoeditdata_data_table.csv` 有 37,716 行，编辑表有 93,644 行；93,644 条编辑均能连接到 source 和 target URL。
- population：全部 93,644 条可连接、instruction 非空的编辑边。
- sampling：`random.Random(20260901).sample(range(93644), 15)`，无放回均匀随机。
- 结果覆盖 15 个不同 scene key、30 个不同 MP4；没有做人物或质量过滤。

发布 instruction 没有形成可独立抽取的纯 add/remove 子总体。仅有的 15 条 `Remove...` 开头指令全部同时要求 `replace with...`，因此本轮只按 object replacement 抽样，不伪造 remove 组。

## 文件

- `random_replace_pairs.csv`：抽样 manifest 与完整来源字段。
- `egoedit_inventory.csv`：视频规格、SHA256 和派生文件位置。
- `egoedit_audit.csv`：默认值审核表，等待人工观看后修改。
- `egoedit_report.md`：阶段结论。
- `sample_random_pairs.py`、`index_remote_zips.py`、`download_random_pairs.py`、`prepare_random_pairs.py`：复现脚本。
- `zip_members.csv`、`zip_indexes/`：8 个远程 ZIP 的成员索引；可精确拉取命中视频，不必下载约 245 GB 全量。

```bash
python sample_random_pairs.py
python index_remote_zips.py
python download_random_pairs.py
python prepare_random_pairs.py
```

前两项下载操作需要已登录且具有数据集权限的 Hugging Face 账号。`prepare_random_pairs.py` 默认不覆盖现有 audit；只有显式传入 `--reset-audit` 才会重建。
