# OpenVE-3M 第二批随机 pair 抽查

从固定 2 GiB 前缀总体中排除第一批 15 条 metadata 后，再从剩余 24 条中随机抽取 15 条。没有按人物、关键词或视觉内容筛选。

## 查看

- `examples/comparison/`：左右对照 H.264 MP4；同名 `.txt` 是 instruction。
- `examples/contact_sheets/`：每对均匀取 6 帧，左 source、右 target。
- `openve_random_pairs_batch2_15_h264.zip`：可下载到本地查看的第二批成品。

抽取阶段使用的 30 个 source/target 角色 MP4 已在 comparison 完成并校验后删除，以免重复保存。

## 抽样口径

- 完整固定前缀总体：39 条。
- eligible population：排除第一批 15 条 metadata 后的 24 条。
- method：`random.Random(20260822).sample(eligible_indices, 15)`。
- manifest：`random_pairs.csv`。
- 该结果未排除与第一批方向相反的相同视频 pair；重复情况见 `../openve_report.md`。

## 文件

- `openve_inventory.csv`：客观技术信息。
- `openve_audit.csv`：人工审核；`0/1/2` 表示不通过/通过/存疑。
- `../sample_random_pairs.py`、`../extract_random_pairs.py`、`../prepare_random_pairs.py`：复现脚本。

```bash
python ../sample_random_pairs.py --prefix /tmp/openve_local_change_prefix_2g.tar.gz --seed 20260822 --output-dir . --exclude-manifest ../random_pairs.csv
python ../extract_random_pairs.py --prefix /tmp/openve_local_change_prefix_2g.tar.gz --manifest random_pairs.csv
python ../prepare_random_pairs.py --manifest random_pairs.csv
```

prepare 默认不覆盖现有 audit；只有明确传入 `--reset-audit` 才会重建。
