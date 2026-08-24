# OpenVE-3M 抽查计划

> 状态：已完成。固定 2 GiB 前缀足以构造 39 条可下载总体；两批各随机抽取 15 条，成品核验通过，临时前缀已删除。

## 当前观察

- 当前 revision：`Lewandofski/OpenVE-3M@d2c134f01c49d22efc2e0d9aefd798772f039a79`；非 gated，许可为 CC-BY-NC-4.0。
- 论文/README 定义 8 类，但当前仓库实际只有 6 个 spatially-aligned 类别的 CSV 和视频：background change、global style、local add、local change、local remove、subtitles edit；camera/creative 的 CSV 与视频尚未出现在文件树中。
- 当前视频共 107 个分卷、约 2.03 TiB（约 2.23 TB）。其中 `local_change_new` 为 8 卷、151.47 GiB，前 7 卷各 20 GiB。
- 视频是一个 gzip tar 的连续字节切分：`.00` 有 gzip header，`.01` 没有，官方也要求 `cat *.tar.gz.* | tar -zxv`。因此不能像 CoinVE 的未压缩 tar 那样直接 Range 到任意 MP4。
- CSV 使用分号分隔，字段为 `video;prompt;original_video`。归档内目录名与 CSV 不完全一致，例如 CSV 的 `local_change/<hash>.mp4` 对应 tar 内的 `local_change_new/<hash>.mp4`。
- 当前 `local_change.csv` 有 479,256 行；其中 230,808 行的 source 和 target 都位于 local-change 归档。其余行跨 local-add、background-change 或 global-style 归档，不能只读取 local-change 就取得完整 pair。

local-change 路径矩阵：

| target <- source | rows |
|---|---:|
| local_change <- local_change | 230,808 |
| local_add <- local_change | 94,323 |
| local_change <- local_add | 80,247 |
| background_change <- local_change | 36,019 |
| local_change <- background_change | 29,998 |
| global_style <- local_change | 4,263 |
| local_change <- global_style | 3,598 |

官方入口：<https://huggingface.co/datasets/Lewandofski/OpenVE-3M>

## 推荐抽样口径

- category：只做 `local_change`，排除 style、字幕、背景和镜头类。
- base population：`local_change.csv` 中 source/target 都属于 local-change 的 230,808 行。
- downloadable population：base population 中，source/target 两个哈希都完整出现在 `local_change_new.tar.gz.00` 固定压缩流前缀内的行。
- method：先固定前缀及完整 member 清单，再用 `random.Random(20260821)` 无放回均匀抽 15 条。
- 这是**固定归档前缀子总体的随机样本**，不是 OpenVE-3M 或完整 local-change 的均匀随机样本。

## 执行记录

1. 固定 revision 并下载 `local_change.csv`；SHA256 为 `eaf9b18ab58ef2f91812a01ca5d5656d2b018b72670ed02d4a36c6366e6765bb`。
2. 固定读取 `.00` 开头 2,147,483,648 字节；前缀 SHA256 为 `12cf8d8d7b924250b30cb2c6526373ea48e74dcca3c6d343f45194c73585ee1b`。
3. 在截断 gzip 中确认 4,189 个完整 MP4 member，与 metadata 连接后得到 39 条完整 pair；因此未扩大到 4 GiB。
4. 第一批使用 seed `20260821` 随机抽 15 条；第二批排除第一批后，使用 seed `20260822` 从剩余 24 条随机抽 15 条。
5. 两批均抽取 30 个角色文件，并生成 H.264 comparison、instruction、contact sheet、manifest、inventory、audit 和 ZIP。
6. 合计 30 条中有 19 条人物相关，20 条指令通过、9 条失败、1 条存疑；只有 19 个唯一无向 pair。详情见 `openve_report.md`。

## 不采用的方案

- 不调用 README 中的全量 `load_dataset`：当前公开视频约 2.03 TiB。
- 不把 full-metadata 随机 15 条写成可低成本抽取：连续 gzip 没有公开索引，随机 member 通常要求顺序读接近完整 151.47 GiB，跨目录记录还会继续增加下载量。
- 不先按“woman/hair/clothing”等关键词筛掉无关行；那是后续定向收集，不是本轮自然覆盖率抽查。
