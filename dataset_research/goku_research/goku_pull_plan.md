# GOKU-2M 抽查计划

> 状态：已获 gated 权限并完成 `swap_alter` 随机分卷内 15 对抽查；最终结果见 `README.md` 和 `goku_report.md`。

## 当前观察

- 官方将约 2M 个编辑 pair 分到两个仓库，共约 9.65 TB，许可为 CC-BY-NC-4.0。
- 当前 revision：`bigfacing/GOKU-2M@f289f4b245648dc57db04afaaeb68659440c0fa6`、`Goku-2M/GOKU-2M@f0ee242abea24e3a410d5e49b9d5821450d08acb`。
- 两个仓库都要求登录并同意共享联系信息；本轮已用本机 Hugging Face 缓存凭据完成授权下载，凭据本身未写入项目。
- 每个任务由一个 `*.jsons.tar` 和若干**可独立解包的未压缩 `*.videos.partNN.tar`** 组成。每个 case 以 `case_id` 组织，单步任务直接给出 `source_video`、`edited_video` 和 `instruction_en`。
- 与人物属性最直接相关的是 `swap_alter`；style 不进入本轮。该任务有 32 个视频 tar，共 519.29 GiB；单卷 14.85–23.08 GiB；metadata tar 为 0.41 GiB。
- 源视频由 Koala-36M 筛选而来，edited video 由合成流程生成；是否为真人、是否真的只改人物属性仍需看随机样本，不能由论文描述代替。

官方入口：<https://huggingface.co/datasets/bigfacing/GOKU-2M>、<https://huggingface.co/datasets/Goku-2M/GOKU-2M>

## 推荐抽样口径

- population：随机选中的一个 `swap_alter` 视频 tar 内，metadata 可匹配且同时含 source/target 的全部完整 case。
- shard sampling：在 32 卷中用 `random.Random(20260821)` 均匀抽一卷，结果为 `swap_alter.videos.part05.tar`，大小 23.08 GiB。
- pair sampling：列出该卷全部完整 case 后，再用固定 seed 无放回均匀抽 15 个。
- 这是**随机分卷内随机样本**，不是 GOKU-2M 全量均匀随机样本。若先从全任务 metadata 抽 15 条，预计会散落到约 12 卷、需要约 200 GiB，暂不值得。

## 实际执行

1. 固定 revision，下载 `swap_alter.jsons.tar` 与 `swap_alter.videos.part05.tar`；SHA256 分别为 `0e2acb13...b70` 和 `42360d43...dae`。
2. 视频卷内有 8,262 个 source/target 完整 case；两个 annotation 是 0-byte JSON，排除后固定 8,260 条总体。
3. seed `20260821` 无放回随机抽 15 个 case，未按人物、质量或关键词预筛。
4. 只提取命中的 30 个角色 MP4，生成左右 H.264 comparison、instruction、6 帧 contact sheet、manifest、inventory、audit 和下载 ZIP。
5. 30 个角色文件、15 个 comparison 均通过解码检查；随后删除重复的独立角色 MP4 和临时 tar，只保留最终观察材料及复现证据。
6. 人工审核结果为人物属性相关 0/15、instruction 严格通过 6/15，因此停止自然随机扩量。

## 停止条件

- gated 权限未通过时停止，不用官方展示或 Goku-Bench 顶替随机训练样本。
- 15 条中人物属性相关样本很少时，先报告自然覆盖率；后续如需收集训练数据，再另建“人物属性定向筛选”总体，不能把它写成随机抽查。
