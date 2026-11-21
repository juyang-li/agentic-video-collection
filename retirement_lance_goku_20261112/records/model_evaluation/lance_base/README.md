# Lance 基模 OmniEdit Spatial 评测

> 状态：240 条基模视频全部生成、完整解码通过；正式 Gemini Vertex v2 评测已完成 240/240，零失败、零重试。Studio v1 仅作不完整历史记录保留。

## 固定范围

- 数据集：`OmniEdit-Bench/OmniEdit-Bench`
- track：Spatial 全集，240 条（24 类 × 10）
- dataset revision：`7447b1a57b52585c3c2c5ac7f00feefdf89f9d61`
- Lance 代码 commit：`9717d9160ed16fedfdad5ce39ccbc6c9d69695e0`
- Lance 权重 revision：`7395315758865e6f56ab87ad06a88c7ac172f056`
- 推理：Lance 默认 video-edit 配置，30 steps、shift 3.5、CFG 4.0、480p、12 FPS、bf16、KV cache 开、prompt enhancement 关
- seed：framework/global、validation data 与 diffusion noise 均固定为 42

本轮只生成 Lance 基模输出。模型微调完成后，对选定 checkpoint 使用完全相同的输入、配置和 seed 再跑一遍，然后交给原始 OmniEdit Spatial evaluator 比较。

2026-09-06 最终核验修正了旧记录的 revision 笔误（末尾多写一个 `d`，成为无效 41 位字符串）。实际主权重 SHA256 为 `7f0550e1d1511b29a4740a67c1e18e176302a4ecb3177c8a5850ff5fe6447c25`，与上述官方 40 位 revision 的文件完全一致；没有更换模型或重新生成视频。

## 正式结果（Vertex v2）

| 指标（0–5） | 240 条均值 |
| --- | ---: |
| Accuracy | 2.4708 |
| Preservation | 2.9500 |
| Realism | 2.7792 |
| Consistency | 2.8542 |
| 官方 Overall | **1.9514** |

评分于 2026-09-06 05:20（新加坡时间）完成，24 类各 10 条。正式 240 次请求费用 `$0.41433`，无重试、无缺失费用。Studio v1 的已报告费用 `$0.39063`、Vertex 大视频传输验证 `$0.00168825` 另计；本次全部已报告 API 费用共 `$0.80664825`。另有 12 次 HTTP 413 与 2 次 HTTP 403 拒绝请求未返回费用字段，不伪装成已确认零扣款；账户最终账单优先。

逐条分数：[scores.csv](./gemini38_low_vertex_v2/scores.csv)；分类均值、token 和费用：[summary.json](./gemini38_low_vertex_v2/summary.json)；协议：[protocol.json](./gemini38_low_vertex_v2/protocol.json)。正式结果只包含 Vertex v2，不包含旧 Studio 分数；只代表 Spatial 单轮编辑，不外推到全部视频编辑能力。

## Judge 配置

Gemini 评测保持官方 source/edited MP4 输入、官方 prompt、四维分数与总分公式，judge 固定如下：

- API 路由：沿用筛选阶段的 OpenRouter 凭据 `OPEN_ROUTER_QWEN`，模型固定 `google/gemini-3.8-flash`，正式 v2 provider 固定 `google-vertex/global`（响应 `provider=Google`），`allow_fallbacks=false`、`data_collection=deny`；不使用 `latest` 或备用模型；
- reasoning：显式设置 `reasoning.effort=low`，映射至 Google `thinkingLevel=low`；这里的 low 指思考等级；
- 视频输入：两段未经重编码、未经抽帧的完整 MP4，以 base64 发送。当前 OpenRouter 路由没有已确认可用的 `mediaResolution` 控制，故如实记录为 **provider default**，不能宣称已固定 `MEDIA_RESOLUTION_LOW` 或保证服务端 token 化永不变化；
- 输出：严格 JSON schema，四维整数 0–5 加 `reason`；`max_tokens=2048`；
- Gemini 3.8 请求不传已弃用的 `temperature`、`top_p` 或 `top_k`；
- 每条保存原始响应及 OpenRouter `usage`，汇总 input/output/reasoning token、请求数、重试数和返回的 `usage.cost`（USD）；缺失费用或请求中断必须单列，不能当成免费；
- 同一正式比较中的基模和所有微调 checkpoint 必须复用同一 judge 配置、prompt 和 evaluator 版本；配置变化必须建立新评测版本，不得混合汇总。

官方 evaluator 与 prompt 的固定副本位于 `runtime/omni_edit_evaluator/`，外围适配不得直接改写这两份文件。论文中表述为“OmniEdit Spatial protocol with Gemini 3.8 Flash (low) judge”。

2026-09-05 的 Studio v1 首条预检成功，但全量只得到 237/240：`Sp_Landscape_change_07`、`Sp_Season_change_02/05` 的原始 source/generated MP4 请求超过该路由 20 MB 限制（HTTP 413）。两次官方公开 URL 传输尝试也返回 403。没有压缩、裁剪或抽帧。

2026-09-06 已实测 `google-vertex/global` 可以接收最大的约 33.8 MB 原始请求，因此建立独立 **Vertex v2**，240 条统一重评，绝不将 Studio 237 条与 Vertex 3 条混合汇总。基模与后续 winner 的正式 OmniEdit 对比必须统一使用 v2。v1、传输诊断和正式 v2 的费用分别记录；同一 v2 内仍按视频 SHA256 和协议哈希复用。

此前“直连 Google + 显式 mediaResolution low”为未执行方案，不属于实际评测协议。

用户已授权本任务的付费 API 调用及重试，无需再次逐次审批。实现与参数依据：[视频输入](https://openrouter.ai/docs/guides/overview/multimodal/videos)、[thinking level 映射](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。

## 文件语义

三个概念严格分开，不把模型输出称作 target：

- `official_source_video`：OmniEdit 发布的输入视频；
- `dataset_target_video`：数据集提供的参考编辑视频；OmniEdit Spatial **没有提供**，manifest 中显式记录为 `not_provided_by_omni_edit`；
- `generated_video` / `model_output`：Lance 根据 source 和 editing instruction 生成的待评测视频。

正式 manifest 为 `omni_edit_spatial_240.csv`。当前单卡配置为 `configs/spatial_full_seed42/shard_00.json`，生成结果写入 `outputs_raw/omni_edit_spatial_base_seed42_shard00/`。

## 准备与运行

准备完整官方 source、manifest 和单卡 config：

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python \
  scripts/prepare_spatial_full.py --download --shards 1
```

在一张 H200 上运行：

```bash
bash scripts/run_spatial_single.sh
```

`model_evaluation/lance_tools/run_video_edit.sh` 会完整解码已有 MP4，只补缺失项，并写出带 SHA256 的 technical inventory。因此中断后可安全重跑。

当前接管运行位于 job `123662`、tmux socket `lance_spatial_123662`。后台 `scripts/watch_generation_and_evaluate.py` 只监控本次基模输出：确认推理进程和原 generation pane 都已结束后才恢复缺失项；损坏的中断 MP4 移到 `quarantine_interrupted/` 保留。进程仍活着或观察失败时不会擅自重启。换 allocation 后必须先核实新分配，不借用同用户名的其他 GPU 作业。

`pipeline_status.json` 记录阶段，`logs/pipeline_vertex_v2.log` 记录当前后台日志（`logs/pipeline.log` 是 v1 历史）。完整 240 条全部解码通过后，自动准备官方目录映射并调用 `../lance_tools/judge_spatial_openrouter.py --route vertex`。正式评分、每次请求回执和总/分类汇总位于 `gemini38_low_vertex_v2/`；必须同时满足 `scope=full_manifest`、`tasks_succeeded=240`、`tasks_failed=0` 才算评测完成。API 失败不记为零分。

生成完成后，把数字编号输出以硬链接映射为官方 evaluator 要求的 `video_id.mp4`：

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python \
  scripts/prepare_evaluator_inputs.py
```

## 保留的 smoke

旧的单条 smoke 仅用于证明本地权重和环境可运行，不属于正式 240 条分数。其输入、输出、日志和配置暂时保留在 `data/smoke/`、`outputs_raw/smoke/`、`logs/smoke.log` 与 `configs/smoke.json`。

## 基模全集审阅包（滚动版，2026-09-06）

用户本次明确将“按下一项”改为**网页连续滚动**，此项覆盖本包原有逐条翻页约定，不改变其他训练审阅包。全量包含 240 条，非低分抽样；每条显示官方输入与 Lance 基模输出、完整指令、类别 / ID，以及 Gemini 3.8 Flash（low）Vertex v2 的四项原始分数、官方总分和完整评语。

- 深色离线 `index.html`，直接滚轮浏览；支持类别 / 指令 / ID 筛选和总分排序。
- 左侧为按 Lance 采样帧对齐的 source 审阅代理，右侧明确标注 `LANCE BASE / GENERATED_VIDEO`；不称为数据集 target / GT。
- Gemini 实际评分始终使用完整原始 MP4。只有审阅副本做 H.264 / yuv420p / faststart 兼容处理，manifest 保留原片 SHA256、审阅路径和源帧索引。
- 配对视频同步播放 / 暂停 / 拖动；视区附近懒加载、远离后释放，避免同时加载 480 个视频。空格控制当前条目。
- 人工结论初始全部**未标注**，仅浏览不生成标签；`1/0/2` 接受 / 拒绝 / 不确定，支持备注、按 sample_id 保存、CSV / JSON 导入导出。
- 正式 `scores.csv`、`summary.json`、`protocol.json` 原样收录；绝不混入 Studio v1 分数。归档不包含模型权重、凭据或账户配置。

构建脚本：[build_scroll_review.py](./scripts/build_scroll_review.py)；滚动模板：[scroll_review.html](./scripts/scroll_review.html)；真实浏览器验收：[test_scroll_review.py](./scripts/test_scroll_review.py)。构建和测试不会重新调用模型或付费 API。

产物根目录：`review/lance_base_spatial240_scroll_gemini38low_vertex_v2_20260906/`。先构建、通过浏览器验收，再用 `build_scroll_review.py --package-only` 打包；打包前检查全部媒体 SHA256、240 条分数与原始 judge 回执的精确视频绑定。ZIP 只有一个顶层目录，解压后直接打开其中 `index.html`。

已按用户授权上传到百度网盘：`/apps/bypy/lance_review/lance_base_spatial240_scroll_gemini38low_vertex_v2_20260906.zip`。2026-09-06 16:12（新加坡时间）重新查询网盘目录，确认路径、文件 ID `115467838360065` 和大小一致。

- 大小：`439511175` 字节（439.51 MB / 419.15 MiB）。
- ZIP SHA256：`8238a43acad5980797100c4a1658369fa64a3c46665d7ccbaa4457c3d2761d27`。
- 校验回执：[sha256.json](./review/lance_base_spatial240_scroll_gemini38low_vertex_v2_20260906.sha256.json)，已以同名上传到相同网盘目录。SHA256 是本地 ZIP 校验值；远端单独验证上传完成、文件 ID 与字节数，不宣称已全量下载远端重算 SHA256。
- 全部 480 个 MP4 完整解码、H.264 / yuv420p / faststart、配对帧数 / FPS 检查通过；240 个生成视频均只做视频流无损重封装，未重新压缩。
- 全部 240 条分数、评语及原视频 SHA256 与正式 Vertex v2 原始回执对应；ZIP CRC、单顶层目录和 480 个媒体路径通过。
- Chromium 151 离线验收通过：首 / 中 / 尾视频实际播放、双向拖动同步、滚动卸载、全部评分文本、CSV / JSON 往返、刷新保存与移动端布局；无 JS 错误、无外部网页请求。
