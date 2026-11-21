# Stage 1 扩量候补池（2026-09-06）

**已完成并通过独立核验：20,000 对输入，保留 18,662 对候补（16,916 accepted + 1,746 uncertain），淘汰 1,338 对，quarantine 为 0。** 分支结果和后续入口见 [REPORT.md](REPORT.md)。

本轮按用户要求扩大 GPU 一级筛选，暂不加入 CoinVE，不运行 Stage 2 或任何付费 judge。输入固定为 **20,000 对新 pair**；一级 survivor 仍只是候补，不代表已可直接训练。

| 分支 | 新候选 |
| --- | ---: |
| GOKU remove | 3,000 |
| GOKU add | 2,000 |
| EgoEdit replace | 5,000 |
| ReCo add | 3,000 |
| ReCo remove | 3,000 |
| ReCo replace | 4,000 |

沿用首批 4K 的六分支比例、固定官方 revision、metadata 排除规则和 `general_edit_stage1_quality_v2` 门槛。seed 为 `20260906`，分支内部沿用现有 scope 分层抽样；ReCo 覆盖已有 15 个 shard，GOKU 覆盖已有 4 个 population，不是全库均匀抽样。

`manifest.csv` 已冻结且不得覆盖。已排除历史运行与初始审核使用过的 pair。为保护冻结验证／测试集，在完整候选加历史的共享媒体图上排除与 held-out 媒体连通的候选，媒体身份沿用首批 `(dataset, locator)` 定义；该检查不声称识别了不同 locator 下的视觉重复。共额外排除 1,520 条 EgoEdit 候选。

生成侧路由：GOKU remove 检查 target，GOKU 反向 add 检查 source 并应用 removal 清晰度硬门；EgoEdit 按逐 pair 的 original/edited provenance 检查生成侧；ReCo 两侧为 `generation_side_unresolved`，两侧均评分。所有角色都进行时长与解码检查。使用 12 帧、224×224 `INTER_AREA`、MUSIQ-Koniq、DINOv2-small 和冻结的近冻结／清晰度门槛；模型离线运行。机器人／CG 沿用生产流程，不按关键词淘汰，配对语义与伪影审核留待后续 Stage 2。

状态入口：

- `selection_summary.json`：候选总体、筛除数量、固定配额、revision 和 manifest SHA256。
- `acquisition_state.json` / `acquisition.log`：媒体获取；tar 校验官方 ETag/SHA256，EgoEdit 校验 ZIP CRC32。
- `progress.json` / `gpu.log`：实际 GPU 评分进度；`score_journal.jsonl` 支持断点续跑。
- `selection_verification.json`：全量清单、历史去重、共享媒体连通图及路由规则的独立复核；`verification.json` 在运行结束后覆盖每条决定与原始分数。
- 最终产物为 `completion_summary.json`、`pair_decisions.csv`、`stage1_survivors.csv` 和 `quarantine.csv`。accepted 和 uncertain 均保留给后续 Stage 2；本轮没有下载／指标错误，quarantine 为空。

共享视频仅复用原始指标，再展开为逐 pair 角色分数；不复用依赖任务上下文的质量决定。原始归档临时保存在 `/tmp/lance_stage1_reserve_20k_20260906/`，正式媒体和状态保存在本目录。历史 4K、阈值文件和人工审核结果均未覆盖。

执行位置：作业 `123778`，节点 `H64Y574`，一张 H200。下载与评分进程均已正常结束；`gpu_runtime.json` 记录执行时的进程与设备，`verification.json` 记录完整逐条复核结果。

当前评分使用 8 个 CPU 解码线程预取最多 16 个视频的小帧数组，GPU 模型调用保持串行。`prefetch_verification.json` 记录初始 4 路预取对原流程的 16 个 EgoEdit/ReCo 视频对照：指标一致，耗时从 10.89 秒降到 3.06 秒。`prefetch_eight_worker_verification.json` 进一步在 64 个 EgoEdit/GOKU 视频上比较 4 路和 8 路：全部指标逐项一致，耗时从 18.11 秒降到 11.90 秒。这些结果是小批测试，不当作整轮吞吐承诺。

`first_ready_snapshot_*` 是首批媒体到齐的 115 对 EgoEdit 的运行中快照，用于确认新媒体复用与逐 pair 判定完整跑通；结果为 92 accepted、23 uncertain。它不是整轮结果，后续使用最终 `stage1_survivors.csv`。
