# 第三阶段：Lance LoRA 微调

> 当前状态：Stage 1/2 已完成，Gemini-only 最终得到 4,141 accepted，并以 seed `20260904` 冻结为 4,000 对编辑池（train/validation/test 为 3,601/195/204）。OpenVid 已完成快速人工审阅且整体可用，但首轮 LoRA 固定为纯编辑训练，不混合 T2V。数据现状与筛选方案的权威摘要见 [`CURRENT_DATASET_STATUS.md`](./CURRENT_DATASET_STATUS.md)。

## 1. 执行顺序

1. OpenVid-1M 高分候选的 50 条离线审核包已完成快速人工验收，整体可用，作为后续 T2V replay 候选保留。
2. 用当前明确的 metadata 与 specialist 规则处理编辑候选。
3. 第二级最终采用 Gemini-only：所有 Stage-1 survivor 都必须获得 `google/gemini-3.8-flash` 的有效判断。Qwen2.5-VL-72B/rescue cascade 只保留为历史 benchmark、路由诊断和消融字段，不再直接决定 canonical accepted。
4. 现有 230 条明确人工标签只用于选择方案和记录误杀/召回；正式 Stage 1 与 Stage 2 之间不设置人工中断。
5. 严格 4,000 条编辑池已形成；首轮 LoRA 只使用其中 3,601 条 train，不使用 validation/test，也不混合 T2V。
6. 先在冻结 validation 上用自适应漏斗从 1.0/1.5/2.0 epoch 中锁定唯一 checkpoint，再做内部 test 与 OmniEdit Spatial；只有出现明显画质/动态退化时，下一轮才考虑 5%–10% OpenVid replay。

## 2. 冻结 4K 编辑池

| 分支 | 冻结编辑池 | 总占比 |
| --- | ---: | ---: |
| GOKU remove | 600 | 15.0% |
| GOKU add | 400 | 10.0% |
| EgoEdit replace | 1,000 | 25.0% |
| ReCo add | 600 | 15.0% |
| ReCo remove | 600 | 15.0% |
| ReCo replace | 800 | 20.0% |
| **合计** | **4,000** | **100%** |

本轮 Stage-2 输出目标是 4,000 条最终 `accepted` 编辑 pair，不是 4,000 条筛选前候选。Stage 2 后的缺口按同一 branch 扩筛，不降低门槛、不复制样本。首轮训练使用其中的 train split 3,601 条，OpenVid 不进入本轮 schedule。

4K `accepted` 编辑池按相同相对分布冻结为：GOKU remove 600、GOKU add 400、EgoEdit replace 1,000、ReCo add 600、ReCo remove 600、ReCo replace 800。`finalize_stage2_pool.py` 使用统一 seed `20260904` 去重并选满这些配额；共享任一 source/target 视频的样本先组成连通组，再按组哈希为 train/validation/test（期望 90%/5%/5%），禁止同一媒体跨 split 泄漏。

## 3. 筛选门和人工验收

- 一级固定使用时长、近冻结、MUSIQ、sharpness、DINOv2 与已知任务排除规则。removal 的严格模糊/低细节硬门不仅作用于常规 remove target，也作用于“把 removal 生成结果反向组织成 add”的 generated source（当前为 GOKU add）；不能只按训练 instruction 的 `add/remove` 字面值决定检查哪一侧。ReCo robot/CG 仅路由为高风险，不用关键词直接拒绝。
- 二级继续使用 5 个避开首帧的对齐时刻、全图 + 紧 ROI 和 provenance-aware 路由。所有 Stage-1 survivor 都由 Gemini 作最终决定；持续 API、内容安全或解析失败进入 quarantine，不能伪装成质量 reject。
- Gemini 3.8 Flash 固定 `reasoning_effort=minimal`，与 230 条 benchmark 保持一致。生产输出按 `sample_id` 可续跑：已成功解析的结果不重复请求，截断、429 或解析失败项必须重试后才能合成最终决策，不能把临时 API 错误误当成模型 uncertain。Parasail 共享池发生 429 时，Qwen 生产请求降为单并发、单行最多 8 次退避重试，外层仍按成功 ID 续跑；接口错误不直接转成 Gemini 的内容判断。
- 后续扩容沿用同一 Gemini-only 规则和 prompt SHA256；不得因分支配额不足而切回 Qwen 直过、降低门槛或改变思考等级。Qwen 字段只用于历史对照。
- VEFX 仅保留为第二阶段的历史对照实验，不进入本轮级联。原因是 VEFX-Reward-4B 在 63 条人工样本上没有形成可用筛坏信号，而本地 4B/32B 部署还分别带来约 13GB/65GB 显存占用与独立环境维护成本。
- 不逐条人工审批，Stage 1 与 Stage 2 之间也没有人工审核口。最终 100 对 HTML 抽查包覆盖最终 4K、accepted reserve、Gemini 推翻旧低风险通过、旧高风险拒绝、补池通过/淘汰、Stage-1 淘汰和 quarantine；每项同页展示 source、target、instruction、dataset/branch、provenance、reference/generated 映射和模型理由。审阅结果用于终审和后续迭代，不回头改写本轮冻结模型结果。
- 所有中间 accepted/rejected/uncertain、原始回答和理由都保留，以便等量 filter ablation；不从人工 uncertain 备注推断二分类标签。
- OpenRouter 密钥只从环境变量 `LIJUYANG_ICLR` 读取，不写入配置、日志或产物。
- 每个阶段完成后，将本阶段的 summary/report、逐样本结果、指标和可审核 HTML 包上传到 `/apps/bypy/lance_review/`，记录远程路径、字节数和 SHA256。不需要人工决策的后续准备继续执行，不因上传本身中断。

### 3.1 统一 Pair 审核界面规范

后续所有 source/target pair 审核包统一采用早期验证过的黑色单页审核器，不再使用把全部样本纵向铺在同一个页面的列表形式：

- 每一页只显示一对视频；`SRC` 与 `TARGET` 左右并排、同时播放/暂停并同步拖动进度，空格控制同步播放，左右方向键翻页。
- 页面顶部必须清楚标注 `dataset`、任务/branch、抽查分组、最终状态与路由，以及 `source_provenance → target_provenance`、哪一侧是 `REFERENCE_VIDEO`/`GENERATED_VIDEO`；instruction 与 sample ID 同页可见。
- 模型区显示 Qwen、Gemini、cascade 的原判定与理由。benchmark 包还必须同时显示冻结人工标签、原人工备注和 split，避免把“旧判定”和“待重新审核”混淆。
- 人工审核口固定为 `accepted / rejected / uncertain`，快捷键为 `1 / 0 / 2`；按 sample ID 自动保存，并支持 CSV/JSON 导出与回传结果导入。允许不打分直接进入下一条，空白保持为“未标注”，不得强制改成 uncertain；这样可以先快速浏览全体再回头判断。
- 抽查默认做确定性的分层覆盖，不用纯随机：覆盖不同数据集、任务、通过/拒绝、直过/复核/救回/边界等路径；只有明确要求时才生成随机包。
- 人工 benchmark 的全量复核包默认保留全部样本，但把模型与人工不一致项放在最前面：先 `false_accept`（漏掉坏样本），再 `false_reject`（误杀好样本），最后才是判断一致项；顶部明确显示错误类型和冻结人工标签。
- 打包前必须检查 HTML 内嵌 JSON、可执行 JavaScript 语法、manifest 行数、每对两段媒体、相对路径和压缩包解包后的结构。浏览器只懒加载当前一对，避免 100/230 对同时建 video 节点导致空白或长时间加载。
- 审核代理视频必须统一为浏览器兼容的 H.264、`yuv420p` 并写入 fast-start 元数据；HEVC/H.265 等源视频只能作为训练数据原件保留，进入审核包前须转码，不能仅凭文件扩展名为 `.mp4` 就判定可播放。转码只修改审核副本，不修改数据池原视频。
- 压缩包内部只允许一层审核包根目录；解压后直接得到该目录，其下就是 `index.html`、`media/`、`review_manifest.csv` 和 `summary.json`，不得保留 `artifacts/...` 等工程目录前缀。

## 4. 费用记录

历史 rescue cascade 正式生产调用约 `$59.1549`。为了把此前 1,882 条 Qwen 低风险直过统一补成 Gemini 判断，新增 `$10.7689461`；Round-4 补池的 373 个最终结果记录 `$2.13383625`，计入一次已被重试覆盖的无 JSON 返回后实际约 `$2.14649`。因此本次 Stage-2 生产、全量补录和补池合计约 `$72.07`。账户总用量还包含 230 条 benchmark 与提示词实验，不能直接和这个生产合计比较。

Parasail Qwen2.5-VL-72B 的实测单价高于 Gemini 3.8 Flash，且 Gemini-only 在冻结 benchmark 上坏样本召回更高，因此后续扩量默认复用当前 Gemini-only 配置，不再把 Qwen 72B 当作廉价主筛。20K 预算必须根据本次各分支 Stage-1 survivor 率与 Gemini accept 率重新反推；不得沿用早期 `$28–34` 或只按最终目标条数估价。`data_collection=deny`、固定 provider、禁止 fallback、严格 JSON schema、可续跑 sample ID 和环境变量密钥保持不变。

230 条 benchmark 及历史级联对照仍见 `CASCADE_BENCHMARK_REPORT.md`，但 canonical 4K 以全量 Gemini 结果为准。

## 5. LoRA 与固定 seed

LoRA 仅注入 Lance 36 层生成专家的 252 个 Linear（每层 attention 的 q/k/v/o 与 MLP 的 gate/up/down），固定 `rank=64, alpha=64, dropout=0.05`。第一轮从官方 `Lance_3B_Video` 基模开始，不继承任何已有微调权重；使用 4×H200、每卡 microbatch 1、梯度累计 2（全局 batch 8）、bf16、480p、12fps、同步 5 秒裁剪、`lr=1e-4` 和 45 step warmup，共训练 2 epoch/902 optimizer steps。

所有可复现对比和后续可能的消融必须共用：

```text
DATA_SELECTION_SEED = 20260904
TRAIN_GLOBAL_SEED   = 42
SCHEDULE_SEED       = 42
EVAL_SAMPLE_SEED    = 42
```

训练前冻结 `training_schedule.csv`，记录 step、sample ID、task slot、sample seed 和 noise seed。对比实验共有样本的顺序、裁剪、diffusion timestep/noise、checkpoint step、评测输入和生成 seed 必须一致。先做两次 32-step 可复现性测试，再启动正式训练。首轮不做 T2V 比例消融。

### 5.1 运行与观察

2026-09-06 06:52（新加坡时间）本轮已完成 902/902 step，正式启动器退出码为 0，W&B 状态为 `finished`。冻结训练 schedule 的 3,601 个唯一样本各出现两次（7,202 次有效曝光，14 个零权重 padding slot）；451/677/902 三个候选均通过完整性、张量有限值与 SHA256 检查，226 也保留。26 个训练代码快照文件在结束时仍与运行前归档一致。核验记录见 `artifacts/lance_4k_r64_seed42_e2_v1/training_completion.json`。该文件是训练结束时的历史快照；后续初筛已锁定 step 677，当前评测状态须查看 evaluation 目录，不能将训练完成当作测评完成。

启动入口为 `scripts/run_general_edit_lora.py`。测试采用连续 32 step 与先 16、再恢复到 32 step；`scripts/verify_general_edit_resume.py` 比较 adapter、optimizer、scheduler、数据游标与每卡 RNG。正式训练始终重新从官方基模初始化，不继承 smoke 权重。

W&B 正式运行归属 `ucantseeit1000-xi-an-jiaotong-university-`，project 为 `lance_general_edit_4k`。启动器在子进程内将环境变量 `WAN_DB` 映射为 `WANDB_API_KEY`；没有环境密钥时允许 SDK 使用 `wandb login` 保存的凭据。密钥不写入配置、启动命令或日志。Smoke 为 offline，正式训练为 online；不上传模型、视频或代码。

`scripts/monitor_training.py` 将逐步日志转为 CSV、TensorBoard 与 `status.json`，提供原始 loss、MA25/MA50、梯度、学习率、速度与 ETA。GPU 指标在计算期间采样；历史日志没有对应的 GPU 观测时留空。TensorBoard 本机端口为 `6006`，可通过编辑器转发该端口访问。正式 run 和 smoke 使用独立子目录。

## 6. 评测与扩量门

版本记录勘误（2026-09-06）：正确 Lance 权重 revision 为 `7395315758865e6f56ab87ad06a88c7ac172f056`（40 位）。旧冻结配置 `configs/lance_4k_r64_seed42_e2_v1.json` 的 `base_revision` 末尾多写了 `d`；为保留已运行训练的配置哈希，此处记录勘误，不回写该冻结文件。基模评测已核对本地主权重 SHA256 与官方正确 revision 完全一致，详见基模 `run.json`。

- checkpoint 筛选不生成 Base 或 0.5 epoch 输出，只比较 step 451/677/902（1.0/1.5/2.0 epoch）。0.5 epoch 的 step226 checkpoint 仍永久保存，只在三个正式候选全部异常时回查。
- 首轮使用 validation 的固定 48 条分层样本，每个编辑分支 8 条，三个候选共生成 144 条。主指标为六分支等权宏平均；同一批样本、推理配置和 seed 在三个 checkpoint 间完全配对。
- bootstrap 区间与六分支胜负仍作为不确定性诊断，但不控制算力分配；首轮宏平均数值最高的 checkpoint 无条件成为唯一晋级者，精确同分时选择更早 checkpoint，并只为它生成剩余 147 条 validation。审核包用于留证和可选抽查，不作为选择门，也不额外生成视频。
- validation 锁定唯一 checkpoint 后不再更换。内部 test 204 和 OmniEdit Spatial 240 都只生成 winner；Base OmniEdit 结果复用 `model_evaluation/lance_base` 的独立 session，禁止重复生成或因 test 分数改选 checkpoint。
- Lance 正式对比统一使用项目现有默认推理配置：`task=video_edit`、30 个去噪 step、`timestep_shift=3.5`、`CFG_TEXT_SCALE=4.0`、`video_480p`、12 FPS、`text_template=true`、`prompt_enhancement=false`、`use_KVcache=true`、bf16；推理涉及的 `validation_data_seed`、`validation_noise_seed` 与 `global_seed` 全部固定为 `42`。基模和微调模型必须使用相同配置，不在 test 上调 CFG、step 或 seed。
- 自动评审与基模复用 OpenRouter `google/gemini-3.8-flash`：正式 OmniEdit v2 固定 `google-vertex/global`、`reasoning.effort=low`、严格 JSON schema、`max_tokens=2048`、禁止 fallback、`data_collection=deny`。发送完整 MP4，不传 `temperature/top_p/top_k`；保存原始响应、`usage`、重试和 `usage.cost`。当前路由无法确认显式控制 `mediaResolution`，记录为 provider default。Studio v1 在基模上有 3 条超过 20 MB，故完整 240 条统一改用已验证大请求可用的 Vertex v2；Base 与 winner 不得混用 v1/v2。已经存在的其他 judge 协议结果独立保留，不自动改写，详见 [`基模评测要求`](../../model_evaluation/lance_base/README.md)。
- OmniEdit Spatial 固定使用官方数据、官方 prompt、四维打分与官方总分公式。论文中表述为“OmniEdit Spatial protocol with Gemini 3.8 Flash (low) judge”。本轮内部 test 只报告 winner 的绝对分数；与 Base 的配对比较限定在复用另一 session 的 OmniEdit 240 条结果，不能据此声称已证明内部 test 的相对提升。4K 扩量前结合内部 test 表现与 OmniEdit 相对收益判断；若基础画质或动态明显退化，再启动 5%–10% OpenVid replay，不在首轮预混。

执行工具：

- `scripts/wait_train_then_generate_screen.py`：绑定本次训练 PID、进程启动时间及 Slurm job；确认训练退出、451/677/902 checkpoint 完整且 GPU 已释放后，自动执行 144 条本地 screen 推理。观察失败只重试观察，不重启训练；不停止其它进程、不调用 judge API，推理子进程去除 API 凭据并强制离线读取模型。状态写入 `evaluation/local_handoff_status.json`。
- `scripts/run_funnel_evaluation.py --plan`：只读核对基模与微调的 judge 协议并列出命令；去掉 `--plan` 自动衔接四卡生成、评分、唯一 winner 晋级及审核包。支持 `--through screen/remaining/test/omni`，默认跑到 Omni；审核包在所请求的评分阶段结束后生成，不阻塞晋级。`evaluation/judge_contract.json` 单独冻结已验证的路由，明确替代训练 profile 中未执行的 `judge_media_resolution=MEDIA_RESOLUTION_LOW`；不改动训练 profile 的恢复哈希。
- `scripts/prepare_evaluation_configs.py --stage screen`：冻结三个候选各四个单卡 shard，共 144 条；`remaining/test/omni` 必须读取自动晋级结果，只准备唯一 winner。
- `scripts/run_evaluation_jobs.py --jobs .../screen_jobs.json`：四卡并行生成同一 checkpoint 的四个 shard，再处理下一个 checkpoint。检查完整 checkpoint、输出 provenance 和 adapter/config SHA256，已有输出完整解码通过后复用。
- `../../model_evaluation/lance_tools/judge_spatial_openrouter.py --route vertex --tasks .../screen_generation_tasks.json --output .../screen_scores_vertex_v2`：从 `OPEN_ROUTER_QWEN` 读取凭据，默认两并发；保存每次请求原始响应、费用、重试和逐样本分数。解析错误、安全拦截或截断不视作零分，成功结果按输入视频、输出视频与协议哈希复用。旧 `scripts/judge_spatial.py`（直连 Google）及 `--route studio` 结果不能与 Vertex v2 基模分数混用；更改协议需完整重评该比较组并使用独立目录。
- `scripts/decide_funnel_promotion.py --scores .../screen_scores_vertex_v2/scores.csv`：首轮 144 条全部有效后选最高六分支宏平均，仅晋级一个 checkpoint。
- `scripts/build_model_audit.py --scores .../screen_scores_vertex_v2/scores.csv --output .../review_val48`：每分支选两个低分/分歧样本，展示三个 checkpoint，共 36 对；复用生成输出并同步源视频采样帧，生成 H.264 单页审核包、ZIP 与 SHA256。人工结果不影响已冻结的选择。

本侧自动漏斗已固定 `--route vertex`，所有评分目录采用 `*_scores_vertex_v2`，与 Base 的完整 v2 协议逐字段核对。未执行的旧 Studio 合同保留为 `evaluation/judge_contract_studio_v1_unexecuted.json`；训练 profile、seed 和本地生成任务不变。用户于 2026-09-06 11:37 明确允许已披露的完整视频载荷经 OpenRouter/Google Vertex 发送及恢复包发往百度网盘；授权记录保留在原 hold 文件中，并重新通过正常工具审核。正式首条 Vertex 评分预检成功（`$0.00139125`），结果复用于全量；未经授权的路由变化仍被拦截。

费用统计优先汇总 OpenRouter 返回的 `usage.cost`（USD），含重试；缺少费用的请求单列，最终扣款仍以账户账单为准。[Google 官方定价](https://ai.google.dev/gemini-api/docs/pricing) 仅用于预算参考，不能代替本轮路由的实际费用记录。

### 6.1 扩量前重算第一级

旧 4,003 条只是旧 5K pilot 的 Stage-2 入口，不是最终训练比例。后续扩量建立新版 Stage-1 manifest：重写 ReCo 为 `generation_side_unresolved` 并检查两侧；复用已有 frame diff、sharpness 和 target MUSIQ/DINO；补算缺失的 ReCo source MUSIQ/DINO；应用最新版近冻结与 removal 模糊硬门槛。新候选完整运行同一版 Stage 1 和 Gemini-only Stage 2，再根据各分支实测 survivor/accept 率反推过采样量。是否混入 T2V 由首轮 VBench 退化结果决定，不预设 10% 为固定比例。

### 6.2 半小时健康监控与本次恢复

2026-09-06 13:28，原串行控制器因 remaining147 中一条评分连续 HTTP 502 而退出；147 条视频均已生成并完整解码，146 条评分成功。15:56 已按冻结 winner 677 恢复四卡 test204 → Omni240 生成，独立评分衔接在每阶段完整解码后使用原 Vertex v2 协议执行；该条验证分数不再阻塞已经选定模型的后续生成。补试累计 8 次仍返回 502 或响应体 504，失败项 `scale-r2-egoedit_replace-70abe58fe920278e` 保留，不记零分、不删除、不更改媒体或协议。此记录描述当时状态，后续结果以回执为准。

用户已将本次评测检查频率明确为每 30 分钟。`scripts/monitor_evaluation.py --expected-job 123779 --interval 1800` 于 16:10 启动；它只观察当前 allocation 的相关进程、四卡利用率、各阶段文件/解码记录/成功评分数，将结果写入 `evaluation/health_status.json` 和追加式 `health_history.jsonl`。该监控不调用 API、不自动重启任务；异常由主会话根据实际 PID、启动时间和日志核对后处理。一次观察失败不能当作进程死亡。

`pipeline_status.json` 暂保留原串行控制器的失败记录，不代表恢复任务停止。恢复期间结合 `health_status.json` 与实际进程确认状态，不要另开整条 funnel 与现有 generation/scoring 进程竞争锁。健康记录中的 `complete_reported` 只是进度依据，不能替代最终完整性、SHA256 和完整评分清单审计。

会话切换后，健康监控于 16:53 以独立进程重新启动；16:57 的 `scripts/resume_evaluation_after_session.py` 接管仍存活的 test wrapper，等待其自然结束并复查缓存，然后衔接四卡 Omni 生成和评分，没有重启已存活的 GPU 任务。用户在 16:56 再次明确同意自动评分发送，授权记录见 `evaluation/api_authorization_hold.json`。当前恢复阶段以 `evaluation/session_recovery_status.json`、`logs/session_recovery_job123779.log` 和实际进程为准，监控日志为 `logs/health_monitor_job123779.log`。

2026-09-06 20:01，初筛可选审核包 `evaluation/review_val48.zip` 已生成：六分支各两条低分/分歧样本，每条展示三个 checkpoint，共 12 个唯一样本、36 对。48 段审核媒体均通过 H.264/yuv420p、完整解码与配对帧数/FPS 检查，内嵌 JSON、JavaScript 语法、ZIP CRC 和单层根目录检查通过；包大小 28,021,367 bytes，SHA256 见同名 `.zip.sha256.json`。仅创建本地审核副本，未新增推理或评分，也不改变 winner 677。

`scripts/summarize_funnel_evaluation.py --output .../evaluation/evaluation_report.json` 仅离线读取冻结 manifest、成功评分回执和费用记录，并重新核对被评分的 source/output SHA256；不调用 API、不修改 winner。内部 validation/test 按实际 `branch` 汇总微平均和六分支宏平均，缺分明确列出且不计零。只有 Base 与 winner 的 Omni 240 对全部有效、输入/指令/seed 和 judge 协议一致，才输出整体、四维、24 类别差异及按类别分层的配对 bootstrap 区间（10,000 次、seed 42）；该区间不涵盖 judge 重复打分或多训练 seed 的变异。报告快照不替代生成 provenance 和完整解码的最终审计。2026-09-06 20:29 的首次实际运行已核对现有全部成功回执及媒体哈希；报告为 `incomplete`，不是最终全量结果。

2026-09-06 20:25，唯一验证缺分项的第 12 次补试在执行前再次被工具安全审核拒绝，没有发出请求或启动重试进程。具体记录为 `evaluation/remaining_retry12_authorization_hold.json`，须取得该载荷/目的地的具体确认并重新通过正常工具审核后才能重试；不得借助另一启动器、路径或协议绕过。此前已运行的 Omni 生成和恢复控制器未被停止或重启。

上述授权随后已解决：用户明确回复“允许”，第 12 次请求于 21:12 通过正常工具审核后发出，21:15 确认仍为上游 504 超时，没有有效评分，费用未返回。累计 12 次上限已用尽，停止自动重试，validation 保留 194/195、整条漏斗保留 `incomplete`；当前障碍是单项请求持续失败，不再是缺少授权。此项不影响已完成的 Omni240 比较，未更改视频/协议或安排第 13 次。

### 6.3 首轮评测结论

2026-09-06 20:53，winner 677 的 Omni240 和内部 test204 已全部评分完成，四卡推理结束。Omni 总分 Base **1.9514** → LoRA **1.2134**，配对差值 **-0.7380**，95% 分层 bootstrap 区间 **[-0.8522, -0.6239]**；本轮未获得预期提升，主要下降为编辑准确度。735 条输出已核对冻结任务、adapter/config provenance 和完整解码 inventory 的当前 SHA256，成功评分与媒体也已重新匹配。整条漏斗仍有一条 validation 缺分，状态保持 `incomplete`；不因此否定完整 Omni 比较，也不按 test/Omni 改选 checkpoint。完整结果、限制与费用见 [`evaluation/RESULTS.md`](artifacts/lance_4k_r64_seed42_e2_v1/evaluation/RESULTS.md)。

## 7. 存储约定

训练真值使用冻结 CSV manifest，Lance loader 使用 Parquet，不新增数据库。大规模解压、Parquet 副本和 cache 放 `$TMPDIR`，最终 manifest、adapter、日志与 SHA256 回传 `/scratch`。只保留 best/last 的建议由评测产生，尚未评测 checkpoint 不自动删除。

本轮本地训练恢复归档为 `artifacts/lance_4k_r64_seed42_e2_v1/training_recovery_v1.tar`，共 35 个文件、3,368,058,880 bytes，所有归档文件已逐一与原文件比对 SHA256；完整清单见同目录 `training_recovery_v1.receipt.json`。902 包含完整恢复状态，226/451/677 在归档内只含推理权重及 LoRA 配置。该包不含视频或凭据，也不等同于后续模型输出审核包。百度上传此前在执行前被安全审核拒绝；2026-09-06 11:37 获得明确用户授权后，重新核对远端目标不存在并通过正常工具审核开始上传。进度见 `training_recovery_v1.baidu_upload.json`；在远端大小和上传回执核验完成前，不把它报告为已完成外部备份。

2026-09-06 20:01 已完成上述上传核验：重新计算本地归档 SHA256 与冻结值一致，上传状态为 `complete`，百度实时列表中指定路径唯一存在，远端大小 3,368,058,880 bytes、文件 ID `898522629005543` 均匹配上传回执。详见 `training_recovery_v1.remote_verification.json`；此次核验未重新下载远端内容，不能将本地 SHA256 表述为远端逐字节哈希校验。
