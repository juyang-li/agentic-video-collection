# 项目协作规则

## 运行位置与集群分区（用户于 2026-11-13 更新）

- **只使用CPU的任务直接在login节点执行，不为此排GPU。** 包括纯CPU数据准备、下载、评分、验收、打包和上传；CPU任务入口须允许login节点运行。
- **需要GPU的集群任务只能提交到 `testqueue` 分区，严禁提交到 `cpup`、`cpun` 或其他分区。** 适用于GPU训练、推理、预处理和续跑。
- 提交GPU任务前检查脚本和实际参数，明确指定 `--partition=testqueue`；不得通过命令行覆盖为其他分区。GPU入口检查 `SLURM_JOB_PARTITION`，不等于 `testqueue` 时立即退出。纯CPU任务不依赖Slurm分配，不得伪造分区变量绕过入口检查。
- **需要排队时就休眠等待，绝不为了绕过排队而换分区。** 默认沿用用户指定的 15 分钟检查节奏；异常修复后恢复等待。
- 发现本任务误入其他分区，应立即停止对应作业并取消其后续依赖，保留可续跑数据；不得继续运行或自行重投其他分区。

## 对话开始前

- 测评执行与汇总先读 [测评经验](model_evaluation/METRIC_EVALUATION_LESSONS.md)，当前入口见 [测评索引](model_evaluation/README.md)，最终分数见 [300／116汇总](model_evaluation/FINAL_RESULTS_300_116.md)。不要根据旧README的作业号或submitted文字自动重启已完成任务。

- 每次开始处理本项目的任务时，先完整阅读根目录下的 `任务描述.md`，再进行分析、讨论或修改。
- 进行数据集样本抽查、质量审核或编写相应脚本时，还要阅读根目录下的 `DATASET_AUDIT_GUIDE.md`。
- 进行 VBench 评分、结果汇总或跨模型比较时，阅读 [VBench 测评经验 README](model_evaluation/VBENCH_README.md)。用户于2026-11-15明确要求：Temporal Flickering按全量原始分数统计，不做静态筛选；旧静态共同子集仅作历史记录，不是当前默认流程或验收门槛。
- 进行输出与GT的SSIM、PSNR、LPIPS测评时，阅读 [三指标测评方式与经验](model_evaluation/SSIM_PSNR_LPIPS_README.md)，固定配对、时间轴、有效画面和数值参数；不与SOURCE比，不混用VBench原生画布与GT指标的归一化媒体。

## 沟通与写作风格

- 采用渐进式披露：每次只讨论当前最关键的一小部分，避免一次输出过长。
- 当内容较多时，在合适的位置暂停，询问是否继续讨论下一部分。
- 写入文件的内容也遵循渐进式披露；先形成最小可用版本，经确认后再逐步扩充。

## 工作方式

- 用户已明确要求：既定任务所需的 API 调用直接执行，后续也不逐次暂停询问；沿用已确定的模型与调用配置。
- 已确认的训练、监控与评测流程持续推进；常规执行步骤默认可行，不逐项中断等待用户回复权限。
- 已授权的审核包上传到百度网盘 `/apps/bypy/lance_review/`，无需逐次确认；当前任务明确要求“不上传”时优先服从。**用户于 2026-11-09 更新默认规则：只上传 ZIP，不单独上传 JSONL、哈希或 sidecar；不额外核对哈希和远端大小，不整包下载回读，以上传工具返回成功为完成依据。** 本地可保留简要上传记录；只有明确要求时才增加传输核对。
- 正式训练进入稳定运行后，人工式状态检查以每小时一次为默认；启动验证、异常和完成节点可额外检查。底层日志与指标继续自动记录，不需要频繁向用户汇报逐步进度。
- 不要把问题过度复杂化。
- 从最小验证开始，先验证核心假设，再逐步扩大数据规模、实验范围和实现复杂度。
- 不追求一次覆盖所有情况；未验证的设想应明确标注，不提前写成既定结论。

## 数据与目录设计

- 避免层级很深的文件夹结构，优先采用少量、稳定、语义明确的浅层目录。
- 数据设计优先使用 ER 模型表达实体、属性、主键、外键和实体间关系，不依赖目录层级隐式表达数据关系。
- 视频等大文件可以保存在文件系统中，其路径、来源、配对关系、标注和处理状态应由结构化元数据统一管理。
- 在扩展完整 schema 前，先用最少的实体和字段跑通一个小规模端到端样例。
- 试验阶段不要主动引入数据库；是否开始使用数据库，必须先询问用户并取得确认。
- 经确认需要数据库时，默认使用 SQLite，并优先将稳定属性设计成关系字段。
- JSON 字段只用于结构高度灵活、暂时无法合理关系化的数据，不把 JSON 当作默认的数据建模方式。
- 后续审核包增加固定任务导航与分组锚点，任意位置可跳转任务；VACE prepared input 和 overlay_video 只在整个包前两个 sample 展示并打包。
- 审核包默认采用无评分表、完整展开的 scroll down 页面；只有明确要求评分时才生成 audit。带评分表时各审核项默认 `1`、`reviewed=0`，不计作人工通过；`1` 表示通过、`0` 不通过、`2` 存疑。完整模式规范见 `DATASET_AUDIT_GUIDE.md`。

## 集群存储与大文件

- 当前项目目录位于 `/scratch` 的 WekaFS 分布式网络文件系统上；`/home`、`/apps` 和 `/cm/shared` 也使用 WekaFS。当前没有发现单独挂载、面向用户长期归档的 NFS/SMB NAS。
- `/scratch` 用于保存项目数据真值、结构化元数据、正式 checkpoint 和评测产物，但不能默认视为有备份的长期归档；关键 final 权重还应复制到经确认的长期存储或外部网盘。
- `/raid` 是计算节点本地盘，不是跨节点共享的 NAS。不同节点的容量和内容可能不同，作业结束后也可能被清理，因此不得把不可重建数据或 final checkpoint 的唯一副本放在 `/raid`。
- 作业临时数据优先放在调度器提供的 `$TMPDIR`；当前环境的 `/tmp` 也可能映射到作业隔离的 `/raid/scratch/...`。不要直接把 `/raid` 根目录当作长期工作区。
- 适合暂存在 `$TMPDIR` 或作业本地 `/raid` 的内容包括：可重新下载的数据 shard、训练 Parquet 副本、解压视频、抽帧、latent/cache、模型读取副本、中间输出和临时 checkpoint。
- 推荐流程是“从 `/scratch` stage 到 `$TMPDIR` → 在本地盘训练或推理 → 定期并在作业结束前 rsync 回 `/scratch`”。回传后应检查文件完整性；重要权重归档时还应记录并核对 SHA256。
- 清理 checkpoint 前必须先盘点精确路径和文件组成，并区分推理权重与恢复状态：`model.safetensors` 可用于推理和从权重启动新实验，optimizer/scheduler/data cursor 仅在需要精确 resume 时保留。不得删除尚待评测的中间 checkpoint 或仍可能用于续训的 final checkpoint。

## 当前目录映射（2026-11-21）

- `dataset_research/coinve_tripvvt_11k_20261108/`：正式11K，canonical媒体已内部化；保留media、清单、reader和元数据。
- `model_evaluation/FINAL_RESULTS_300_116.md`：最终分数；300为六模型，V4自建集为116条四模型，不是166条。
- `model_evaluation/METRIC_EVALUATION_LESSONS.md`：GT三指标与VBench正式经验和实测速度。
- `model_evaluation/frozen300_20261113/`：300条冻结输入、GT及A849原始输出。
- `model_evaluation/ltx25_distilled_v7_300/`：蒸馏8步无LoRA基座输出。
- `model_evaluation/frozen300_gt_metrics_20261115/`、`frozen300_vbench_20261115/`：四模型正式GT／VBench，后者最终不做静态筛选。
- `model_evaluation/frozen300_data_ablation_20261115/`：纯合成step600、纯真实step232；已完成5400项新增分数。205167已释放，单次CPU节点并行授权不改变login默认规则。
- `model_evaluation/a849_external_vbench_v4_116/`：最终V4的464输出、2784项六维分数；旧V1／V2／V3为历史。
- `model_evaluation/ds_full_eval_20261115/`：正式DeepSeek VLM，最终4748项含重复评分；固定Prompt和校准／非校准分表分开保留。
- `model_evaluation/inference_models/`、`model_evaluation/ltx25_vace_a100_20261111/`：保留可复用模型入口、runtime、cache和conditions。
- `model_evaluation/ltx25_action_full121_20261112/`：整体保护，不纳入普通副本清理。
- `retirement_lance_goku_20261112/`：退役任务最小结论和交付审计，不是当前执行入口。

Lance/GOKU 的训练、评测、消融、真实人物压力测试和源码已于 2026-11-12 退役；不要恢复为当前入口。最小删除清单、关键结论和百度交付凭据位于 `retirement_lance_goku_20261112/`。

历史审核包的远端目录约定为 `/apps/bypy/lance_review/`；是否上传以当前任务授权为准。Lance/GOKU 本地任务已退役，历史交付只依据退役档案中的成功回执追溯。

## Git 与本次任务收尾

- Git 仓库根目录为 `video_collection/`，不要在上级共享工作目录重复初始化。源码、说明和提交脚本进入版本管理；媒体、生成清单、运行日志、模型、缓存、ZIP、凭据及第三方 runtime/vendor 保留在本地且由 `.gitignore` 排除。
- 当前 11K 数据只做本地交付，不再制作新的评分审核包、执行独立全量验收或上传；历史 50 对无评分滚动审核 ZIP 和上传收据保留，不代表全量数据已经上传。
- 正式数据入口和 `media/` 不能因目录清理而失效。11K 的所有 canonical 媒体路径已内部化为 `media/...`；可重建 smoke、副本和暂存可清理，冻结来源、历史审核 selection、SHA sidecar 和回执应保留。
- 11K 的 CPU 相似度评分仍只复用 `dataset_research/tripvvt_research/score_pair_similarity.py`，不要复制修改多个版本；CoinVE tar 通用读取器位于 `dataset_research/coinve_research/download_random_pairs.py`。

## 2026-11-12 阶段整理

- `ltx25_vace_a100_20261111/README.md` 是当前阶段总结；目录名a100表示百样本，实测GPU为H200。旧运行流程保存在任务archive，包含已过期jobid。
- 本批输入/结果ZIP及其中已上传视频已删除，清理记录位于任务metadata；review只剩上传回执。metadata/deliveries中的HTML是无媒体快照，不能当作可播放审核包。
- 用户后续还需LTX2.5、VACE和师兄模型，保留本任务runtime、cache、conditions以及外部共享模型。其他历史任务的数据未经本次清理。

## 保守清理与交接（2026-11-21）

- 清理前记录精确路径、大小、依据和恢复方式，检查活动进程与正式清单引用；不按cache／review／tmp名称批量删除，不跟随符号链接清理共享外部目录。
- 有成功上传回执的ZIP、确认不被正式流程引用的可重建浏览／临时副本才进入本轮清理；无可靠依据的内容保留并记录原因。以既有上传工具回执为准，不增加远端回读或哈希核对。
- VLM实际送评卡片、参考帧、请求签名、原始响应，VBench评分投影，冻结输入／GT、参考、正式输出、分数和协议证据均保护。浏览副本与送评证据不能混为一类。
- root与测评README维护当前入口；原运行记录保持原样，必要时加“已完成／历史”说明。不得改写既有分数、Prompt或冻结清单。
- Git只纳管经审阅的源码、说明、依赖、固定Prompt和必要配置；运行结果与凭据留本地。历史删除按档案核对后提交，不用git add -A直接纳入混合工作区；推送只做正常快进或合并，不强推。
