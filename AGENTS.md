# 项目协作规则

## 集群分区硬性限制（用户于 2026-11-08 明确要求）

- **所有集群任务只能提交到 `testqueue` 分区。严禁提交到 `cpup`、`cpun` 或其他分区。** 此限制适用于数据准备、下载、CPU 评分、训练、推理、验收、打包、上传及续跑，不因任务不使用 GPU 而例外。
- 每次提交前检查脚本和实际提交参数，明确指定 `--partition=testqueue`；不得通过命令行覆盖为其他分区。运行入口应检查 `SLURM_JOB_PARTITION`，不等于 `testqueue` 时立即退出。
- **需要排队时就休眠等待，绝不为了绕过排队而换分区。** 默认沿用用户指定的 15 分钟检查节奏；异常修复后恢复等待。
- 发现本任务误入其他分区，应立即停止对应作业并取消其后续依赖，保留可续跑数据；不得继续运行或自行重投其他分区。

## 对话开始前

- 每次开始处理本项目的任务时，先完整阅读根目录下的 `任务描述.md`，再进行分析、讨论或修改。
- 进行数据集样本抽查、质量审核或编写相应脚本时，还要阅读根目录下的 `DATASET_AUDIT_GUIDE.md`。

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
- 审核包默认采用无评分表、完整展开的 scroll down 页面；只有明确要求评分时才生成 audit。带评分表时各审核项默认 `1`、`reviewed=0`，不计作人工通过；`1` 表示通过、`0` 不通过、`2` 存疑。完整模式规范见 `DATASET_AUDIT_GUIDE.md`。

## 集群存储与大文件

- 当前项目目录位于 `/scratch` 的 WekaFS 分布式网络文件系统上；`/home`、`/apps` 和 `/cm/shared` 也使用 WekaFS。当前没有发现单独挂载、面向用户长期归档的 NFS/SMB NAS。
- `/scratch` 用于保存项目数据真值、结构化元数据、正式 checkpoint 和评测产物，但不能默认视为有备份的长期归档；关键 final 权重还应复制到经确认的长期存储或外部网盘。
- `/raid` 是计算节点本地盘，不是跨节点共享的 NAS。不同节点的容量和内容可能不同，作业结束后也可能被清理，因此不得把不可重建数据或 final checkpoint 的唯一副本放在 `/raid`。
- 作业临时数据优先放在调度器提供的 `$TMPDIR`；当前环境的 `/tmp` 也可能映射到作业隔离的 `/raid/scratch/...`。不要直接把 `/raid` 根目录当作长期工作区。
- 适合暂存在 `$TMPDIR` 或作业本地 `/raid` 的内容包括：可重新下载的数据 shard、训练 Parquet 副本、解压视频、抽帧、latent/cache、模型读取副本、中间输出和临时 checkpoint。
- 推荐流程是“从 `/scratch` stage 到 `$TMPDIR` → 在本地盘训练或推理 → 定期并在作业结束前 rsync 回 `/scratch`”。回传后应检查文件完整性；重要权重归档时还应记录并核对 SHA256。
- 清理 checkpoint 前必须先盘点精确路径和文件组成，并区分推理权重与恢复状态：`model.safetensors` 可用于推理和从权重启动新实验，optimizer/scheduler/data cursor 仅在需要精确 resume 时保留。不得删除尚待评测的中间 checkpoint 或仍可能用于续训的 final checkpoint。

## 当前目录映射（2026-11-09）

```text
video_collection/
├── AGENTS.md                         # 本项目协作规则与目录映射
├── 任务描述.md                       # 项目背景、目标和数据职责
├── DATASET_AUDIT_GUIDE.md            # 人工抽查字段与填写规范
├── dataset_research/                 # 各外部数据集的调研与样本审核
│   ├── <dataset>_research/           # CoinVE、TripVVT、OpenVE、ReCo 等独立调研
│   └── coinve_tripvvt_11k_20261108/ # 已完成的本地 11K 数据
│       ├── README.md / HANDOFF_FOR_COLLEAGUE.txt
│       ├── manifest.jsonl / manifest.csv / crops.csv
│       ├── dataset_reader.py / read_example.py
│       ├── scripts/                 # 准备、预取、交接、滚动审核、可选检查、测试
│       ├── jobs/                    # 仅 testqueue 的 Slurm 入口
│       ├── media/                   # 原媒体；正式路径保持不变
│       ├── metadata/                # 冻结候选、来源、哈希、探测、评分、排除和状态
│       ├── logs/                    # 作业历史与重试记录
│       ├── reviews/                 # 已交付 ZIP、冻结抽样、SHA256 和回执
│       └── archive/                 # 历史过程说明，不是当前入口
├── lance_general_edit_finetune/      # 真实数据筛选、VLM 过滤与 Lance LoRA 微调
│   ├── 1_specialist_filtering/
│   ├── 2_vlm_filtering/
│   └── 3_lance_lora_finetuning/
└── model_evaluation/                 # 模型推理、对比评测与人工审核交付
    ├── lance_base/                   # Lance 基模评测与百度网盘核验工具
    ├── lance_tools/                  # 共用 Lance 评测工具
    ├── ivebench_lance/               # IVEBench 系列评测及便携浏览器运行时
    ├── omni_baselines_20260926/       # 冻结的 OmniEdit-48 对齐子集与基线
    ├── vace_coinve/                   # VACE/CoinVE 评测
    └── gtfresh1150_vs_coinve677_20260927/
        ├── README.md                  # 本次任务结论、经验和交付信息
        ├── configs/                   # 8 个正式 OmniEdit 分片配置
        ├── inference_checkpoints/     # CoinVE step677 的只读兼容推理入口
        ├── logs/                      # 正式单轮与多轮运行日志
        ├── metadata/                  # 冻结任务、来源凭据与完成状态
        ├── outputs_raw/               # 两模型共 96 个正式单轮输出
        ├── multiturn/                 # 两模型各 28 个多轮/诊断输出及 manifest
        ├── technical/                 # 8 个正式分片的技术检查 CSV
        ├── scripts/                   # 准备、推理、打包和浏览器验收脚本
        └── review/
            ├── omni48_multiturn_*_human_review_20260927.zip
            │                           # 已上传百度网盘的精确归档
            └── *.json                 # SHA-256 sidecar 与上传收据
```

历史审核包的远端目录约定为 `/apps/bypy/lance_review/`；是否上传以当前任务授权为准。`gtfresh1150_vs_coinve677_20260927/` 中不保留 smoke 临时产物、`*_pending.json`、Python 字节码缓存、runner 复制的 adapter 暂存副本或可由最终 ZIP 恢复的解压态审核副本。

## Git 与本次任务收尾

- Git 仓库根目录为 `video_collection/`，不要在上级共享工作目录重复初始化。源码、说明和提交脚本进入版本管理；媒体、生成清单、运行日志、模型、缓存、ZIP、凭据及第三方 runtime/vendor 保留在本地且由 `.gitignore` 排除。
- 当前 11K 数据只做本地交付，不再制作新的评分审核包、执行独立全量验收或上传；历史 50 对无评分滚动审核 ZIP 和上传收据保留，不代表全量数据已经上传。
- 正式数据入口、`media/` 和被引用的历史原素材不能因目录清理而失效。可重建 smoke、副本和暂存可清理，冻结来源、历史审核和回执应保留。
- 11K 的 CPU 相似度评分仍只复用 `dataset_research/tripvvt_research/score_pair_similarity.py`，不要复制修改多个版本；CoinVE tar 通用读取器位于 `dataset_research/coinve_research/download_random_pairs.py`。
