# Lance removal 双数据集微调计划

> 建立日期：2026-08-24  
> 状态：第一阶段技术 smoke 与第二阶段双 500-step 消融均已完成；固定 15 项评测未支持继续扩到 1K

## 1. 目标

融合 InsAVE-80K 与 GOKU-2M 的 removal 数据，验证 Lance 的 removal 弱项能否通过少量高质量监督得到明显改善，并同时建立可扩展到后续多数据集混合训练的数据链路。

本轮不是从零训练视频编辑模型，也不追求一次建立完整数据平台。核心问题是：

> Lance 能否学会更可靠的人物与物体 removal，同时不明显破坏已有 add、replace、appearance 和 clothes 能力？

两个数据集承担不同作用：

- **InsAVE forward remove**：质量较稳定，主要提供整个人物删除监督；只使用 `original_video + instruction -> target_video`，不使用 reverse add 或音频条件。
- **GOKU remove**：补充衣物、配饰、手持物、前景物体和背景物体等更广的 removal 分布；接入前需要更严格检查目标存在性、交互逻辑和修补质量。

音轨不参与训练，统一记录为 `audio_policy: ignored`。视频 loader 只读取视觉帧，不为去音轨预先重编码。

## 2. 扩量路线

采用逐级门禁，不把 1,000 条视为最终上限：

| 阶段 | InsAVE | GOKU | 总量 | 目的 |
| --- | ---: | ---: | ---: | --- |
| 技术 smoke | 8 | 8 | 16 | 验证双数据源、混合采样和训练链路 |
| 首轮质量门禁 | 300 | 200 | 500 | 判断 removal 是否出现明确学习信号 |
| 第二级扩量 | 600 | 400 | 1,000 | 判断收益是否稳定、是否跨域泛化 |
| 后续正式扩量 | 约 60% | 约 40% | 3,000–5,000 | 仅在前两级通过后执行 |

Smoke 使用 8:8 是为了确保两个数据源都真正进入 loader，不代表正式训练比例。

正式 1,000 条中的 GOKU 数据暂按以下目标构成准备：

- 约 250 条人物相关 removal：整人、穿戴物、配饰和手持物；
- 约 150 条一般物体 removal：前景物体和背景物体。

具体构成以后续人工审核通过率为准，不用低质量样本强行凑数。

## 3. 数据组织

### 3.1 总体结构

使用“视频文件 + 极简 SQLite + 派生 Parquet”：

```text
视频文件 + removal.sqlite
        │
        ├── 导出 insave_train.parquet
        └── 导出 goku_train.parquet
                    │
                    └── Lance training loader
```

- 视频保存在文件系统，不作为 SQLite BLOB。
- SQLite 只连接数据集、instruction、source 和 target 路径。
- Parquet 是可重复生成的训练输入，不是数据源真值。
- InsAVE 与 GOKU 分别导出 Parquet，训练 YAML 用独立 dataset group 控制混合权重。

未来实际运行时使用以下浅层目录：

```text
lance_remove_pilot/
├── README.md
├── removal.sqlite             # 本地工作索引，不提交 Git
├── media/
│   ├── insave/
│   └── goku/
├── exports/                   # Parquet 与冻结 manifest
├── reviews/                   # comparison 与人工审核材料
├── scripts/
├── runtime/                   # 独立训练 checkout
└── runs/                      # 日志、checkpoint、推理输出
```

当前不提前创建空目录；在对应阶段需要时创建。

### 3.2 SQLite 最小 schema

初版只有两张表：

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    repo_id    TEXT NOT NULL,
    revision   TEXT NOT NULL
);

CREATE TABLE pairs (
    pair_id      TEXT PRIMARY KEY,
    dataset_id   INTEGER NOT NULL,
    official_id  TEXT NOT NULL,
    instruction  TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    target_path  TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    UNIQUE (dataset_id, official_id)
);
```

当前明确不加入：

- exclusion 表；
- split、group、status；
- FPS、宽高、帧数和时长；
- SHA256；
- person relevance、remove target type 和审核分数；
- reference image、mask 或 pose 字段；
- 训练混合权重。

现有 InsAVE 和 GOKU 各 15 条随机调研样本只是数据集质量调研，不作为训练排除项。正式评测数据不导入训练数据库；以后开始冻结 validation 时，再增加 split/group 设计。

需要人工审核时新增独立 `audits` 表，不把审核列提前塞进 `pairs`。需要单张额外条件时按语义增加 nullable 列，例如 `reference_image_path` 或 `mask_path`；只有一条 pair 对应多个同类条件时，才考虑单独的 conditions 子表。

### 3.3 文件命名

媒体文件使用包含数据集来源的稳定 ID：

```text
insave-001234_source.mp4
insave-001234_target.mp4
goku-kkuh7xijlqwkwx6j_source.mp4
goku-kkuh7xijlqwkwx6j_target.mp4
```

SQLite 中只保存相对于 `lance_remove_pilot/` 的相对路径，避免绑定当前机器的绝对路径。

### 3.4 Lance Parquet

当前 Lance v2v local loader 要求每行严格包含：

```text
caption: string
input_video_bytes: binary
output_video_bytes: binary
```

导出映射固定为：

```text
caption            = instruction
input_video_bytes  = source video
output_video_bytes = target video
```

数据库新增 reference image 或 mask 路径，不代表 Lance 会自动使用它。接入此类条件时还必须单独扩展 Parquet schema 和训练 loader。

## 4. 全景执行计划

### 阶段 1：16 条双数据集技术 smoke

建立独立 runtime，完成数据获取、技术检查、SQLite 导入、双 Parquet 导出、混合 loader、单 batch forward/backward、梯度检查、checkpoint、resume 和微调 checkpoint 推理。

通过后才进入人工审核和数据扩量。

### 阶段 2：构建候选池

- InsAVE 只从 `add_and_remove` 选择 forward remove。
- GOKU 只从官方 `remove` task 选择独立 single-task pair。
- 先做 metadata 预筛和完整解码，不下载完整数据集。
- 技术规格写入自动生成的 `technical_inventory.csv`，不放入 SQLite。

### 阶段 3：人工审核与冻结 500 条

- 生成人工可看的 H.264 左右 comparison。
- 审核遵循 `1/0/2 = 通过/不通过/存疑`。
- 新审核项默认 1，但必须另有未审核状态，默认值不能冒充人工结果。
- 冻结通过样本、数据 revision、文件哈希和后续 group split。

### 阶段 4：首轮短程微调

- 从固定 base checkpoint 出发。
- 先采用 InsAVE:GOKU = 60:40 的主实验。
- 保留一个相同步数和 seed 的 InsAVE-only 小型对照，以区分数据融合收益与单纯增加更新步数。
- 不加入 curriculum、LLM 筛选、多任务混合或复杂 sampler。

### 阶段 5：扩到 1,000 条

只有 500 条实验满足以下条件才扩量：

- removal 有明确改善；
- 非编辑区域和时序没有明显下降；
- add、replace、appearance 和 clothes 没有大幅退化；
- GOKU 域内结果说明更广 removal 数据带来有效补充。

### 阶段 6：固定评测

分别报告：

1. OmniEdit 现有 3 条 removal，观察跨数据集迁移；
2. InsAVE holdout，观察整人删除；
3. GOKU holdout，分别观察人物/穿戴/手持物和一般物体；
4. OmniEdit 原有 12 条非 removal，检查灾难性遗忘。

所有 pre/post 输出保留模型原始 MP4，并生成 H.264 左右 comparison。训练 loss 下降不能代替视频评测。

### 阶段 7：Go/No-Go

- 1,000 条仍有稳定收益：扩到 3,000–5,000，并再讨论是否引入其他任务或数据集。
- loss 下降但 removal 行为不变：停止扩量，检查 conditioning、训练目标和可训练参数。
- removal 提升但其他能力明显退化：调整参数冻结范围、训练步数或后续混合数据，而不是继续增加单任务数据。

## 5. 第一阶段详细计划

### 5.1 固定 runtime 与输出边界

- Base baseline 保持在 `model_evaluation/lance_base/`，不写入训练产物。
- Pilot runtime 固定官方 commit：`9717d9160ed16fedfdad5ce39ccbc6c9d69695e0`。
- 复用环境：`/scratch/prj0000000275/jcy/.conda/envs/xks_lance`。
- 复用模型：`Lance_3B_Video`、Qwen2.5-VL ViT 和 Wan2.2 VAE。
- 若训练实现需要 patch，只改 pilot runtime，并保存最小 diff 和原因。

### 5.2 准备 8+8 smoke 数据

- 固定抽样 seed。
- InsAVE 读取 forward `instruction`，保持 original -> target。
- GOKU 读取 official remove instruction，保持 source -> edited。
- 检查 instruction 非空、source/target 路径存在且角色没有反转。
- source/target 均做完整解码检查；技术不通过的 pair 保留失败记录并替换。
- Smoke 数据只用于技术门禁，不宣称为最终人工 accepted 数据。

### 5.3 初始化 SQLite 与导出 Parquet

- 按最小 schema 创建数据库并写入两个 datasets 记录。
- 插入 16 个 pair，验证外键和 `(dataset_id, official_id)` 唯一约束。
- 分别导出 InsAVE/GOKU Parquet。
- 从 Parquet 反向读取 bytes，确认 input/output 与原始文件一致且没有交换。

### 5.4 Loader 语义检查

在训练前检查一个 packed batch：

- instruction 是文本条件；
- source 同时进入 ViT condition 和 clean VAE condition；
- target 进入带噪 latent 分支；
- `mse_loss_indexes` 只覆盖 target；
- source condition 不计算 MSE；
- 两个 dataset group 都被实际采样；
- 记录实际样本比例和 token 数，不能只相信 YAML weight。

### 5.5 Smoke 训练默认值

推荐初始设置：

```text
GPU: 1 x H200
resolution: official 480p bucket
sample_fps: 12
max_duration: 1 second（仅 smoke，保证双源样本可共同装入 65K token；后续训练恢复为 3 seconds）
total_steps: 2
save_every: 1
validation_step: -1
use_ema: false
freeze_vit: true
freeze_llm_embed_tokens: true
freeze_und_params: true
vae_cond_dropout_prob: 0
text_cond_dropout_prob: 0（仅 smoke，保证文本条件存在）
vit_cond_dropout_prob: 0（仅 smoke，保证 source 的 ViT/VAE 条件存在）
wandb: offline
```

第一轮冻结 understanding/shared 参数，训练 generation MoE、generation I/O 和 timestep 模块。官方 generation 脚本默认 `total_steps=500000`，不得原样启动；pilot wrapper 必须显式传入总步数。

### 5.6 四级门禁

1. **Loader-only**：连续读取两个数据源，无静默丢样本或角色反转。
2. **Forward-only**：MSE loss 存在且有限，无 CE loss、OOM 或 batch exception。
3. **两步 forward/backward**：gradient norm 有限且非零；冻结参数无梯度；预期 generation 参数发生更新。
4. **Checkpoint/resume/inference**：保存 step 1，完整恢复 optimizer/scheduler，再运行一步；微调 checkpoint 可被正式推理入口加载并生成完整可解码 MP4。

官方训练循环会捕获异常并继续，因此日志出现 `Done` 不算成功。外部 wrapper 必须检查实际 optimizer step、loss 数量、checkpoint 文件、resume step 和最终 MP4。

### 5.7 训练过程记录与图示

所有 smoke、短程微调和正式训练都必须同时保存三类过程记录：

1. **TensorBoard event**：逐 step 写入 total loss、MSE loss、学习率、gradient norm、样本数、MSE token 数、显存占用和 step wall time；日志保存在对应 run 的 `tensorboard/`。
2. **结构化 CSV**：相同指标同步写入 `metrics.csv`，每个数据源的实际入 batch 数量写入 `dataset_samples.csv`，作为不依赖 TensorBoard 的可审计原始记录；写入时逐行 flush，训练中断时仍保留已完成 step。
3. **静态曲线**：训练结束后从 `metrics.csv` 生成 `loss_curve.svg`，至少展示 total loss 与 MSE loss 随 optimizer step 的变化；正式结果报告必须嵌入或链接该图。

监控写入只由 global rank 0 执行。TensorBoard 和 CSV 的 step 必须使用真实 optimizer step，不记录被异常跳过的 batch；resume 后继续原 step 序列，不从 0 重新计数。训练正常结束或异常退出时均关闭 writer。

Smoke 阶段只有少量 step，曲线仅验证记录链路，不用于判断收敛。500/1,000 条训练的结果报告必须同时给出完整曲线和必要的局部放大图，不能只报告最终 loss。

## 6. 第一阶段完成条件

全部满足后才开始 500 条候选准备：

- 16 个 pair 的方向、路径和 Parquet schema 正确；
- 两个 dataset group 均进入训练，实际采样记录可追溯；
- target-only flow-matching loss 正确；
- 连续训练 step 的 loss 与梯度均有限；
- 梯度只进入预期参数；
- H200 有合理显存余量；
- checkpoint 可保存、恢复并用于推理；
- 推理视频完整解码；
- TensorBoard event 可被读取，`metrics.csv` step 连续且与日志一致，`dataset_samples.csv` 证明每个 smoke step 均包含两个数据源；
- `loss_curve.svg` 可打开，并包含 total loss 与 MSE loss；
- base baseline 没有变化；
- 所有代码 commit、数据 revision、seed 和配置有运行记录。

若出现 source/target 反转、条件/目标索引错误、冻结参数有梯度、loss 非有限、异常被循环吞掉或 checkpoint 无法推理，应停止扩量并先修复链路。

### 6.1 实际执行结果（2026-08-24）

数据链路结果：

| 项目 | 结果 |
| --- | --- |
| 官方分片 | InsAVE `12,254,924,800` bytes；GOKU `21,421,004,800` bytes |
| SQLite | 仅 `datasets`、`pairs` 两表；InsAVE 8 对、GOKU 8 对；外键检查为空 |
| 完整解码 | 32/32 个 source/target 视频通过；0 个失败 |
| Parquet | `insave_smoke.parquet` 8 行；`goku_smoke.parquet` 8 行；三列 schema 与 bytes 反向哈希一致 |
| Loader | 连续两个真实 packed batch 均包含双源；每批 5 个真实 pair、约 59.8K tokens |
| Loss 语义 | 每批 23,850 clean VAE condition tokens、23,850 target MSE tokens、0 CE target tokens |

训练使用单张 H200，首跑 step 0–1，再从 `checkpoints/0000001` 完整恢复模型、optimizer、scheduler 和 data cursor 执行 step 2：

| step | total/MSE loss | grad norm | InsAVE:GOKU | allocated MB | reserved MB | wall time s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.122307 | 0.205597 | 3:2 | 87,303 | 96,900 | 61.71 |
| 1 | 0.144532 | 0.174247 | 1:4 | 93,043 | 109,362 | 18.40 |
| 2（resume） | 0.119569 | 0.192895 | 3:2 | 87,305 | 108,830 | 22.11 |

训练循环报告 `completed_steps=3`、`exception_count=0`。TensorBoard 的 total/MSE loss 等指标为 step 0–2 连续序列，CSV 与 event 一致。图示和原始记录位于：

- [`runs/smoke_mixed/loss_curve.svg`](runs/smoke_mixed/loss_curve.svg)
- [`runs/smoke_mixed/metrics.csv`](runs/smoke_mixed/metrics.csv)
- [`runs/smoke_mixed/dataset_samples.csv`](runs/smoke_mixed/dataset_samples.csv)
- [`runs/smoke_mixed/tensorboard/`](runs/smoke_mixed/tensorboard/)

参数审计确认 513 个 trainable tensors 仅属于 generation MoE、time embedder、`vae2llm` 和 `llm2vae`。抽查的 generation 参数全部更新，UND 与 ViT frozen 参数逐值不变。最终 checkpoint 可被正式 `video_edit` 入口加载，并生成：

- [`runs/smoke_mixed/inference_step_0000002/000001.mp4`](runs/smoke_mixed/inference_step_0000002/000001.mp4)：848×480、12 FPS、41/41 帧完整解码、3.42 秒；
- [`runs/smoke_mixed/parameter_update_audit.json`](runs/smoke_mixed/parameter_update_audit.json)；
- [`runs/smoke_mixed/inference_step_0000002/decode_report.json`](runs/smoke_mixed/inference_step_0000002/decode_report.json)。

实现中确认单进程 `accelerate` 不会为上游训练入口设置 `RANK`，因此 pilot 训练 wrapper 固定使用单进程 `torchrun`。首轮结果中的 `metrics.csv:total_samples=6` 是 5 个真实 pair 加 1 个 flex pad slot；逐源 CSV 一直正确。监控实现已经修正为未来只记录 5 个真实样本。最终 full-state checkpoint 使用 FP32，单个模型约 28.4 GB、optimizer 约 22.2 GB；后续长程训练不能保持 smoke 的每步保存频率。

3 个 step 只证明数据、梯度、记录、保存、恢复和推理链路成立。loss 曲线点数太少，不用于判断收敛；生成视频也只做技术解码门禁，不据此宣称 removal 已改善。

## 7. 第一阶段执行边界

- 不部署数据库服务或引入数据库迁移框架；
- 不把 mask、reference image、pose 等未来条件提前加入 schema；
- 不修改 Lance base baseline；
- 不执行旧的 CoinVE+GOKU 5K+5K 方案。

## 8. 相关基线

- 项目交接：[`../HANDOFF_LANCE_REMOVE_FINETUNE.md`](../HANDOFF_LANCE_REMOVE_FINETUNE.md)
- Lance base 报告：[`../model_evaluation/lance_base/README.md`](../model_evaluation/lance_base/README.md)
- InsAVE 调研：[`../dataset_research/insave_research/insave_report.md`](../dataset_research/insave_research/insave_report.md)
- GOKU remove 调研：[`../dataset_research/goku_research/remove/goku_remove_report.md`](../dataset_research/goku_research/remove/goku_remove_report.md)
- 数据审核规范：[`../DATASET_AUDIT_GUIDE.md`](../DATASET_AUDIT_GUIDE.md)

## 9. 第一阶段可复现命令

以下命令均从项目根目录执行。正式运行前后都由脚本做存在性和非覆盖检查：

```bash
# 1. 获取两个固定 revision 的官方分片
python lance_remove_pilot/scripts/download_smoke_shards.py

# 2. 提取、完整解码、初始化 SQLite 并分别导出 Parquet
python lance_remove_pilot/scripts/prepare_smoke_data.py \
  --insave-tar lance_remove_pilot/media/_shards/train_data/add_and_remove/part_01.tar \
  --goku-tar lance_remove_pilot/media/_shards/remove/remove.videos.part32.tar

# 3. 用真实 Lance loader 连续审计两个 packed batch
python lance_remove_pilot/scripts/inspect_smoke_loader.py --batches 2

# 4. 首跑两个 optimizer step
TOTAL_STEPS=2 AUTO_RESUME=False \
  bash lance_remove_pilot/scripts/run_smoke_train.sh

# 5. 从 step 1 完整恢复 optimizer/scheduler，再跑 step 2
TOTAL_STEPS=3 AUTO_RESUME=True \
  bash lance_remove_pilot/scripts/run_smoke_train.sh

# 6. 严格验收训练记录、双源采样与最终 checkpoint
python lance_remove_pilot/scripts/verify_training_run.py \
  lance_remove_pilot/runs/smoke_mixed --expected-steps 3

# 7. 验证可训练范围、generation 参数更新与 frozen 参数稳定
python lance_remove_pilot/scripts/verify_parameter_updates.py \
  --base lance_remove_pilot/runtime/upstream/downloads/Lance_3B_Video/model.safetensors \
  --final lance_remove_pilot/runs/smoke_mixed/checkpoints/0000002/model.safetensors \
  --trainable-report lance_remove_pilot/runs/smoke_mixed/configs/trainable_parameters.txt \
  --output lance_remove_pilot/runs/smoke_mixed/parameter_update_audit.json

# 8. 用微调 checkpoint 生成并完整解码一个 removal MP4
bash lance_remove_pilot/scripts/run_smoke_inference.sh
```

TensorBoard 查看命令：

```bash
tensorboard --logdir lance_remove_pilot/runs/smoke_mixed/tensorboard
```

## 10. 第二阶段：500-pair 数据混合消融

本阶段用两个等规模、等步数实验回答“加入 GOKU removal 是否比纯 InsAVE 更有益”：

| 实验 | 数据 | optimizer steps | 其他训练设置 |
| --- | --- | ---: | --- |
| `mixed_300_200_steps500` | 300 InsAVE + 200 GOKU | 500 | 固定 base、seed、3 秒、12 FPS、LR `1e-5` |
| `insave_500_steps500` | 500 InsAVE | 500 | 与混合实验相同 |

混合组的 300 条 InsAVE 是对照组 500 条 InsAVE 的严格子集。混合采样权重固定为 0.6:0.4；不使用 smoke 的 mandatory 双源机制。两个实验都从同一个原始 Lance checkpoint 独立初始化，不从另一个实验继续训练。单张 H200 串行执行，step 249 和 499 保存 checkpoint。

冻结文件：

- `experiment_manifest_700.csv`：两组实验所需的 500 InsAVE + 200 GOKU 去重总目录；
- `mixed_300_200_manifest.csv`：混合实验 500 条；
- `insave_500_manifest.csv`：纯 InsAVE 对照 500 条；
- `experiment_data.json`：revision、seed、子集关系和审核状态；
- `configs/mixed_300_200.yaml`、`configs/insave_500.yaml`：训练数据配置。

这些样本通过官方 remove metadata 筛选、source/target 角色检查、逐文件完整解码和 pair 对齐检查，但尚未逐条完成人工画质审核。它们用于快速判断数据混合趋势，不得写成“500 条人工 accepted 高质量集”。最终质量结论仍需固定 pre/post 视频评测。

队列状态保存在 `runs/ablation_queue_status.json`，训练指标分别写入各 run 的 `metrics.csv`、`dataset_samples.csv` 和 `tensorboard/`。查看命令：

```bash
cat lance_remove_pilot/runs/ablation_queue_status.json
tail -n 30 lance_remove_pilot/runs/mixed_300_200_steps500/train.log
tensorboard --logdir lance_remove_pilot/runs
```

### 10.1 启动记录（2026-08-24）

- 700/700 pair、1,400/1,400 个视频通过完整解码；SQLite 外键检查为空；
- 混合与 InsAVE-only 配置分别通过独立 3-step 预检；
- 正式串行队列已启动，当前先运行 `mixed_300_200_steps500`，随后自动运行 `insave_500_steps500`；
- 正式混合实验前 5 步通过早期门禁：loss 均有限、gradient norm 均有限且非零、0 CE token、无训练异常；实际样本数为 InsAVE:GOKU = 3:2；
- 早期 loss 为 `0.114792, 0.097050, 0.067920, 0.101818, 0.098727`，峰值 reserved 显存约 101,326 MB。

这里的早期 loss 只证明训练稳定，不能用于判断混合数据优劣。必须等两个 500-step 实验结束后，在同一冻结评测集上比较 removal 质量，并同时检查非 removal 能力。

### 10.2 完成与评测结论（2026-08-25）

- 两个实验均完成 500/500 optimizer steps，训练异常数均为 0，final checkpoint 为各自的 `checkpoints/0000499`；
- mixed 实际采样为 InsAVE:GOKU = 307:193；两个数据源各自的前后窗口 loss 也均下降；
- 固定 15 项评测中，mixed 相对 base 没有观察到明确 `better`，InsAVE-only 有 2 项 `better`，其中只有 1 项是 removal；
- 两个模型都在 adding、replacement、appearance 或 clothes 上出现编辑抑制、目标延迟/消失和时序退化；
- 当前证据不支持“混合优于纯 InsAVE”，也不满足直接扩到 1,000 条的门禁。

loss 曲线、完整技术清单、逐项初步观察和四联视频见 [`../model_evaluation/lance_remove_ablation/`](../model_evaluation/lance_remove_ablation/README.md)。下一步应先评测 step 249，并增加 InsAVE/GOKU 各自的隔离 removal holdout，再决定是缩短训练、调整配方还是检查 conditioning/训练目标。
