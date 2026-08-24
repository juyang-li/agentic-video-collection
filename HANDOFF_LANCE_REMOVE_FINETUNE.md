# Lance removal 初步微调 handoff

> 更新日期：2026-08-24
>
> 当前目标：用少量高质量 removal 监督验证 Lance 是否值得继续作为项目基模
>
> 第一轮主数据：InsAVE-80K `add_and_remove` 的 forward remove
>
> 第一轮规模：约 500 个通过筛选的 pair；链路稳定后最多扩到 1,000

## 新 session 必须先读

开始修改或运行任何内容前，依次完整阅读：

1. [`AGENTS.md`](./AGENTS.md)
2. [`任务描述.md`](./任务描述.md)
3. [`DATASET_AUDIT_GUIDE.md`](./DATASET_AUDIT_GUIDE.md)
4. 本文件
5. [`model_evaluation/lance_base/README.md`](./model_evaluation/lance_base/README.md)
6. [`lance_base_evaluation_plan.md`](./lance_base_evaluation_plan.md)
7. [`dataset_research/insave_research/README.md`](./dataset_research/insave_research/README.md)
8. [`dataset_research/insave_research/insave_report.md`](./dataset_research/insave_research/insave_report.md)
9. [`dataset_research/dataset_recommendations.md`](./dataset_research/dataset_recommendations.md)

准备训练时，还必须读取实际 Lance checkout 中的 `README.md`、`train.md`、`train_dataset.md`、训练入口、dataset loader 和配置。若进入 DataMake 或其他仓库，先查找并阅读该目录适用的 `AGENTS.md`。不要只依据本 handoff 猜训练接口。

旧的 [`dataset_integration_pilot_plan.md`](./dataset_research/dataset_integration_pilot_plan.md) 记录了较大的 CoinVE+GOKU 方案，已不是第一轮执行计划；其中存储、去重和审核原则仍可参考，但不要执行其 5K+5K 扩量方案。

## 已确认的基线

Lance 原始模型已经在独立干净 checkout 上跑通，不需要重新部署或下载权重：

```text
代码：model_evaluation/lance_base/runtime/upstream/
commit：9717d9160ed16fedfdad5ce39ccbc6c9d69695e0
环境：/scratch/prj0000000275/jcy/.conda/envs/xks_lance
权重：/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/models/Lance
模型：Lance_3B_Video
GPU：已在单张 H200 验证
```

`model_evaluation/lance_base/` 是不可随意改写的 base baseline：

- smoke 1/1 成功；
- OmniEdit-Bench Spatial 固定 15 条，15/15 生成并完整解码；
- 参数为 30 steps、seed 42、CFG 4.0、50 requested frames、`video_480p`、prompt enhancement 关闭；
- 原始输出在本地 `outputs_raw/`，左右 H.264 观察视频在本地 `comparison/`；二进制产物被 Git 忽略，但生成脚本、清单和报告受版本控制；
- 人工 `scores.csv` 尚未正式填写，当前结论来自稀疏帧初查。

15 条初步表现：

| 类别 | 初步表现 |
| --- | --- |
| add | 3/3 明显完成 |
| remove | 3/3 均未明显完成，最确定的系统性弱项 |
| replace | 2 条完成、1 条部分完成 |
| appearance | 年龄/换脸均有明显问题 |
| clothes | 2 条完成、1 条部分完成 |

因此第一轮只回答一个问题：

> Lance 的 removal 能否通过约 500 条高质量监督得到明显改善，同时不破坏原有能力？

## 第一轮数据决定

只使用 InsAVE `add_and_remove` 的 forward 方向：

```text
original_video + instruction(remove) -> target_video
```

官方 metadata 字段为：

```text
original_video
target_video
instruction
instruction_reverse
```

`add_and_remove` 共 21,377 个 pair；约 21,374 条 forward instruction 以 remove 开头。固定调研 revision：

```text
suimu/InsAVE-80K@8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289
```

选择依据：forward remove 随机 15 条中 instruction 15/15 通过、严格七项 13/15 通过，而且 15/15 都删除整个人物。它适合验证实体/整人 removal，但不能补 replace，也不能代表肤色、年龄、发型、配饰等细粒度人物编辑。

第一轮明确不使用：

- `instruction_reverse` 和 reverse add；
- `clone_id`、`clone_voice`、`clone_id_voice`；
- `general_editing`；
- GOKU、CoinVE、OpenVE 或其他数据集。

音轨不参与训练和筛选：loader 只读取视频帧，manifest 记录 `audio_policy: ignored`；不需要预先为了去音轨而重编码。不得把 reverse instruction 中的 `<S>...<E>` 语音内容混入本轮。

## 数据筛选与隔离

目标是约 500 个最终 accepted pair，不是从完整 21,377 条直接训练。第一版不部署 GPT/Gemini；先使用 metadata 规则、完整解码检查和小规模人工视频审核。

保留条件：

1. 方向必须是 `original_video -> target_video`；
2. instruction 是单一、明确、画面中可观察的 remove；
3. source 中目标确实存在，target 中目标确实消失；
4. 排除残影、黑洞、大面积模糊、明显修补边界和严重交互逻辑错误；
5. source/target 均完整解码，帧数、FPS、分辨率和时长可对齐；
6. 按原始视频、人物或场景 group 划分 train/validation，不能按 pair 随机拆分；
7. 当前 InsAVE 调研的 15 条 remove 和所有最终评测样本必须进入 exclusion list，不得参与训练；
8. OmniEdit-Bench 15 条是纯评测数据，严禁进入训练。

人工审核遵循 `DATASET_AUDIT_GUIDE.md`：`1/0/2 = 通过/不通过/存疑`，新表默认 1，脚本不得默认覆盖已填写 audit。

## 新 session 的最小执行顺序

### 1. 先确认训练接口

- 保持 `model_evaluation/lance_base/` 不变，另建浅层的 removal pilot 训练目录；
- 从真实训练入口和 loader 推导所需格式，不猜 JSON/Parquet 字段；
- 确认哪些参数训练、VAE 编解码器如何冻结、checkpoint 如何保存和恢复；
- 当前共识是 latent flow-matching loss 不需要经过 decoder，冻结 VAE 后训练主干，但必须以实际代码为准；
- 先写一页最小训练计划和资源估算，再开始数据扩量或长任务。

### 2. 跑通小数据链路

- 从现有 InsAVE metadata 和获取脚本构造一个很小的 forward-remove manifest；
- 先用 16 条以内完成：下载/提取、技术检查、训练 adapter、loader、单 batch forward/backward；
- 验证 loss 有限、梯度只进入预期参数、显存和 checkpoint 正常；
- 这一步不以画质提升为目标，也不要直接启动 500 条长训练。

### 3. 冻结约 500 条训练数据

- metadata 预筛后精确获取候选，不下载完整 InsAVE；
- 生成人工可看的 H.264 左右 comparison 和 audit；
- 需要用户人工审核时停下来给出路径，不得把默认值冒充人工结果；
- 冻结 accepted manifest、group split、exclusion list、数据 revision 和文件哈希；
- 不引入数据库，不提前构建大而全的数据 schema。

### 4. 做短程微调

- 固定 base checkpoint、seed、数据版本和配置；
- 先跑短程、小数据训练并保存完整运行记录；
- 不在第一轮加入 curriculum、多数据集混合、LLM 筛选或复杂 sampler；
- 训练 loss 下降不等于 removal 已学会，必须回到固定视频评测。

### 5. 固定评测

评测包含：

1. 当前 OmniEdit 15 条中的 3 条 remove，观察跨数据集迁移；
2. 另外冻结 15–30 条与训练 group 隔离的人物 remove，观察目标域提升；
3. 当前其余 12 条 add/replace/appearance/clothes，检查灾难性遗忘。

所有 pre/post 结果生成 H.264 左右 comparison：左 source、右 output，同名 TXT 保存 instruction。模型原始输出必须保留，comparison 只用于观察。

继续扩量的条件：removal 明显改善，非编辑区域和时序没有明显下降，原有 add/replace/clothes 没有大幅退化。若 loss 下降但 removal 行为几乎不变，停止扩量，优先检查 conditioning、训练目标或可训练参数。

## 后续数据集边界

- **GOKU remove**：第二阶段扩充。随机 15 条 instruction 13/15、严格六项 9/15；类型更多，但人物相关性和交互逻辑弱于 InsAVE。
- **CoinVE-200K**：2–5 个有序编辑、只有最终 target；必须保持 `multi_edit_pair`，不能拆成伪 single-task。后续用于组合能力或缓解偏科。
- **OpenVE-3M**：人物局部属性覆盖好，但重复、正反 pair、prompt 改写和错误较多；先按无向 pair 去重再考虑。

第一轮未证明 InsAVE remove 有效前，不引入以上数据集。

## Git 与本地大文件

Git 只保存 Markdown、脚本、配置、小型 CSV/JSON 清单和审核结论。以下内容由根目录 `.gitignore` 排除但可继续保留在当前机器：

- comparison MP4、contact sheet 和 ZIP；
- 原始数据、完整官方 metadata、tar shard；
- Lance clean runtime、模型权重、原始输出和日志；
- 未来 checkpoint、W&B 和训练运行目录。

不得把 Hugging Face token、Gemini/OpenAI key 或其他 credential 写入代码、配置、日志和 Git。需要授权时只使用环境变量或现有 CLI 登录态。
