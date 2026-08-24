# 人物视频编辑数据整合起步计划（历史方案）

> 状态：已被根目录 [`HANDOFF_LANCE_REMOVE_FINETUNE.md`](../HANDOFF_LANCE_REMOVE_FINETUNE.md) 的 InsAVE removal 小实验取代，不作为第一轮执行计划
>
> 日期：2026-08-24
> 起步范围：CoinVE-200K + GOKU-2M `remove` + Lance + OmniEdit-Bench Spatial

本文件保留较大规模数据整合、存储和评测设计供后续参考。当前不要实施 5K CoinVE + 5K GOKU；先完成 handoff 中约 500 条 InsAVE forward remove 的最小微调闭环。

## 1. 目标与起步范围

第一版只整合两个数据集：

- **CoinVE-200K**：筛选 5,000 个高质量人物相关 multi-task pair；
- **GOKU-2M `remove`**：筛选 5,000 个高质量人物相关 single-task pair；
- **训练基线**：使用当前已经跑通的 Lance，取得 DataMake/Lance 仓库路径后再生成专用输入格式；
- **评测**：使用 OmniEdit-Bench Spatial，对原始模型、CoinVE 微调模型、CoinVE+GOKU 混合微调模型进行统一评测。

不下载两个数据集全量。先按 metadata 筛候选，再分批精确拉取命中视频。整个 pilot 的原始数据、训练派生数据和评测结果暂以约 200 GB 为存储预算。

## 2. 存储设计

### 2.1 建议目录

```text
person_edit_pilot/
├── README.md
├── catalog/
│   ├── datasets.parquet
│   ├── assets.parquet
│   ├── samples.parquet
│   ├── instructions.parquet
│   ├── quality.parquet
│   └── splits.parquet
│
├── raw/
│   ├── coinve/
│   │   ├── videos/
│   │   │   ├── <sha256>.mp4
│   │   │   └── ...
│   │   └── masks/
│   │       ├── <sha256>.mp4
│   │       └── ...
│   └── goku_remove/
│       └── videos/
│           ├── <sha256>.mp4
│           └── ...
│
├── review/
│   ├── audit.csv
│   ├── contact_sheets/
│   └── rejection_report.csv
│
├── release/
│   └── v0.1/
│       ├── manifest.jsonl
│       ├── train-coinve-00000.tar
│       ├── train-goku-remove-00000.tar
│       ├── val-00000.tar
│       ├── test-00000.tar
│       └── SHA256SUMS
│
├── adapters/
│   └── lance/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── dataset_config.yaml
│
└── benchmark/
    └── omniedit_spatial/
        ├── source/
        ├── outputs/
        │   ├── lance_base/
        │   ├── lance_coinve/
        │   └── lance_coinve_goku/
        └── results/
```

`raw/` 只保存一份原始 MP4，以 SHA256 命名去重。官方文件路径保存在 `assets.parquet`，不依靠目录表达配对关系。筛选中的视频先进入临时 staging，只有通过的资产才进入 `raw/`。

### 2.2 最小 ER 数据模型

起步阶段不引入数据库，使用 Parquet 表保存关系，CSV 用于人工审核，JSONL 只作为训练 loader 的导出视图。

| 表 | 主键 | 作用 |
| --- | --- | --- |
| `datasets.parquet` | `dataset_id` | 官方仓库、revision、许可和来源信息 |
| `assets.parquet` | `asset_id` | SHA256、官方路径、本地路径、文件角色和技术规格 |
| `samples.parquet` | `sample_id` | source/target 外键、监督类型、任务、人物相关性和状态 |
| `instructions.parquet` | `sample_id + step_index` | 每一步指令、operation、object scope 和人物细分类别 |
| `quality.parquet` | `sample_id` | 自动检查、人工审核、最终决策和拒绝原因 |
| `splits.parquet` | `sample_id` | split 版本、group id 和 train/validation/test 归属 |

实体关系为：一个 dataset 对应多个 assets 和 samples；一个 sample 引用 source/target asset，并对应一条或多条 instruction；quality 和 split 分别与 sample 一一关联。

### 2.3 CoinVE 转换示例

CoinVE 官方记录：

```json
{
  "source_video_path": "src_videos/src_video_050/xxx.mp4",
  "edited_video_path": "tgt_videos/tgt_video_037/xxx_edit.mp4",
  "instruction": [
    "Add black headphones to the woman's head.",
    "Replace her red top with a white shirt.",
    "Replace the background with an office."
  ],
  "instruction_operation": ["Add", "Replace", "Replace"],
  "instruction_object": ["subject", "subject", "background"]
}
```

整合后：

```text
samples.parquet
sample_id       source_asset_id  target_asset_id  supervision_type
coinve:000001   sha256:a1...     sha256:b2...     multi_edit_pair
```

```text
instructions.parquet
sample_id       step  operation  person_category  text
coinve:000001   0     add        accessory        Add black headphones...
coinve:000001   1     replace    clothing         Replace her red top...
coinve:000001   2     replace    none             Replace the background...
```

CoinVE 的多条指令必须作为一个有序组合保留，不能伪造为多个 single-task pair。训练时生成带编号的组合 prompt。GOKU `remove` 只有一条 instruction，标记为 `atomic_edit_pair`。

CoinVE mask 作为可选 asset 保留引用，但第一轮公共训练接口只要求 source、instruction 和 target，避免因为 GOKU 没有等价 mask 而引入不一致监督。

### 2.4 训练存储与传输

- 原始 MP4 保持官方分辨率和编码，不覆盖、不提前降采样；
- Lance 所需分辨率、帧数和 fps 作为独立派生 release 生成；
- 训练文件使用不再压缩的 1–2 GiB tar shard；MP4 已经压缩，再套 ZIP 收益小且不利于流式读取；
- CoinVE 和 GOKU 分开打 shard，使 sampler 能独立控制混合比例；
- 传输时先传 `manifest.jsonl` 和 `SHA256SUMS`，训练任务只复制本轮需要的 shard 到作业节点本地临时盘；
- 每次筛选规则、split 或训练规格变化都生成新 release 版本，不原地修改旧版本；
- 下载前先用 500 对估算平均体积；预计超过 200 GB 时停止扩量并报告，不静默压缩或删除已接收的原始视频。

## 3. 筛选设计

### 3.1 人物集中筛选

两个数据集都先筛成 `person_focus=true`，使用统一人物类别：

```text
identity_face
skin_tone
age
hair
facial_hair
accessory_jewelry
clothing
expression
body_motion
held_object
whole_person
```

筛选原则：

- CoinVE 至少有一条 instruction 真正编辑人物或人物所穿、所戴、所持内容；官方 `subject` 不能直接等同人物，必须结合 instruction 判断；
- GOKU 只保留删除人物、人物细节、穿戴物或手持物的样本；
- 肤色、年龄、表情和动作不设置虚假配额；若没有足够数据，在覆盖报告中明确记为缺口；
- 服装等高频类别不需要物理删除，在训练 sampler 中降权；稀有人物细节使用反频率采样提高出现概率；
- CoinVE 中虽然只要一项人物编辑即可进入候选，但其余 background/object 指令仍完整保留并参与质量审核。

### 3.2 不合格 pair 的门槛

#### 技术硬门槛

- source、target 都能完整解码；
- 分辨率一致，fps 差不超过 0.01；
- frame count 完全一致；
- 达到 Lance 最低训练帧数；
- 文件哈希不重复，不是静态图伪装视频。

#### 内容质量门槛

- instruction 的前提在 source 中真实存在；
- target 完成全部 instruction；
- 非编辑区域、人物身份和动作在没有相关指令时得到保持；
- 删除穿戴物或手持物后，身体和交互逻辑合理；
- 无明显闪烁、漂移、遮罩边缘、人物形变和局部模糊。

#### 固定拒绝原因

```text
decode_error
spec_mismatch
instruction_premise_missing
instruction_not_executed
edit_leakage
interaction_invalid
temporal_artifact
visual_artifact
duplicate
license_excluded
```

#### 实施顺序

1. 先用 metadata 和规则过滤；
2. 对候选 pair 提取同步全局帧和人物/面部局部帧，使用视觉模型输出结构化质量分数；
3. 样本状态只使用 `accepted / rejected / review`；
4. 先人工审核 CoinVE、GOKU 各 50 条校准自动阈值，以 accepted 的人工严格通过率不低于 90% 为目标；
5. 按每库 1,000 条一批获取和筛查，直到各有 5,000 条 accepted；
6. release 前再从 accepted 中随机复核每库 100 条；若严格通过率低于 90%，收紧阈值并重新生成 release；
7. 人工 audit 沿用项目的 `0/1/2 = 不通过/通过/存疑` 规则，生成时默认填 `1`，且脚本不得默认覆盖已有 audit。

## 4. 划分和训练组合

### 4.1 数据划分

- 先划分，再转码和打 shard；
- 按原始视频 lineage 分组：CoinVE 同一个基础 source、GOKU 同一个原始 clip 必须进入同一 split；
- 使用哈希和视频指纹检查两个训练数据集之间及其与 benchmark 之间的近重复；
- group-level 划分为 `train 90% / validation 5% / internal test 5%`；
- 划分时尽量保持 dataset、人物类别和 instruction 数量分布；
- 物理存储不按任务复制视频，任务组织由 metadata 和 sampler 完成。

### 4.2 起步微调实验

固定同一个 Lance 初始 checkpoint、训练步数、随机种子、输入规格和优化器，只改变数据：

- **E0：Lance base**，不微调；
- **E1：CoinVE-person 5K**，观察人物 multi-task 数据是否带来收益；
- **E2：CoinVE-person 5K + GOKU-person-remove 5K**，训练时两个数据集按 1:1 采样。

E1 和 E2 使用相同训练 step 数，避免 E2 因看到更多 batch 而天然占优。E2 内部对人物细分类别使用反频率采样，但不改变两个数据集 1:1 的上层比例。

第一轮不采用课程学习，避免把数据作用和训练顺序混在一起。如果 E2 相比 E1 明显提高 remove、却损害组合指令，再在下一阶段实验 atomic→multi 或 multi→atomic 的 curriculum。

## 5. Benchmark 与验收

### 5.1 OmniEdit-Bench

OmniEdit-Bench 共 790 项，分 Spatial、Temporal、Reasoning、Audio 和 Reference 五条 track。第一轮只使用 **Spatial 240 项**，覆盖 object add/remove/replace、appearance、color、clothes 等任务，多数视频为 720p，和当前 pilot 最匹配。[官方数据卡](https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench)

官方评审同时观察 source、编辑结果和 instruction，从 accuracy、preservation、realism、consistency 四个维度打 0–5 分，并让 accuracy 占主要权重。输出文件按 benchmark `video_id` 命名。

除了完整 Spatial 分数，在运行模型前还应根据 source 和 instruction 冻结一个人物相关 slice。该 slice 只作为项目内部诊断，不能冒充完整官方 benchmark 分数。

### 5.2 Gemini 评审成本

官方脚本当前调用 Gemini 3.1 Pro，每项同时上传约 10 秒 source 和 10 秒 edited video。[官方评测脚本](https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench/blob/main/gemini_eval.py)

Gemini 视频约消耗每秒 100–300 个输入 token；Gemini 3.1 Pro 标准价为输入 2 美元/百万 token、输出及思考 12 美元/百万 token。[视频 token 说明](https://ai.google.dev/gemini-api/docs/video-understanding)｜[Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing)

按 Spatial 240 项估算：

- 每个模型约 2–7 美元；
- base、E1、E2 三个模型约 6–21 美元；
- 为重试和实际 thinking token 留余量，首轮预算按 10–30 美元；
- 正式全跑前先用 20 项调用 `countTokens` 并记录真实 usage，再修正预算。

上述费用只包括自动评审 API，不包括三个模型生成 benchmark 结果所需的 GPU 计算。

### 5.3 GPT API 能否替代

可以用 GPT API 做辅助评审，但不能直接照搬官方 OmniEdit-Bench 脚本。OpenAI 官方文档显示当前 GPT-5.6 系列支持图像输入、**不支持视频输入**；因此必须先将 source 和 edited video 提取为同步帧或 contact sheet，再提交 Responses API。[OpenAI 模型文档](https://developers.openai.com/api/docs/models/gpt-5.6-luna)

这会带来两个限制：

- 帧评审不能完整观察闪烁和快速运动；
- 得到的分数不再与 OmniEdit-Bench 官方 Gemini 结果严格可比。

因此第一轮采用 **Gemini 官方评审作为主结果，GPT 帧评审只作为人物细节的辅助交叉检查**。先对 20 条样本做 Gemini、GPT、人工三方一致性测试，再决定是否扩大 GPT 评审。

### 5.4 验收条件

- 10,000 个 release 样本全部能被 Lance loader 读取；
- 随机读取 100 个 tar 内样本，source、instruction、target 对应正确；
- split 间不存在同源或近重复视频；
- E0、E1、E2 在 OmniEdit-Bench Spatial 上均生成完整结果；
- 同时报告 accuracy、preservation、realism、consistency、总分和人物 slice；
- 若微调没有提升，也视为有效实验结果，下一轮从类别覆盖、过滤精度或混合比例中定位原因，不立即增加更多数据集。

## 6. 实施阶段

### 阶段 A：最小数据层

1. 创建浅层目录和六张最小 Parquet 表的 schema；
2. 从现有 CoinVE metadata 与 GOKU `remove` metadata 各导入 100 条候选；
3. 只复用现有已下载样本或少量精确拉取，不扩量；
4. 跑通 metadata → asset → sample → instruction → quality → split；
5. 生成一个不超过 200 对的 `v0.0-smoke` release，并验证读取。

### 阶段 B：筛选校准

1. 建立人物类别分类和固定拒绝原因；
2. 生成人工 audit 与可观察 MP4/contact sheet；
3. 两库各审核 50 条，校准自动筛选；
4. 验证 accepted 严格通过率达到 90%。

### 阶段 C：扩至 5K + 5K

1. 每库每批 1,000 条迭代筛选；
2. accepted 进入 canonical raw，rejected 只留 metadata、来源和拒绝理由；
3. 达到每库 5,000 条后冻结 split；
4. 生成 `release/v0.1`、校验和、统计报告与 200 条 release 复核。

### 阶段 D：Lance 与评测

1. 用户提供 DataMake/Lance 仓库路径；
2. 读取真实 loader 和配置，生成 Lance adapter，不猜测训练字段；
3. 完成 E0、E1、E2；
4. 先跑 20 项 OmniEdit cost/评审校准，再跑 Spatial 240；
5. 输出总体、分类、人物 slice 和人工观察报告。

---

# 新 Session 实施提示词

以下提示词按阶段使用。推荐一次只执行一个阶段，验收后再开下一阶段或发送下一段提示词。

## 提示词 A：搭建最小数据层

```text
请在 /scratch/prj0000000275/jcy/ljy 中开始实施人物视频编辑数据整合计划。

开始前必须完整阅读：
1. video_collection/AGENTS.md
2. video_collection/任务描述.md
3. video_collection/DATASET_AUDIT_GUIDE.md
4. video_collection/dataset_research/dataset_integration_pilot_plan.md

本次只实施计划中的“阶段 A：最小数据层”，不要提前扩量到 5K，不要下载完整数据集，不要开始微调。

具体目标：
- 在 video_collection 下建立 person_edit_pilot 的最小浅层目录；
- 先定义并实现 datasets、assets、samples、instructions、quality、splits 六张 Parquet 表的最小 schema；
- 使用现有 CoinVE metadata、现有 GOKU remove metadata 和已下载样本，各导入最多 100 个候选；
- CoinVE 的多条 instruction 必须按 step_index 保持顺序，不能拆成伪 single-task pair；
- 不引入数据库；
- 不覆盖现有 audit 或调研文件；
- 生成不超过 200 对的 v0.0-smoke manifest/release，并实现可重复的校验脚本；
- 用自动测试验证主外键、文件存在性、哈希、split group 隔离和训练样本读取。

先检查现有目录、脚本和依赖，复用已有 CoinVE/GOKU 下载与审核逻辑。实现过程中使用 apply_patch 编辑文件，保存明确的 README 和运行命令。完成后汇报实际导入数量、磁盘占用、测试结果、未解决问题和进入阶段 B 前需要我确认的事项。
```

## 提示词 B：筛选校准并扩量

```text
继续实施 video_collection/dataset_research/dataset_integration_pilot_plan.md。

开始前重新阅读 AGENTS.md、任务描述.md、DATASET_AUDIT_GUIDE.md 和当前 person_edit_pilot README，并检查阶段 A 的真实产物与测试；不要假设阶段 A 已正确完成。

本次实施阶段 B，并在筛选精度达标后进入阶段 C：
- 建立统一人物类别 person_focus taxonomy；
- CoinVE 只保留至少一条真实人物/穿戴/手持编辑指令的候选，但必须保留完整有序 multi-task instruction；
- GOKU 只保留人物、人物细节、穿戴物或手持物的 remove；
- 实现技术硬过滤、固定 reject reason、accepted/rejected/review 状态；
- 生成符合 DATASET_AUDIT_GUIDE 的 audit、H.264 MP4 对照和 contact sheet，默认审核值为 1，不覆盖已有 audit；
- 先对两库各 50 条进行校准。需要我观看或填写 audit 时停在这里并给出文件路径，不要用默认值冒充人工结果；
- 人工确认 accepted 严格通过率达到 90% 后，才按每库 1,000 条的批次继续；
- 使用 metadata 先筛选，再精确拉取命中成员，不下载完整 CoinVE/GOKU；
- 最终目标为 CoinVE accepted 5,000、GOKU remove accepted 5,000；
- 先用 500 对估算体积，整个 pilot 预计超过 200 GB 时停止并汇报；
- 达标后按 lineage group 做 90/5/5 split，检查跨 split、跨数据集及 benchmark 近重复，生成 release/v0.1、tar shards 和 SHA256SUMS。

每个批次都记录候选数、下载量、各拒绝原因、人物类别分布和磁盘占用。不要删除现有用户文件；临时 staging 只在确认 accepted 已入库、metadata 已落盘和校验通过后清理。
```

## 提示词 C：接入 Lance 并微调

```text
请基于 video_collection/dataset_research/dataset_integration_pilot_plan.md 和已完成的 person_edit_pilot/release/v0.1 接入 Lance。

DataMake/Lance 仓库路径是：<在这里填写路径>

开始前完整阅读 video_collection/AGENTS.md、任务描述.md、整合计划、person_edit_pilot README，以及 DataMake/Lance 仓库中适用的 AGENTS.md、训练入口、dataset loader、配置和已有示例。

要求：
- 先从真实 loader 推导输入格式，不猜测字段；
- canonical raw 和 Parquet catalog 不改动，只在 adapters/lance 生成训练视图；
- 先用 16–100 条样本做 loader 和单 batch smoke test；
- 固定同一个初始 checkpoint、训练步数、随机种子、输入规格和优化器；
- 运行或生成可复现配置：E0 Lance base、E1 CoinVE-person 5K、E2 CoinVE-person 5K + GOKU-person-remove 5K；
- E2 的 dataset sampler 为 1:1，E1 与 E2 训练 step 数相同；
- 不在首轮引入 curriculum；
- 保存配置、数据版本、代码 revision、日志和 checkpoint 对应关系。

若实际训练需要提交昂贵的长任务，先完成 loader/single-batch smoke 和资源估算，再向我报告预计 GPU、时间和磁盘需求；不要在没有说明的情况下直接启动大规模训练。
```

## 提示词 D：运行 OmniEdit-Bench

```text
请按 video_collection/dataset_research/dataset_integration_pilot_plan.md 对 E0、E1、E2 运行 OmniEdit-Bench Spatial 评测。

开始前检查三个模型 checkpoint、Lance 推理入口、OmniEdit-Bench 官方数据与当前官方评测脚本。只使用 Spatial 240，不运行 Audio、Reference、Reasoning 或完整 Temporal。

要求：
- 先按官方 video_id 生成三个模型的输出目录，并验证分辨率、时长、解码和缺失项；
- 在查看模型结果前，仅根据 source 和 instruction 冻结人物相关 slice；
- 先选择 20 条做 Gemini 3.1 Pro cost calibration，记录 countTokens、实际 usage、失败重试和单位样本费用；
- API key 只从环境变量读取，严禁写入代码、配置、日志或 Git；
- 20 条校准没有明显异常且预算仍在计划范围内后，再跑 Spatial 240；
- 按 accuracy、preservation、realism、consistency、overall、官方分类和人物 slice 比较 E0/E1/E2；
- 从人物 slice 做盲化人工复核，并记录自动评审与人工判断不一致的案例；
- GPT API 如需使用，只能以同步抽帧/contact sheet 方式作为辅助评审，结果与 Gemini 官方分数分开报告，不能声称为官方 OmniEdit-Bench 分数。

完成后生成一个简洁 Markdown 报告，包含实验配置、完整率、费用、总体与分类结果、人物案例、失败模式和下一轮是否应调整筛选或混合比例。
```
