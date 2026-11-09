# 第零级筛选：Metadata filtering（v1.2）

> 更新日期：2026-08-30  
> 状态：已完成；结果输入 [`1_specialist_filtering/`](./1_specialist_filtering/README.md)。

## 1. 总体判断

从“强化 removal”改成“优先质量，并补齐主要编辑门类”是正确方向。此前的问题不只是 GOKU 与 InsAVE 的比例，而是训练分布几乎只有 removal；即使 removal loss 下降，模型也没有持续看到 add、change 和 replace，原有能力自然容易退化。

数据来源的选择单位应是 **dataset × 官方分支 × 编辑方向 × 任务格**，而不是整个 dataset。例如，GOKU `remove` 的抽查结果明显好于 `swap_alter`；这只能支持使用 GOKU 的 `remove`，不能支持把 GOKU 全部任务一起导入。OpenVE 同理，各分支必须分别抽查和决定。

**2026-08-30 决策更新：InsAVE `add_and_remove` forward 的 `person remove` 整个分支退出当前数据集。** 早期 15 条抽查对它过于乐观；扩大人工审核后，85 条已完成样本中只有 22 条 target 严格通过、25 条 pair 严格通过，失败具有普遍性，不值得再依赖后续 specialist 从中打捞少量样本。已有 5k pilot 结果只作为反例和筛选校准记录保留，不进入正式训练候选，也不继续送入 Stage 1.2/1.5/1.6。

当前分类可以保留，但建议改两个名称：

- `object on person` 改为 **accessory（穿戴配饰）**，包括眼镜、帽子、首饰、手表等；服装单列，不混入 accessory。
- `face` 改为 **appearance（人物外观）**，包括身份、脸、年龄、头发、发色和胡须；服装与配饰仍单列。

`change` 和 `replace` 不合并：change 表示同一实体仍在、只改变属性；replace 表示换成另一个身份、类别或实体。具体的模糊边界留到后续 filter 阶段处理。

## 2. 建议的任务构成

原子编辑主池共有 18 个任务格：

| 操作 | 作用对象 |
| --- | --- |
| add | person、object、clothes、accessory |
| remove | person、object、clothes、accessory |
| change | appearance、clothes、accessory、object、background |
| replace | appearance、clothes、accessory、object、background |

补充约定：

- 手持物与独立物体暂时都记为 `object`；手持物必须额外检查手部姿态和交互逻辑，但先不增加新标签。
- 只有操作直接作用于衣物、且人物主体保持时才记为 `clothes`；“add a woman wearing a shirt”仍是 `person`，不能因为描述里出现衣服就误分到 clothes。
- InsAVE 与 GOKU 的 add/remove 若**直接作用对象**是衣物、鞋袜或裤装，在第零级直接排除；只在人物描述中出现服装词不触发排除。ReCo 的 clothes 质量相对可用，继续保留并进入后续 specialist，而不是套用这一规则。
- `remove/clothes` 只保留 target 中存在合理内层衣物或人体表面的 pair；透明、黑块、模糊填充或没有明确补全依据的一律排除。由高质量 clothes add pair 反向构造 remove 时也必须重新检查这一条件。
- `background change` 是保留原场景结构、修改背景属性；`background replace` 是换成新场景。全局画风变化不算 background change。
- style transfer 和 camera 不进入当前数据集。字幕、动作修改、音频编辑和 creative/freeform 也不属于上述 16 格，首版一并排除。
- 排除 style/camera 符合当前研究目标，但这不保证微调后仍保留基模的 style/camera 能力；是否保持这些能力应在训练与评测阶段另行处理。

18 个任务格首先是数据目录，不是训练配额。先把选定数据分支中的样本全部归类和筛选，得到各任务真实可用的数量与质量；之后再根据这份统计单独决定训练总量和采样比例。第一阶段只处理 single-step；CoinVE 等 multi-step 数据在 single-step 能力稳定后再考虑。

## 3. Metadata 筛选结果

以下状态只针对具体分支，不对整个 dataset 下结论。本节决定先处理哪些分支，不规定它们最终进入训练的数量。

| 数据集与分支 | 当前决定 | 对应任务 | 依据 |
| --- | --- | --- | --- |
| InsAVE `add_and_remove` forward | **不使用** | remove/person | 早期 15 条抽查结论已被扩大审核推翻：85 条已完成样本中 target 仅 22 条、pair 仅 25 条严格通过，人物删除质量整体偏低。整个分支退出当前训练候选，不再尝试用 specialist 打捞。 |
| InsAVE `general_editing` 的纯视觉 Replace 子总体 | 不进入简单 baseline | replace/person、object（本轮实际覆盖） | 固定 `part_04` 的 1,295 条候选中随机 10 对：instruction 通过 0、失败 8、存疑 2；有 no-op、替错对象和 endpoint 方向错误。若使用必须另做强语义过滤。 |
| InsAVE `clone_id`、`clone_id_voice` | 不用于 replacement | speech/lip-sync | instruction 明确要求保持人物外观或 identity。 |
| InsAVE `clone_voice` | 暂不用于 single-step | replacement + speech | 会换人物/服装，但全部同时改变 spoken words，是复合音视频任务。 |
| GOKU `remove` | 优先处理 | remove/person、object、accessory | 随机 15 条中 9 条严格通过，弱于 InsAVE，但能补充更局部、更多样的 removal；直接作用于衣物、鞋袜或裤装的样本先按 instruction 排除。 |
| GOKU `add` | 进入第一级 | add/person、object、accessory | 已完成 34 条：target 34/34、source 22/34、pair 21/34 严格通过。该分支由 remove pair 反向构造，source 是编辑侧、target 是原始侧；只对 source 跑生成质量 specialist。直接作用于衣物、鞋袜或裤装的样本先排除。 |
| EgoEditData 训练 edit table | 优先进入第一级 | replace/object | 已完成 22 条：source 22/22、pair 15/22 严格通过；保留为 object replacement 主力候选。 |
| ReCo `replace` | 优先进入第一级 | replace/person、clothes、accessory、object | 当前完成 20 条：target 18/20 严格通过、pair 15/20 严格通过；object 子组 target 10/10、pair 8/10。需区分精细机器人与粗劣 3D/动画角色。 |
| ReCo `add` | 进入第一级 | add/person、clothes、accessory、object | 已完成 15/20 条，target 11/15、pair 9/15 严格通过；出现质量较好的 clothes add，也出现粗劣角色 add。剩余 5 条空白不阻塞第一轮模型校准。 |
| ReCo `remove` | 保守进入第一级 | remove/person、clothes、accessory、object | target 17/20 严格通过，但 pair 仅 10/20；要额外过滤多目标指令、目标方位错误、源视频缺陷和不合理 clothes remove。 |
| InsAVE `add_and_remove` reverse | 暂不进入 | add/person | 虽然是 single-step add，但现有严格通过仅 8/15，明显弱于 forward remove。 |
| GOKU `swap_alter` | 暂不使用 | change/replace | 自然随机质量较差。 |
| OpenVE 全部分支 | 不使用 | - | 当前项目已决定不把 OpenVE 加入筛选或训练队列。 |
| TripVVT dresses/upper/lower body | 不进入当前 single-step 文本池 | replace/clothes | 质量较好，但依赖 garment reference，与当前 Lance 纯文本条件不一致。 |
| CoinVE：operation 不含 Stylize | single-step 完成后再使用 | add/remove/replace 的 multi-step 组合 | 数据质量较好，但只有全部 2–5 步完成后的最终 target，不能拆成 single-step。 |
| style、camera、movement、subtitles、creative | 不使用 | 当前范围外 | 不属于当前任务构成。 |

### 3.1 Source/target provenance 规则

provenance 不能只凭通用列名或文件名猜测，必须按固定的“dataset × 官方分支 × revision × 构造方向”推导：

| 数据分支 | source | target | 生成质量 specialist 检查侧 |
| --- | --- | --- | --- |
| InsAVE forward remove（仅历史 5k pilot，已停用） | original | edited | target |
| GOKU remove | original | edited | target |
| GOKU add（remove 反向） | edited | original | source |
| ReCo add/remove/replace，revision `63c6481a…` | generation_side_unresolved | generation_side_unresolved | both |
| EgoEdit original→edited | original | edited | target |
| EgoEdit edited→original | edited | original | source |
| EgoEdit edited→edited | edited | edited | both |

ReCo 官方 config 只有 `src_video`、`tar_video` 和 instruction，没有 `source_is_original` 或逐 pair 的生成 lineage。官方 README 给出的是训练 source/target 角色；[ReCo 论文补充材料](https://arxiv.org/pdf/2512.17650)说明数据还通过 removal 反转、replacement 双向交换和跨任务生成端点复用扩增，所以不能把所有 pair 推导成 `original→edited`。固定 revision 上将两侧记为 `generation_side_unresolved` 并检查两侧；当前本地 `source_is_original=1` 是旧脚本兼容值，不再作为证据。只有连构造规则也无法确认时才使用 `unknown` 并停止自动路由。

所有 pair 的 source 和 target 都执行时长、文件存在性等低成本技术检查；MUSIQ、DINOv2、frame diff 及后续单视频质量 specialist 只检查表中的生成/编辑侧。`original` 仅表示编辑 pair 中的原始侧，不等于该视频在整个生产历史中从未处理，也不等于画质必然完美。

## 4. 后续数据集建设顺序

加入 ReCo 三个官方原子分支、并扣除已停用的 InsAVE forward remove 后，本级 metadata 候选约 89 万条；这只是过滤前上限，不是最终训练量。摘要维护在 [`1_specialist_filtering/README.md`](./1_specialist_filtering/README.md)。后续每一级筛选单独建立带序号的目录，不混写在同一份 README 中。

1. 选择要处理的官方数据分支，并固定 repo revision。
2. 遍历该分支的全部 metadata，将每条 pair 归到 `operation × target_scope`；无法可靠归类的样本单独标记，不强行分栏。
3. 在分栏结果上执行该分支需要的 filter，并把通过、拒绝和存疑结果写回目录。
4. 汇总每个“数据集 × 分支 × 任务格”的原始数量、筛选通过率和最终可用数量。
5. 数据目录冻结后，才另写训练 sampling plan；训练样本总量和任务比例由实际覆盖、质量与模型目标共同决定。

这里的“遍历”是先完整处理 metadata；视频只对选定分支和候选样本按需获取。当前没有必要把所有候选数据集的全部视频先下载一遍。

## 5. 存储方案

总体采用 **云端归档 + SQLite 目录 + 训练节点临时数据**：

```text
云端：官方原始 archive + 筛选后的 release shard
                         ↑
/scratch：catalog.sqlite + metadata + 冻结 manifest
                         ↓
训练任务：按 manifest 下载到 $TMPDIR → 生成 Lance Parquet → 训练
```

### 5.1 云端保存什么

- 不无差别备份所有候选数据集；先保存完整 metadata，再归档已决定处理的官方任务分支。
- 原始 tar/zip 保持官方结构、revision 和校验和，作为可重建的数据源真值，不在云端展开成数百万个小 MP4。
- 筛选完成后，把 accepted pair 另外打成较小的 release shard。训练优先下载 release shard，避免为了少量样本反复拉取几十 GB 的官方 archive。
- 若可以选择，S3 类对象存储比普通同步云盘更适合大 shard 和按需下载；只有普通云盘时也沿用相同目录和 manifest 设计。

### 5.2 SQLite 记录什么

继续使用简单 SQLite，不把视频存成 BLOB。第一版只需要两张核心表：

- `datasets`：dataset 名称、repo、revision；
- `pairs`：dataset、官方分支、official ID、instruction、source/target URI、operation、target scope、filter status 和 filter version。

URI 指向云端 archive/member 或筛选后的 release 位置，不绑定某台机器的绝对路径。当前仍不记录 split、FPS、宽高和训练采样权重；这些不是数据目录的核心属性。

SQLite 工作副本放在 `/scratch`，定期把只读快照和 manifest 备份到云端。不要直接在同步云盘挂载目录上并发写 SQLite；训练时也只读取数据库导出的冻结 manifest。

### 5.3 训练时如何整合

1. 根据之后确定的 sampling plan 查询 SQLite，导出包含精确 `pair_id` 的冻结 manifest。
2. 从云端把 manifest 需要的 release shard 拉到作业节点 `$TMPDIR`。
3. 在 `$TMPDIR` 解包或生成 Lance 所需 Parquet；Parquet 是可重建的训练输入，不是数据真值。
4. 保存本次 manifest、数据 revision 和训练配置；训练结束后可删除节点本地数据缓存。

这样数据集建设保留全部合格样本，而每次训练只物化当次真正使用的子集。数据目录与训练分布彼此独立，后续改变采样比例不需要重新整理或重新上传原始数据。

## 6. 后续阶段

当前先完成 single-step 数据的分栏和筛选，再根据实际统计构造 baseline。之后补充 single-step change；single-step 稳定后，才加入 CoinVE multi-step。具体 filter 和训练比例分别形成后续文档，不在本计划中预设。
