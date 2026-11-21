# 第一级筛选：审核结论与第一轮自动过滤计划

2026-09-06 已完成 [20K Stage 1 候补扩筛](artifacts/stage1_reserve_20k_20260906/REPORT.md)：沿用最新生产门槛，保留 **18,662 对**（16,916 accepted + 1,746 uncertain），淘汰 1,338 对，异常隔离为 0。完整逐条核验已通过，CoinVE 未加入，未运行付费 Stage 2。首批 4K 的正式门槛与结果见 [STAGE1_PRODUCTION_REPORT.md](STAGE1_PRODUCTION_REPORT.md)；以下 pilot 记录保留原版本语境。

> 更新日期：2026-08-31  
> 当前状态：正式 5k `general_edit_stage1_pilot_v1` 已完成 Stage 1.0/1.1 并保持冻结。从 4,563 个 survivor 中排除 485 个 InsAVE、60 个手持物体 remove，以及修复 metadata 目标解析后发现的 15 个 GOKU 衣物 add/remove，正式继续入口为 4,003 对（3,737 accepted / 266 uncertain）。Stage 1.2 处理其中 683 条 ReCo robot/CG；Stage 1.5 仍取消；旧 Stage 1.6 不执行，pair/instruction 检查移至 `2_vlm_filtering`。OpenVE 不进入当前队列。

## 1. 为什么先普查

第零级最初进入本级的 metadata 候选约 91 万条；停用 InsAVE person remove 后，当前正式候选约 89 万条：

| 分支 | 任务 | 约数 |
| --- | --- | ---: |
| InsAVE `add_and_remove` forward | remove/person | 21,374（已整支停用，仅保留历史 pilot） |
| GOKU `remove` | remove/person、object、accessory | 174,000 |
| GOKU `add` | add/person、object、accessory | 230,000 |
| EgoEdit edit table | replace/object | 93,644 |
| ReCo `add` | add/person、clothes、accessory、object | 115,652 |
| ReCo `remove` | remove/person、clothes、accessory、object | 121,672 |
| ReCo `replace` | replace/person、clothes、accessory、object | 156,678 |

已有审核没有显示普遍的人脸身份跳跃、关节错位、头发闪变或服装区域系统性异常；产生主要担忧的 InsAVE person remove 和低质衣物 add/remove 已在上游整支或按任务排除。因此当前流程取消 Stage 1.5，不运行 InsightFace、DWPose 或 human parsing，也不为人脸单设检查。ReCo 仍有一个清晰的例外：机器人、动画或 3D 角色中既有精细且与实拍场景融合的可用样本，也有低模、游戏感或粗劣贴入的失败样本，不能只按 `robot` 关键词整体保留或整体删除。

目前真正观察到的问题是局部涂抹/失焦、source 编辑残留、source-target 错配、轨迹偏离、no-op、错误对象和不合理交互。尤其 EgoEdit 是编辑图：完整 93,644 条中包含 29,241 条 original→edited、16,113 条 edited→original、48,290 条 edited→edited，不能只检查 target。

当前审核已经足够决定第一轮 specialist 的路由和 pilot 构成；尚未审核的记录只影响通过率精确值，不阻塞自动过滤开发。

## 2. 固定抽样

[`survey_config.json`](./survey_config.json) 冻结 version、revision、seed 和配额：

- InsAVE remove：`part_00`、`part_01` 各 50，排除旧 15 条调研样本。
- GOKU remove：固定随机选中的 part22、part24 各 50，排除旧调研样本。
- GOKU add：固定随机选中的 part22、part27 各 50。
- EgoEdit：100 个不同 scene，按 30/20/50 覆盖三种方向，并排除旧调研样本。

最终 manifest 必须恰好 400 条、每分支 100 条。缺少任一 GOKU population 时，构建脚本会失败，不会用单一现有 shard 冒充全分支抽样。

### 2.1 ReCo 定向补充

原 400 条的 `sample_id`、顺序和媒体 ID 全部保持不变，ReCo 只追加在第 401–460 条。样本来自官方完整 metadata 和按 seed `20260829` 固定的 add 38、remove 26、replace 43 分片；每个 source/target tar 都只把前 768 MiB 中完整落入前缀的 pair 作为抽样总体。它们没有复用先前查看过的 204 条官方示例。

| 任务 | person | wearable/accessory | ordinary object | 合计 |
| --- | ---: | ---: | ---: | ---: |
| add | 5 | 5 | 10 | 20 |
| remove | 5 | 5 | 10 | 20 |
| replace | 5 | 5 | 10 | 20 |

remove 在抽样前排除衣物、鞋裤删除；当前固定前缀内因此剔除 44 条，另排除 4 条旧官方示例。最终 60 条的 source/target 都不少于 3 秒。该样本只判断 ReCo 三个分支是否值得进入一级筛选，不能用来估计完整 ReCo 的精确通过率。

## 3. 执行顺序

所有大 archive 放在作业节点临时盘；`artifacts/` 只保存抽样 population、被抽中的原视频、代理视频、审阅结果和 SQLite，不提交 Git。

### 3.1 准备 GOKU population

需要能访问 gated `bigfacing/GOKU-2M` 的 Hugging Face 凭据。每个分支脚本下载一次 metadata 和两个配置分片，并生成完整 case population：

```bash
python scripts/fetch_goku_survey_shards.py \
  --branch goku_remove \
  --work-dir /path/on/tmpdir/goku_remove

python scripts/fetch_goku_survey_shards.py \
  --branch goku_add \
  --work-dir /path/on/tmpdir/goku_add
```

若 archive 已由其他方式下载，可直接运行：

```bash
python scripts/prepare_goku_population.py \
  --task remove \
  --metadata-tar /path/remove.jsons.tar \
  --video-tar /path/remove.videos.part22.tar \
  --output artifacts/populations/goku_remove_part22.csv
```

### 3.2 生成冻结 manifest

InsAVE 与 EgoEdit 使用现有完整 metadata；GOKU 使用上一步 population：

```bash
python scripts/build_survey_manifest.py
```

输出为 `artifacts/survey_manifest.csv`。脚本验证 400 条配额、唯一 sample ID、旧调研排除和 EgoEdit scene 去重。

以下命令准备 ReCo 固定分片、提取 60 对入选原视频并生成追加清单；它不会改写原 400 条清单：

```bash
python scripts/extend_survey_with_reco.py --work-dir /tmp/reco_survey_63c6481
```

已有各分片不少于 768 MiB 的完整或可续传本地文件时，可加 `--use-local-prefixes`，避免等待未使用的后续字节。

### 3.3 获取被抽中的视频

InsAVE 可由脚本下载两个官方 archive，也可复用现有 part01：

```bash
python scripts/fetch_insave_survey_shards.py \
  --work-dir /path/on/tmpdir/insave
```

将本地 tar 映射到 manifest 中的官方名字，一次只提取命中的成员：

```bash
python scripts/extract_archive_pairs.py \
  --archive train_data/add_and_remove/part_00.tar=/path/part_00.tar \
  --archive train_data/add_and_remove/part_01.tar=/path/part_01.tar \
  --archive remove/remove.videos.part22.tar=/path/remove.videos.part22.tar \
  --archive remove/remove.videos.part24.tar=/path/remove.videos.part24.tar \
  --archive add/add.videos.part22.tar=/path/add.videos.part22.tar \
  --archive add/add.videos.part27.tar=/path/add.videos.part27.tar
```

EgoEdit 使用已建立的远程 ZIP central-directory 索引，只进行成员 Range 请求：

```bash
python scripts/download_egoedit_survey.py
```

### 3.4 构建本地 HTML 包

每个唯一 MP4 只转码一次，输出最长边不超过 720p、CRF 20、无音频的 H.264 proxy：

```bash
python scripts/build_review_bundle.py
```

将整个 `artifacts/review_bundle/` 下载到本地，双击 `index.html` 即可使用，不需要 Web 服务或 Python 环境。

当前已生成便于一次下载的归档：

```text
artifacts/specialist_error_survey_400_v1_review_bundle.tar
```

下载到本地后解包并打开 `review_bundle/index.html`。归档约 573 MB，SHA256 为 `86e14f1179f6cd7e0dcdc48feed5295eb2757141e14982cf6605d3834a9a33f4`。

追加 ReCo 的当前归档为：

```text
artifacts/specialist_error_survey_460_reco_v3_review_bundle.tar
```

新版包含 460 条 pair、920 个唯一代理视频，大小 `668016640` bytes，SHA256 为 `82245fff76d32d29350b6ea188c92b96963487cd0ac40ece2b92899a5d339450`。解包后打开 `review_bundle_reco_v3/index.html`，点击“导入”并选择包内 `previous_results/specialist_error_survey_400_v1_backup.json`；页面会按稳定 `sample_id` 合并旧结果。该包生成时 ReCo 60 条为空白，当前最新结果已经另存到 `artifacts/review_results/specialist_error_survey_460_reco_v3_review.csv`。JSON/CSV 的旧版本号不会造成拒绝。

百度网盘位置：`/apps/bypy/lance_review/specialist_error_survey_460_reco_v3_review_bundle.tar`。

### 3.5 Stage 1.1 存疑样本快速复核包

`general_edit_stage1_uncertain_review_15_v1` 从排除 InsAVE 后的 Stage 1.1 `uncertain` 中抽取 15 条：EgoEdit replace 10 条，GOKU add/remove 与 ReCo add/remove/replace 各 1 条。优先避开原 300 条审计清单；由于少数小分支没有足够的新候选，最终有 2 条复用旧清单中的未审核候选。冻结清单、抽样摘要和代理视频都包含在归档中。

```text
artifacts/stage1_uncertain_review_15_v1_bundle.tar
```

归档大小为 `35,983,360` bytes，SHA256 为 `b6072c2e38547ef9825726bc45ac9e4a3a296806ed2477d6d14c1654869a807e`。解包后打开 `stage1_uncertain_review_15_bundle/index.html`。已通过 9 个校验分片上传至百度网盘：`/apps/bypy/lance_review/stage1_uncertain_review_15_v1_bundle.tar`。

### 3.6 手持物 removal 快速复核包

`general_edit_stage1_handheld_remove_review_15_v1` 从排除 InsAVE 后的 Stage 1.1 survivor 中抽取 15 条新样本：GOKU remove 8 条、ReCo remove 7 条。只选择删除手中物体、且删除后人物与手部应保留的 pair；“删除拿着物体的人/机器人”不进入清单。样本覆盖手机、书、杯子、吉他、剑、礼盒、鞋、兔子等，并避开此前的 300 条审计和 15 条存疑复核样本。

```text
artifacts/stage1_handheld_remove_review_15_v1_bundle.tar
```

归档大小为 `14,807,040` bytes，SHA256 为 `874bb48a3a59bebcddd3aa787450e468a6558ced1af64d0ff9e662b1a5a4f267`。解包后打开 `stage1_handheld_remove_review_15_bundle/index.html`，页面同时显示 source、target、instruction 并只记录一次整体判断。已通过 4 个校验分片上传至百度网盘：`/apps/bypy/lance_review/stage1_handheld_remove_review_15_v1_bundle.tar`。

**审核结论：手持物 object remove 退出当前数据集。** 15 条定向抽样中大多数不可用；常见问题是物体消失后手指、手掌或握持姿态发生混乱、残缺、粘连和不合理补全。此类 target 会把错误手部形态作为训练监督，损害模型在其他编辑任务中的手部保持与生成能力。相比暂时缺少这一小类 removal，保留坏手监督的风险更高，因此当前不运行关节模型尝试打捞，而是在 instruction/任务解析后直接拒绝。

排除条件必须以 instruction 的**直接删除对象**为准：删除正在被手持、抓握、携带或挥动的物体时记为 `excluded_handheld_object_remove`；删除“拿着物体的人/机器人”不属于这一规则，因为删除目标是 person/character，手部会随主体一起消失；仅仅有手出现在附近、但物体放在桌面或背景中，也不能仅凭 `hand/holding` 关键词误删。当前首版不为该任务格保留最低配额，未来只有找到经过独立审核的高质量来源后才重新开放。

## 4. HTML 审阅口径

从 `general_edit_stage1_uncertain_review_15_v1` 开始，所有新人工审核包统一采用**一次一条 pair、一次整体判断**的紧凑页面：

- 页面始终同时显示 source、target 和 instruction，不再依次进入 Target、Source、Pair 三个步骤。
- 每条只输出一个整体结论：`可用 / 不可用 / 仍存疑`。`可用` 表示视频自身质量、source-target 对应关系和 instruction 完成度综合可接受；任一方面存在明确硬错误就判 `不可用`。
- 审核页不展示 MUSIQ、DINOv2 等自动边界原因，避免锚定人工判断；必要信息只写在可选备注中。
- `1/0/2` 对应可用/不可用/仍存疑，左右方向键切换样本，空格同步播放 source 与 target。
- 结果自动写入浏览器 local storage，并可导出 CSV 或 JSON；重新导入时按稳定 `sample_id` 合并已有进度。

[`compact_pair_reviewer_template.html`](./compact_pair_reviewer_template.html) 已成为 `build_review_bundle.py` 的默认模板。旧 [`reviewer_template.html`](./reviewer_template.html) 只用于复现已有三阶段审核包；如确需重建旧包，显式传入 `--template reviewer_template.html`，后续新审核不再使用旧界面。

## 5. 当前审核结论

下表只统计 source、target、pair 三项都已填写的记录；“严格通过”不包含 uncertain。

| 分支 | 完成 | Target 严格通过 | Source 严格通过 | Pair 严格通过 | 当前决定 |
| --- | ---: | ---: | ---: | ---: | --- |
| InsAVE remove | 85/100 | 22/85 | 79/85 | 25/85 | **整支停用**；不进入正式训练，也不再依赖后续 specialist 打捞 |
| GOKU add | 34/100 | 34/34 | 22/34 | 21/34 | 保留；重点过滤反向构造产生的坏 source |
| GOKU remove | 37/100 | 27/37 | 37/37 | 19/37 | 保留；加强局部模糊、遮罩边和 pair 过滤 |
| EgoEdit replace | 22/100 | 16/22 | 22/22 | 15/22 | replacement 主力候选 |
| ReCo add | 15/20 | 11/15 | 12/15 | 9/15 | 保留；机器人/角色走专门路由 |
| ReCo remove | 20/20 | 17/20 | 14/20 | 10/20 | 画面质量较好，但必须做 pair/instruction 过滤 |
| ReCo replace | 20/20 | 18/20 | 18/20 | 15/20 | 当前最优先的 ReCo 分支 |

ReCo 中 11 条机器人、动画或 CG 相关记录按 pair 口径为 3 条通过、4 条明显失败、4 条存疑。失败集中在低模/游戏感、粗劣角色贴入和删除后悬浮残影；通过样本则包含细节完整、光照与场景较融合的机器人。因此 `robot/cartoon/animated/3D` 关键词只用于路由，不能直接作为删除条件。

ReCo 的衣物编辑出现了质量较好的样本，正式筛选时单列 `clothes`。但 metadata 分类必须识别真正的编辑目标：“add a suit to the man”是 clothes，“add a man wearing a suit”是 person。`remove/clothes` 只有在 target 合理显示内层衣物或人体表面时才允许进入。InsAVE 与 GOKU 不沿用 ReCo 的策略：其 add/remove 中直接作用于衣物、鞋袜或裤装的样本在 metadata 阶段排除；人物描述里附带服装词不误删。

审核备注中的“方向反转”统一解释为 **source 是编辑/生成视频，target 是原始或更自然的视频**，不表示 instruction 写反。它与真正的 `instruction_direction_wrong` 必须分开。原始 review CSV 不改写，但后续派生统计不再把这些 `direction_wrong` 标记直接计为指令方向错误。

ReCo 是另一种情况：官方 config 只有训练角色 `src_video`、`tar_video` 和 instruction，没有逐 pair 的原始/生成 lineage。README 中的 source/target 表述只能证明训练方向；[ReCo 论文补充材料](https://arxiv.org/pdf/2512.17650)还明确包含 removal 反转构造 add、replacement 双向交换及跨任务复用生成端点，因此不能据此把所有 ReCo 写成 `original→edited`。固定 revision `63c6481a4aedae39ec602ce8e17a836cc07b4ca9` 的 ReCo 两侧统一记为 `generation_side_unresolved`，Stage 1 单视频质量检查两侧；`source_is_original=1` 只是旧构建脚本产生的兼容值，不再作为 provenance 证据。

## 6. 第一轮自动过滤计划

过滤严格按顺序短路：前一步明显失败，后续模型不再运行。阈值先在 pilot 上校准，不在文档中凭经验写死。

### 6.1 Stage 1.0：Metadata 与硬条件

1. 只处理第零级明确入选的官方分支，固定 repo revision、pair ID 和 instruction 方向。
2. source 与 target 都必须不少于 3 秒；暂不增加一次全量解码初筛。
3. 重新解析 instruction 的直接作用对象，分开 person、object、clothes、accessory；仅仅在人物描述里出现衣服不算 clothes。InsAVE/GOKU 的 add/remove 若直接宾语是衣物、鞋袜或裤装则立即拒绝，ReCo clothes 不套用该规则。
4. 对所有数据分支，若 remove 的直接对象是正在被手持、抓握、携带或挥动的物体，按 `excluded_handheld_object_remove` 直接拒绝；删除持物人物不触发该规则。
5. 排除 style、camera、multi-step，以及明确包含两个独立编辑目标但只完成其一的指令。
6. 对每条 pair 分开记录 `instruction_direction`、`source_provenance`、`target_provenance`、`provenance_basis` 和 `quality_checked_roles`。现有 `source_is_original` 只作兼容字段，不作为独立证据。

InsAVE `add_and_remove` forward 在历史 5k pilot 中仍按原规则保留，以保证实验可复现；正式扩量和训练 manifest 在本阶段首先按 `branch=insave_remove` 整支拒绝，不再下载媒体或运行后续 specialist。

生成质量检查侧按构造规则确定，不对 source/target 使用相同门槛：

| 数据分支 | source provenance | target provenance | `quality_checked_roles` |
| --- | --- | --- | --- |
| InsAVE forward remove（仅历史 5k pilot，已停用） | original | edited | `target` |
| GOKU remove | original | edited | `target` |
| GOKU add（remove 反向） | edited | original | `source` |
| ReCo add/remove/replace，固定 revision `63c6481a…` | generation_side_unresolved | generation_side_unresolved | `both` |
| EgoEdit original→edited | original | edited | `target` |
| EgoEdit editenal | edited | original | `source` |
| EgoEdit edited→edited | edited | edited | `both` |

所有 source/target 都做文件存在性、时长不少于 3 秒等低成本技术检查。只有表中生成/编辑侧运行 Stage 1.1 及之后的单视频 specialist；若构造规则明确存在生成端点交换、但发布 metadata 不提供逐 pair lineage，则使用 `generation_side_unresolved` 并检查两侧。真正缺少数据集构造证据的情况才记为 `unknown` 并停止自动路由，不能把它与已知“生成侧不可解析”混为一谈。

### 6.2 Stage 1.1：通用单视频质量

只检查 Stage 1.0 路由出的生成/编辑侧；同一 pair 若两侧都是编辑产物或 provenance 不确定，则检查两侧。每个被检查的视频均匀抽帧，使用：

- **MUSIQ**：看低分位帧而不是只看平均值，筛掉全局模糊、粗糙画质和局部严重失真。
- **DINOv2 时序特征**：检测相邻抽帧的异常特征跳变，覆盖人物、人脸、头发或物体突然变形。
- **frame diff**：检测冻结/重复、黑白帧、突发闪变和几乎静止的错误片段。

被路由为 edited 的一侧无论是 source 还是 target，都使用同一套生成质量门槛：透明/黑块、模糊填充、遮罩残影、局部冻结、人体异常或明显编辑痕迹均直接淘汰。未运行 specialist 的 original 侧仍在 pair 检查中识别明显错片和严重异常；`original` 只是 pair 角色，不是质量通过标签。这里不运行 RAFT、CoTracker、AMT、VEFX 或通用 VLM judge。

通过质量门槛后仍按 provenance 分池：可确认 `original + pass` 的训练 source 进入当前 single-step 主池；可确认 `edited + pass` 的训练 source 进入独立辅助池，首版 baseline 不让它占主导；`edited + uncertain/fail` 不进入训练。`generation_side_unresolved` 不强塞进任一 provenance 池，保留训练 source/target 方向并要求两侧质量和 Stage 2 pair 判断均通过，再作为单独的 ReCo 候选池统计。

#### 下一次大规模筛选的阈值调整

`general_edit_stage1_uncertain_review_15_v1` 的人工复核结果为 14/15 可用；唯一不可用的是第 14 条 ReCo remove，它的 target 几乎冻结。该样本现有 12 帧抽样指标为 `frame_diff_mean=0.00617108`、`frame_diff_p95=0.00955988`、`freeze_ratio=0.09090909`，说明当前主要依赖极低 `frame_diff_p05` 或高 `freeze_ratio` 的规则会漏掉“所有帧都有微小变化、但整体几乎不动”的视频。

本次已经完成的 5k `general_edit_stage1_pilot_v1` 保持冻结，不追溯重算。下一次大规模筛选时按以下口径更新：

- 收紧近冻结检测：除现有 freeze ratio 外，增加 `frame_diff_mean + frame_diff_p95` 的持续低运动联合条件，并把上述第 14 条作为必须拒绝的负例；阈值同时用其余 14 条人工通过样本约束，避免误杀正常慢运动和固定机位视频。
- 对 removal 的编辑侧增加严格的整体清晰度硬门槛：`MUSIQ p10 < 34.274181368` 或 `sharpness p10 < 106.029077758` 任一成立就直接 `rejected`，不再进入 `uncertain`，也不能被后续存疑回收流程捞回。阈值取 133 条人工通过编辑侧视频的第 10 百分位，两个指标按 OR 执行。
- 该 removal 硬门槛在 InsAVE 人工标签上筛中 19/19 条 `global_blur` target；代价是同时触发 13/22 条 InsAVE 人工 pass。InsAVE 已整支停用，因此这个代价不影响训练池。迁移到较高质量分支时，人工 pass 触发量为 GOKU remove 2/27、ReCo remove 2/17；在冻结 5k 分数上的预计触发量分别为 13/440 和 92/873，符合优先质量的目标。
- 上述严格门槛只用于 **removal 生成侧**，不能按训练 instruction 的字面值全局套到 add/replace：常规 remove 检查 edited target；GOKU add 是 removal pair 的反向训练组织，因此同样严格检查 edited source。此次人工确认可用的 EgoEdit 边界样本说明 MUSIQ/清晰度在不同任务与内容域上存在偏差；其他非-removal 生成任务的单项边界继续保守路由，DINOv2 单项异常也不能独立硬拒绝。
- 这项调整只作用于未来新运行及大规模筛选版本；现有 CSV、报告、候选池与计数继续作为原版本实验记录保存。

未来规则的机器可读草案保存在 [`stage1_1_future_policy.json`](./stage1_1_future_policy.json)。正式扩量前由筛选脚本显式加载该版本，不能修改或覆盖当前冻结的 `stage1_1_thresholds.json`。

### 6.3 Stage 1.2：ReCo 机器人与角色路由

1. 用 instruction 词表找出 robot、humanoid、cartoon、animated、puppet、3D/game character 等候选，但不据此拒绝。
2. 只对候选运行 YOLO-World 的 robot/person/character 定位；检测不到时保留全帧与显著区域两套 crop。
3. 对 crop 运行 MUSIQ，并用 CLIP/SigLIP 的成对提示分数比较“细节完整、真实感、与场景融合”与“低模、粗劣 3D、动画/游戏、贴纸感”。同时用 DINOv2 检查角色区域外的场景是否稳定。
4. 高质量机器人正常进入对应 add/remove/replace；低模、粗劣动画主体、明显贴入或删除后残影直接拒绝；边界样本进入 uncertain，不让一个零样本提示分数直接作最终裁决。

这一分支在 5k pilot 中定向抽 500 条机器人/CG 候选；人工复核只覆盖高分、低分和决策边界，不逐条观看全部候选。

### 6.4 Stage 1.5：已取消的人物与服装专项

当前不运行 InsightFace、DWPose、human parsing、RAFT 或 CoTracker，也不设置独立人脸一致性门槛。原因是保留分支的审核中没有观察到稳定、可规模化复现的人脸身份、人体拓扑或服装区域问题，而原先引发担忧的分支已经在上游淘汰。手持物 remove 的坏手问题也已经按任务格直接排除，不再用 DWPose 打捞。继续运行这些模型缺少收益依据，还会增加算力与误杀。

现有 `stage1_5_human_queue.csv` 是冻结 5k 在旧计划下产生的历史路由，不执行、不进入训练决策。`build_specialist_queues.py` 默认不再生成 Stage 1.5 队列；只有显式传入 `--legacy-human-route` 才能为历史复现重建。未来若加入新的数据分支并在抽查中重新发现系统性人脸、人体或服装异常，再按该分支单独恢复，而不是全局开启。

### 6.5 Stage 1.6：已取消

原计划用 instruction parser、YOLO-World 对象存在性和 DINOv2 区域差异间接判断 pair 是否编辑成功。这与第二级综合 judge 职责重叠，且对开放词汇、replacement 和细粒度对象容易形成脆弱规则，因此不再执行。现有 `stage1_6_pair_candidate_queue.csv` 只作历史产物保留，不进入最终决策。

pair 的 no-op、方向反转、错对象、未完成、非目标内容破坏和时序异常交给第二级综合 judge。当前 Qwen3-VL-32B 版本只保留为校准实验，未达到硬判定门槛，不能直接裁决训练数据。

### 6.6 5k pilot

这次历史 pilot 没有直接跑当时的 91 万条候选。phase A 先构造 4.5k：七个分支各随机 500 条，共 3,500；再补 500 条 ReCo 机器人/CG 和 500 条真正的 clothes 编辑。Phase A 完成后，phase B 从同一批已固定 archive 中再抽 500 条**新样本**：每分支至少 50 条，剩余额度按 Phase A 的 uncertain 率分配。这样总计为 5k 且严格去重；真正最靠近模型阈值的已评分样本另进入 300 条人工审计，不把重复审计样本伪装成新增数据。该 pilot 完成后才根据扩大审核停用 InsAVE，因此其 InsAVE 样本只保留作历史证据，不沿用到正式扩量。

ReCo 采用 archive-aware 抽样，避免全量随机把少量样本散到全部 tar。固定的 15 个 shard 为：add `00/37/38/42/56`、remove `00/20/39/44/50`、replace `11/34/37/58/72`；基础随机分别从 add 38、remove 50、replace 58 抽取，定向样本在同组 shard 内抽取。这样仍保留 metadata 内随机性，同时把 ReCo 下载范围限制为 15 对 source/target tar。

人工只复核 300 条：自动接受、自动拒绝、边界样本各 100 条，并在三组内按 branch、pilot group、target scope 轮转覆盖，确保包含 ReCo robot/clothes。后续新建或重建审核包时统一同时展示 source、target、instruction，并只记录一次整体判断。根据误杀和漏放调整阈值后，再决定是否扩到全量。

## 7. 结果持久化与当前边界

当前审核真值保存在 `artifacts/review_results/specialist_error_survey_460_reco_v3_review.csv` 和对应 JSON。CSV 可继续导入 SQLite；自动过滤结果写入新的 filter run 表或 manifest，不覆盖人工审核原件。

- source 和 target 都必须不少于 3 秒并通过技术检查；正式生成质量门槛作用于构造规则能确定的 edited 侧。`generation_side_unresolved` 或 edited→edited 检查两侧；普通 `unknown` 不自动猜测，先停止路由并查明 metadata。
- OpenVE 不进入队列；CoinVE 等 multi-step 留到 single-step 稳定以后。
- 当前只保留有审核依据的 robot/CG specialist；人物、服装和人脸专项不运行。
- 全量吞吐和算力只按 5k pilot 的实测结果外推。

## 8. 正式运行状态（`general_edit_stage1_pilot_v1`）

### 8.1 已完成

- [`filter_config.json`](./filter_config.json) 固定 revision、seed、分支配额、EgoEdit 方向配额和 ReCo shard。
- phase-A manifest 共 4,500 条：`branch_random=3500`、`reco_robot_cg=500`、`reco_clothes=500`；排除了 460 条既有审核 pair。
- Phase A Stage 1.0：4,248 accepted、252 rejected；Phase B 的 500 条新增样本全部通过 metadata，因此最终 5k 为 4,748 accepted、252 rejected。拒绝原因计数为 `excluded_task_scope=247`、`excluded_garment_add_remove=164`、`excluded_style=4`、`excluded_camera=1`；同一条可有多个原因。
- 阈值校准：旧审核集中 510/510 个需检查视频完成技术、frame diff、MUSIQ-Koniq 和 DINOv2-small；完整模型评分 CPU 耗时 716 秒。233 个有生成侧人工标签的视频中，MUSIQ p10、sharpness p10、DINO 最大特征跳变的 hard-fail badness AUC 分别约为 `0.707/0.725/0.621`；frame diff 与 freeze ratio 区分度不足，只保留极端硬错误，不用于普通边界判定。
- 自动拒绝规则保持保守：MUSIQ p10 与 sharpness p10 同时落入人工 pass 的 2% 极端尾部才按低画质拒绝；DINOv2 或单个质量分只进入 uncertain。黑/白帧和极端重复超出人工 pass 已见范围时可硬拒绝。
- InsAVE 500 对已全部获取并完成双侧技术检查：990/1,000 视频通过，10 个短视频对应 5 个 pair。生成侧模型筛选结果为 197 accepted、192 uncertain、111 rejected。
- GOKU remove 387 对全部完成：380 accepted、7 uncertain、0 rejected；7 个暂存均为 DINOv2/清晰度保守边界。
- GOKU add 377 对全部完成：369 accepted、8 uncertain、0 rejected；按数据构造规则只把 edited source 用于生成质量裁决，未把 original target 混入门槛。
- EgoEdit 的 975 个唯一媒体和 ReCo 固定 30 个 source/target tar 已完成定向提取。EgoEdit 复用视频会复制到所有 pair 的稳定本地路径；ReCo remove 有 2 对 config 条目在两个官方 tar 中都缺失，已在 acquisition state 中留痕，最终按 `media_missing` 拒绝。
- Phase B 按 Phase A uncertain 率自适应抽取 500 条严格去重新样本，分支配额为 InsAVE 123、GOKU remove/add 53/54、EgoEdit 99、ReCo add/remove/replace 54/62/55；结果为 382 accepted、88 uncertain、30 rejected。
- 完整 5k：metadata 后 4,748 对；Stage 1.1 为 4,054 accepted、509 uncertain、185 rejected。分支结果见 `artifacts/filter_run_v1/stage1_5k_report.md`。
- 后续候选：3,323 条 original-source 主池、731 条 edited-source 辅助池；Stage 1.2 robot/CG 队列 683 条，历史 Stage 1.5 队列 1,695 条（现已停用），Stage 1.6 pair 候选 4,563 条。
- 2026-08-30 扩大审核后的分支级决定：InsAVE person remove 整支退出。冻结 CSV 不改写；正式继续时排除 623 条 InsAVE，其中 Stage 1.1 accepted 245、uncertain 240。相应地，主池从 3,323 降为 3,078，辅助池仍为 731；Stage 1.6 候选从 4,563 降为 4,078。排除 InsAVE 后历史 Stage 1.5 队列虽剩 1,210 条，但该阶段已经取消，不再处理。此前生成的 CSV 必须先按 `branch != insave_remove` 过滤，不能直接当作训练 manifest。

### 8.2 当前产物与继续命令

运行产物位于 `artifacts/filter_run_v1/`：

- `pilot_manifest_phase_a.csv`：冻结 4.5k 清单；
- `pilot_manifest_phase_b.csv` / `pilot_manifest_5k.csv`：500 条自适应新增样本与最终 5k 清单；
- `stage1_0_results.csv` / `stage1_0_survivors.csv`：metadata 决策与 4,748 条 survivor；
- `quality_calibration_scores.csv` / `stage1_1_thresholds.json`：原始校准分数与冻结阈值；
- `stage1_technical_available.csv` / `stage1_quality_available.csv`：已获取媒体的双侧技术分与生成侧模型分；
- `stage1_1_pair_decisions.csv`：4,748 条 metadata survivor 的最终 Stage 1.1 决策，无 pending；
- `training_main_candidates.csv` / `training_auxiliary_candidates.csv`：按 source provenance 分开的训练候选；
- `stage1_2_robot_queue.csv`：仍有效的 robot/CG 路由；`stage1_5_human_queue.csv` 和 `stage1_6_pair_candidate_queue.csv` 都只是旧计划的历史产物，当前不执行；
- `stage1_audit_300.csv`：accepted/uncertain/rejected 各 100 条的分层人工审计清单；
- `stage1_5k_report.md`：自动生成的紧凑结果报告。

关键复现命令：

```bash
python scripts/build_filter_pilot.py
python scripts/run_stage1_metadata.py

PYTHON_BIN=/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python
$PYTHON_BIN scripts/run_stage1_quality.py \
  --input artifacts/filter_run_v1/stage1_0_survivors.csv \
  --metric-set technical --role-mode both --skip-missing --resume \
  --output artifacts/filter_run_v1/stage1_technical_available.csv

HF_HOME=artifacts/model_cache/huggingface \
TORCH_HOME=artifacts/model_cache/torch \
TRANSFORMERS_OFFLINE=1 \
$PYTHON_BIN scripts/run_stage1_quality.py \
  --input artifacts/filter_run_v1/stage1_0_survivors.csv \
  --metric-set full --role-mode quality --skip-missing --resume --device cpu \
  --output artifacts/filter_run_v1/stage1_quality_available.csv

$PYTHON_BIN scripts/apply_stage1_thresholds.py \
  --manifest artifacts/filter_run_v1/stage1_0_survivors.csv \
  --scores artifacts/filter_run_v1/stage1_quality_available.csv \
  --technical-scores artifacts/filter_run_v1/stage1_technical_available.csv \
  --video-output artifacts/filter_run_v1/stage1_1_video_decisions.csv \
  --pair-output artifacts/filter_run_v1/stage1_1_pair_decisions.csv \
  --summary artifacts/filter_run_v1/stage1_1_decision_summary.json \
  --finalize-missing
```

媒体与 Stage 1.1 已完成。大 tar 下载到 `$TMPDIR`，只把命中视频写回 `artifacts/filter_run_v1/media/`，提取后删除；评分使用 4 个独立 CPU 分片，再由 `merge_stage1_scores.py` 按稳定 `video_id` 去重合并。300 条人工审计包位于 `artifacts/stage1_audit_review_bundle/`；它用于校验误杀/漏放，不自动改变当前冻结阈值。
