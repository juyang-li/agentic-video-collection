# 开源视频编辑数据集推荐顺序

本排序只针对当前项目。现在把**实际视频抽查质量放在第一位**，再考虑是否符合 `source video + 单条 add/remove/replace 指令 + target video`、人物属性相关性、数据许可和获取成本。未完成可靠视频审核的数据集不再因规模大或标签形式漂亮而排在前面。

这里仍保留两个硬边界：参考图条件的服装数据即使画质很好，也不能直接替代纯文本编辑数据；由 target 自身遮罩得到输入的重建数据，也不能当作真实 source-target 编辑 pair。

## 1. InsAVE-80K

- **数据形式**：`源音视频 + 单条正向/反向指令 + 目标音视频`，视觉和音轨可能同时改变。
- **主要任务**：人物/物体 add-remove、identity/voice cloning、speech 和通用音视频编辑。
- **任务组织结构**：训练数据按任务目录独立发布；每类 CSV 都用 `original_video、target_video、instruction、instruction_reverse` 连接对应 tar 中的视频，因此同一 pair 可提供正反两个方向的监督。

  ```text
  InsAVE-80K/
  ├── train_data/
  │   ├── add_and_remove/ + add_and_remove.csv
  │   ├── clone_id/ + clone_id.csv
  │   ├── clone_id_voice/ + clone_id_voice.csv
  │   ├── clone_voice/ + clone_voice.csv
  │   └── general_editing/ + general_editing.csv
  └── eval_data/
      └── eval/ + eval.csv
  ```

  `add_and_remove` 是当前 atomic 主线；其余分支涉及身份、音色、语音或宽口径复合编辑，不能直接混入纯视觉 add/remove。
- **数据集大小**：88,074 pair、约 138.56 GB；其中 `add_and_remove` 为 21,377 pair、约 32.25 GB。
- **抽查与推荐**：随机样本全部涉及整个人物；forward remove 严格通过 13/15，reverse add 为 8/15，是当前已抽查的同方向数据中最稳的一组。优先保留 remove，add 需过滤删除残影；必须标注为联合音视频数据。[官方数据](https://huggingface.co/datasets/suimu/InsAVE-80K)｜[本地调研](insave_research/insave_report.md)

## 2. CoinVE-200K

- **数据形式**：`源视频 + 2–5 条有序原子指令 + 目标视频`，另有 operation/object 和 mask。
- **主要任务**：add、remove、replace、style 等操作的组合编辑。
- **任务组织结构**：它不是按 add/remove/replace 分目录的 single-task 数据，而是一个统一的 compositional 样本池。每行 metadata 同时保存有序指令数组、对应 operation/object 数组、逐指令 mask 和合并 mask。

  ```text
  CoinVE-200K/
  ├── metadata_coinve200k.jsonl
  ├── src_videos/          # source MP4 分片
  ├── tgt_videos/          # 完成全部步骤后的 target MP4 分片
  ├── instruction_masks/   # 每条原子指令一个 mask video
  └── combined_masks/      # 多步编辑区域的合并 mask

  单条 metadata：
  ├── instruction[]           # 2–5 条有序指令
  ├── instruction_operation[] # Add / Remove / Replace / Stylize
  ├── instruction_object[]    # subject / object / background
  └── instruction_mask_video_paths[]
  ```

  训练时应保留整条为 `multi_edit_pair`；不能因为内部有原子标签，就把最终 target 拆成缺少中间目标的多个 single-task pair。
- **数据集大小**：200,916 个 1080p 样本、约 1.18M 个视频文件，完整发布约 2.8 TB。
- **抽查与推荐**：15 条中 instruction 通过 13 条、visual 通过 14 条，9 条涉及人物属性，整体观感较稳；主要扣分是 5 对 target 截短及少量保持、时序问题。质量较好，但它是 multi-task，只适合作为组合编辑池，不能混入 single-task 主池。[官方数据](https://huggingface.co/datasets/FireCRT/CoinVE-200K)｜[本地调研](coinve_research/coinve_report.md)

## 3. GOKU-2M

- **数据形式**：`源视频 + 单条指令 + 目标视频`；不同任务分别发布 metadata tar 和视频 tar。
- **主要任务**：add、remove、swap/alter 等独立任务分支。
- **任务组织结构**：逻辑上是一个数据集，因总量大拆到两个 gated 仓库，并非按 remove / 非-remove 划分。第一个仓库包含当前需要的基础外观编辑，第二个仓库只放动作和组合编辑。

  ```text
  bigfacing/GOKU-2M/
  ├── add/
  ├── remove/
  ├── swap_alter/
  ├── reference_add/
  ├── reference_swap/
  ├── style_transfer/
  └── camera/

  Goku-2M/GOKU-2M/
  ├── subject_movement/
  └── multi_task/

  每个普通任务：
  ├── <task>.jsons.tar
  └── <task>.videos.partNN.tar
      └── <case_id>/{source.mp4, edited.mp4}
  ```

  `add/remove` 可直接作为 atomic 候选；`swap_alter` 仍需拆成 object replace、attribute change 和区域替换。reference 任务额外需要参考图，`multi_task` 则包含有顺序的中间步骤。
- **数据集大小**：约 2M 编辑 pair；两个 gated 仓库合计约 9.65 TB。
- **抽查与推荐**：`remove` 随机 15 条中 instruction 通过 13 条、六项严格通过 9 条，明显优于 `swap_alter` 的 4/15 严格通过。保留 single-task `remove`，下一步应按同样方法抽 `add`；不能用较差的 swap 分支代表整个数据集。[官方数据](https://huggingface.co/datasets/bigfacing/GOKU-2M)｜[remove 调研](goku_research/remove/goku_remove_report.md)

## 4. EgoEditData

- **数据形式**：`第一视角源视频 + 单条替换指令 + 目标视频`，不需要参考图或音频条件。
- **主要任务**：手持或操作中物体的 atomic replacement；当前发布标注没有独立纯 remove 子总体。
- **任务组织结构**：EgoEditData 没有按 add/remove/replace 建任务目录，也没有在训练 edit table 中提供 task 标签；它使用一个视频表和一个有向编辑边表，通过 ID 连接 source、target 和 prompt。另行发布的 EgoEditBench 只有 source 和 instruction，不提供训练 target。

  ```text
  annotation/
  ├── egoeditdata_data_table.csv
  │   └── data_id -> data_url
  ├── egoeditdata_edits_table.csv
  │   └── source_data_id + target_data_id + source_to_target_edit_prompt
  ├── egoeditbench_data_table.csv
  └── egoeditbench_edits_table.csv
      └── source_data_id + prompt + edit_task + sub_edit_task

  videos_001.zip ... videos_008.zip
  ```

  当前 93,644 条 EgoEditData 编辑边应视为统一的 replacement 候选池；不能根据 prompt 中出现 `Remove` 就推断存在独立 remove 任务。
- **数据集大小**：93,644 条编辑边、37,319 个唯一 MP4；8 个 ZIP 合计约 245 GB。
- **抽查与推荐**：当前正式队列完成 22/100，source 22/22、pair 15/22 严格通过；主要风险在 pair 对应与轨迹，而不是输入视频自身质量。适合补 object replace、交互和遮挡，不补人物属性或 add/remove，现作为 replacement 主力候选继续进入第一轮。[官方数据](https://huggingface.co/datasets/liguang0115/EgoEdit)｜[第一轮计划](../lance_general_edit_finetune/1_specialist_filtering/README.md)

## 5. TripVVT-10K

- **数据形式**：`合成换装源视频 + garment 参考图 + 可选 mask/pose + 真实目标视频`。
- **主要任务**：dresses、upper-body、lower-body 的 reference-conditioned virtual try-on。
- **任务组织结构**：它只有 virtual try-on 一种训练任务，`cloth_type` 再按服装区域分为三类；train/test 是 metadata split，不是不同任务。每条 triplet 同时连接 source、真实 target、garment 图及可选结构条件。

  ```text
  TripVVT-10K/
  ├── input/                # 合成换装 source video
  ├── groundtruth/          # 未修改的真实 target video
  ├── input_cloth_ch/       # garment reference image
  ├── input_cloth_ch_line/  # garment line image
  ├── video_input_mask/     # video mask
  ├── pose_dw/              # pose video
  ├── train.jsonl           # 9,931 triplet
  └── test.jsonl            # 100 triplet

  cloth_type：upper_body / lower_body / dresses
  ```

  官方 test 与 train 存在 target video 复用，真正评估泛化时要按 `groundtruth/output`，最好再按人物身份重建 split。
- **数据集大小**：10,031 个 triplet，18 个 tar shard 约 72.6 GiB。
- **抽查与推荐**：15/15 人物相关、12/15 严格通过，身份、动作和背景保持明显好于多数文本编辑集，因而从原第 10 位上调。但它只做服装、依赖参考图，source 还是生成视频；仅在模型支持 garment reference 时作为高质量条件分支。[官方数据](https://huggingface.co/datasets/TripVVT/TripVVT-10K)｜[本地调研](tripvvt_research/tripvvt_report.md)

## 6. OpenVE-3M（当前项目决定不使用）

- **数据形式**：`源视频 + 单条指令 + 目标视频`，任务通过独立 metadata 类别组织。
- **主要任务**：local add、local remove、local change，以及背景、字幕、风格和相机编辑。
- **任务组织结构**：官方设计为 6 个 spatially-aligned 类别和 2 个 non-spatially-aligned 类别；本轮固定 revision 的文件树实际只提供前 6 类。它使用任务 CSV 描述有向编辑边，source/target 视频来自共享归档，并非每个 case 自包含。

  ```text
  OpenVE-3M/
  ├── spatially aligned
  │   ├── local_add
  │   ├── local_remove
  │   ├── local_change
  │   ├── background_change
  │   ├── global_style
  │   └── subtitles_edit
  └── non-spatially aligned
      ├── camera_edit
      └── creative_edit

  csv_files/<task>.csv：video;prompt;original_video
  videos/<task>.tar.gz.00 ...
  ```

  CSV 类别与视频物理目录并非严格一一对应，同一 pair 还可能形成正向、反向或多条 prompt 改写。`local_add/remove` 清洗后可映射 atomic task；`local_change` 还需拆成 object replace 和 attribute change。
- **数据集大小**：官方称 3M+ triplet，仓库约 2.23 TB；local add/remove/change metadata 分别为 400,498、337,463、479,256 条。
- **抽查与推荐**：文件技术质量整齐，但两批 30 条只有 19 个唯一无向 pair；19 条人物相关记录中 instruction 仅 10 条通过、8 条失败、1 条存疑。复核这 30 条 `local_change` contact sheet 时，没有发现明确的全屏动画或主要动画人物，但 `d95db` 是一条疑似生成真人的边界样本。完整 metadata 则能确认 `local_add`、`local_remove`、`local_change` 三个分支都混有明确写作 `cartoon character`、`animated character/girl` 或 `video game-like figure` 的记录，因此不能把任一分支直接视为纯实拍池。先用精确 metadata 词组排除显式案例，再由一级视觉域筛选兜底；同时仍须按无向 pair 去重并复核指令。[官方数据](https://huggingface.co/datasets/Lewandofski/OpenVE-3M)｜[本地调研](openve_research/openve_report.md)

## 7. ReCo-Data

- **数据形式**：`源视频 + 单条指令 + 目标视频`，另有原始物体 mask；任务独立组织。
- **主要任务**：object add、remove、replace，以及暂不优先的 global style。
- **数据集大小**：官方称 500K+ 编辑 pair；Hugging Face 仓库约 1.38 TB。
- **抽查与推荐**：新的独立审查从官方完整 metadata 与固定分片重新抽取 60 条，当前完成 55 条。`replace` 的 target 18/20、pair 15/20 严格通过，是最优先分支；`remove` target 17/20 但 pair 仅 10/20，说明画质强于指令/pair 可靠性；`add` 已完成 15 条中 target 11、pair 9 条严格通过。机器人/动画/CG 相关的 11 条中既有 3 条严格可用，也有 4 条明显失败和 4 条存疑，不能按 robot 关键词一刀切；应路由到角色 crop 的细致度、低模/游戏感和场景融合筛选。衣物编辑出现了可用质量较高的样本，后续单列真正的 clothes 目标，并与“新增一个穿衣人物”区分。三个分支均进入第一级，但 remove 采用更严格 pair filter。[官方数据](https://huggingface.co/datasets/HiDream-ai/ReCo-Data)｜[第一轮计划](../lance_general_edit_finetune/1_specialist_filtering/README.md)

## 8. ViViD

- **数据形式**：`masked target video + garment 参考图 + mask/DensePose + target video`。
- **主要任务**：garment-reference 条件下的服装重建和 virtual try-on。
- **数据集大小**：9,700 pair，单个 ZIP 约 32.06 GB。
- **抽查与推荐**：随机样本 15/15 严格通过，视频很整齐；但输入由 target 自身遮罩得到，不是人物穿 A 到穿 B 的真实编辑 pair。只作为服装重建辅助监督，不能因为抽查质量高而混入文本编辑主池。[官方数据](https://huggingface.co/datasets/alibaba-yuanjing-aigclab/ViViD)｜[本地调研](vivid_research/vivid_report.md)

## 9. Eevee

- **数据形式**：`masked target video + garment 参考图/人物图 + matched full-shot 与 close-up 目标视频`。
- **主要任务**：高分辨率 garment-reference 服装重建和 virtual try-on。
- **数据集大小**：论文称 9,364 条，发布包实测 9,363 个样本目录；5 个分卷约 215.00 GiB。
- **抽查与推荐**：15 条中 14 条严格通过，视频和服装细节很好；但同样不是 source-target 编辑 pair，而且当前数据许可不明确。在作者澄清许可前，只保留研究样本，不进入训练。[官方数据](https://huggingface.co/JianhaoZeng/Eevee)｜[本地调研](eevee_research/eevee_report.md)

## 10. Señorita-2M

- **数据形式**：`源视频 + 单条指令 + 目标视频`，任务按发布目录区分。
- **主要任务**：论文将任务划为 2 个大类、18 个子类；公开展示明确覆盖 object add/remove/swap、object stylization、global style transfer、multi-region、shape/size 和 motion editing。这里的 `object swap` 示例是把一个对象**替换**为另一个对象，不是交换两个对象的位置。
- **任务组织结构**：论文中的 18 个语义子类没有原样展开成 18 个下载目录。公开仓库把视频 tar 和 JSON metadata 合并为 9 个物理任务族，其中 `controlable_videos` 承载 motion、方向、形状/大小和多区域等较宽泛控制任务；若要抽取空间任务，必须继续按 instruction 做语义分类。

  ```text
  Senorita/
  ├── controlable_videos_upload/       # motion / direction / shape-size / multi-region 等
  ├── grounding_upload/                # object grounding / 局部对象控制
  ├── inpainting_upload/               # inpainting
  ├── local_style_transfer_upload/     # 局部对象风格化
  ├── obj_removal_videos_upload/       # object removal（第一批）
  ├── obj_removal_videos_upload2/      # object removal（第二批）
  ├── obj_swap_upload/                 # object replacement，不等于 location swap
  ├── outpainting/                     # outpainting
  ├── style_transfer_upload/           # 全局风格迁移
  ├── updated_data_controllable_videos.json
  ├── updated_data_inpainting.json
  ├── updated_data_local_style_transfer.json
  ├── updated_data_obj_addition_videos.json
  ├── updated_data_obj_grounding_videos.json
  ├── updated_data_obj_removal.json
  ├── updated_data_obj_swap_videos.json
  ├── updated_data_outpainting.json
  └── updated_data_style_transfer.json
  ```

  `object addition` 只有 metadata 文件、没有同名顶层目录，视频实际路径应以 JSON 为准；不能仅靠目录树拼接。当前若补一般空间能力，优先从 `updated_data_controllable_videos.json` 中筛 movement、direction、location、size/shape 等原子指令，而不是专门使用质量较差的 `obj_swap` 分支。
- **数据集大小**：约 1.95M–2M pair、仓库约 1.60 TB；`object_swap` 为 177,336 pair、约 76.2 GB。
- **抽查与推荐**：`object_swap` 是原子任务，但随机 15 条仅 6 条严格通过、人物相关 2 条，且全部只有 592×336。保留为低优先级通用 object swap 备选，不继续扩大下载。[官方数据](https://huggingface.co/datasets/SENORITADATASET/Senorita)｜[本地调研](senorita_research/senorita_report.md)

## 11. Ditto-1M

- **数据形式**：`源视频 + 指令 + 目标视频`。
- **主要任务**：local editing、global style 和 global freeform editing；local 包含对象修改、属性改变和局部变换，global freeform 包含场景、环境与复杂创意修改。
- **任务组织结构**：视频按生成管线和难度分为 6 个 target 目录，共享 `source/`；训练关系由 JSON 或 DiffSynth CSV 连接。`local_replace`、`global+local` 和 `sim2real` 是 metadata 视图，不是额外的独立视频目录。

  ```text
  Ditto-1M/
  ├── videos/
  │   ├── source/
  │   ├── local/
  │   ├── global_style1/
  │   ├── global_style2/
  │   ├── global_freeform1/
  │   ├── global_freeform2/
  │   └── global_freeform3/          # 相对更难的 freeform 编辑
  ├── training_metadata/
  │   ├── local.json
  │   ├── local_replace.json
  │   ├── global_style.json
  │   ├── global.json
  │   ├── global_freeform3.json
  │   ├── global+local.json
  │   └── sim2real.json
  ├── csvs_for_DiffSynth/             # 与上述训练视图对应的 CSV
  ├── source_video_captions/
  ├── mini_test_videos/
  └── enhancing_test_videos/

  单条 metadata：source_path + instruction + edited_path
  ```

  Ditto 没有官方的 location-swap 标签。若目标是一般空间编辑，只能从 `local.json`、`local_replace.json` 和部分 `global_freeform*.json` 中按 instruction 筛 object movement、direction、location、size/shape；`local_replace` 仍主要是对象替换，不能把 replace 自动当成位置交换。
- **数据集大小**：1M+ triplet，仓库约 2.29 TB；其中 local editing 约 530 GB。
- **抽查与推荐**：本地随机子总体只有 4/15 严格通过、人物属性 1/15；主要问题是贴图感、修改越界、背景变化和交互不合理。虽接口正确，但当前质量与人物密度都偏低，不再盲目扩量。[官方数据](https://huggingface.co/datasets/QingyanBai/Ditto-1M)｜[本地调研](ditto_research/ditto_report.md)

## 12. InsViE-1M

- **数据形式**：名义上是 `源视频/图像序列 + 指令 + 目标视频/图像序列`，动态视频与静态图来源混合。
- **主要任务**：宽口径通用编辑，混有对象操作、风格、打光、天气、背景和复合指令。
- **任务组织结构**：它不按编辑任务建目录，也没有官方 task label。所有任务共用一张 `train_insvie_align.csv`；`source_zip/` 和 `edited_zip/` 只按 source/target 及分卷保存文件，不能由物理目录判断 add、remove、replace 或空间任务。

  ```text
  InsViE/
  ├── source_zip/                 # source 文件分卷
  ├── edited_zip/                 # edited target 文件分卷
  └── train_insvie_align.csv
      ├── video                   # source/edited 共用的文件名或配对键
      └── instruct                # 自然语言编辑指令

  metadata 中混合的来源族：
  ├── Pexels                      # 动态视频为主
  ├── OpenVid                     # 动态视频
  ├── InstructPix2Pix             # 静态图来源
  ├── MagicBrush                  # 静态图来源
  └── 其他静态来源
  ```

  因此 InsViE 的“任务分类”只能是项目侧二次分类：先区分动态/静态来源，再对 instruction 做 atomic add/remove/replace、spatial、style/global 和 multi-edit 分类。按首词统计只适合召回候选，不能直接作为训练标签。
- **数据集大小**：1,019,570 条 metadata，视频发布约 765 GB。
- **抽查与推荐**：目前只有 metadata 调查，没有可靠的随机视频质量证据；任务首词也不能可靠代表原子操作。按质量优先原则排在最后，只有完成语义分类、去重和真实视频抽查后才考虑上调。[官方数据](https://huggingface.co/datasets/wyh6666/InsViE)｜[本地调研](insvie_research/metadata_report.md)

## 13. SpatialEdit-500K（图像辅助数据）

- **数据形式**：`源图像 + 空间编辑指令/控制 + 目标图像`；由 Blender 管线合成，不是 source-target 视频 pair。
- **主要任务**：object moving、object rotation 和 camera control。具体能力包括对象位置/位姿变化、三维方向旋转，以及相机 viewpoint、trajectory 和 framing 变化；它是覆盖多类空间变换的综合数据，不是专门的 location-swap 数据集。
- **任务组织结构**：公开仓库直接按三种任务建 WebDataset tar 分片。每个 tar 内保存配对图像与对应控制信息；文件名中的 worker、scene 和轨迹缩写只是生成/分片信息，不应当作语义任务标签。

  ```text
  SpatialEdit-500K/
  ├── object_moving/
  │   └── worker*-*.tar
  ├── object_rotation/
  │   └── worker*-*.tar
  └── camera_control/
      └── <scene>_<trajectory>-shard-*.tar

  顶层任务：object moving / object rotation / camera control
  ```

  该数据可用于补“空间关系和几何控制”的图像级辅助监督，但不能直接与当前视频 pair 等权混合：它没有真实视频时序，合成域也与实拍视频不同。若采用，建议作为小比例独立分支，并先验证现有训练代码能否正确消费 image pair。
- **数据集大小**：约 499K image pair，WebDataset 格式，Apache-2.0。
- **推荐**：若下一轮确实要补一般空间能力，它比从无标签大池中盲筛更直接；但它改变了数据模态，优先级仍低于先从现有视频数据中筛出高质量 spatial 子集。[官方数据](https://huggingface.co/datasets/EasonXiao-888/SpatialEdit-500K)｜[官方代码](https://github.com/EasonXiao-888/SpatialEdit)

## 14. OmniEdit-Bench（只用于测评）

- **数据形式**：非 reference 任务为 `源视频 + 指令`，reference 任务额外提供参考图/视频；不提供训练 target。
- **主要任务**：Spatial、Temporal、Reasoning、Audio、Reference 五条测评轨道。Spatial 的 240 条包含 direction change、location change、location swap、replacement、add/remove、number/state、attribute 与 global edit；Temporal 的 200 条包含动作、运动方向/轨迹/速度、相机控制和时序组合。
- **任务组织结构**：每条轨道各有一个 JSON，source video 按轨道/层级/细分类目录组织；生成结果统一写入 `eval_720p_10s/<model_name>/`，文件名必须与 `video_id` 一致。

  ```text
  OmniEdit-Bench/
  ├── Spatial/
  │   ├── Subject-Level/
  │   │   ├── Object_direction_change/
  │   │   ├── Object_location_swap/
  │   │   ├── Object_location_change/
  │   │   ├── Object_replacement/
  │   │   ├── Object_removal/
  │   │   └── Object_adding/ ...
  │   ├── Attribute-Level/
  │   └── Global-Level/
  ├── Temporal/
  ├── Reasoning/
  ├── Audio/
  ├── Reference/
  ├── spatial_benchmark.json       # 240
  ├── temporal_benchmark.json      # 200
  ├── reasoning_benchmark.json     # 50
  ├── audio_benchmark.json         # 100
  ├── reference_benchmark.json     # 200
  ├── eval_configs/prompts.json
  ├── eval_720p_10s/<model_name>/
  ├── evaluation_results/
  └── gemini_eval.py

  非 reference metadata：video_id + source_video + editing_instruction
  reference metadata：再加 reference
  ```

- **数据集大小**：共 790 条 benchmark task，其中 Spatial 240 条。
- **推荐**：这是当前能力缺口的 taxonomy 和固定测试集，不是训练候选。`Object_location_swap` 等样本必须与任何后续训练/筛选池隔离，避免 benchmark 泄漏；可以借它的类别定义去筛训练数据，但不能把它的源视频或指令放进训练集。[官方数据与测评说明](https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench)

## 当前建议

single-step 主干保留 **InsAVE remove、GOKU add/remove 和 EgoEdit replacement**。排除 OpenVE 后，唯一值得新增一轮正式抽查的是 **ReCo add/remove/replace**；应按三个官方分支分别判断，并先排除不合理的衣物/鞋裤 remove。CoinVE 等 single-step 完成后再作为 multi-step 阶段的数据。TripVVT、ViViD、Eevee 单列为参考图/重建条件分支，不混入纯文本原子编辑主池。

若下一轮从 atomic appearance edit 扩到一般空间编辑，先对 **Señorita `controlable_videos`、Ditto local/freeform 和 InsViE 动态来源**做 instruction 语义筛选与小规模视频抽查，不因为数据集“包含该任务”就整库混入。SpatialEdit-500K 只作为图像级空间辅助分支；OmniEdit-Bench 仅用于 taxonomy 与独立测评，严格禁止训练泄漏。Señorita、Ditto、InsViE 在完成这轮专项筛选前仍不扩量。
