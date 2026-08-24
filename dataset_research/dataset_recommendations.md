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
- **抽查与推荐**：初步逐帧预览中 13/15 大体对应，但发现 1 条明显错配和 1 条后段轨迹偏离；正式人工审核表尚未完成，因此证据强度低于前三名。适合补 object replace、交互和遮挡，不补人物属性或 add/remove。[官方数据](https://huggingface.co/datasets/liguang0115/EgoEdit)｜[本地调研](egoedit_research/egoedit_report.md)

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

## 6. OpenVE-3M

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
- **抽查与推荐**：文件技术质量整齐，但两批 30 条只有 19 个唯一无向 pair；19 条人物相关记录中 instruction 仅 10 条通过、8 条失败、1 条存疑。接口和人物覆盖很理想，但真实语义质量不足以排在前列，必须先按无向 pair 去重并复核指令。[官方数据](https://huggingface.co/datasets/Lewandofski/OpenVE-3M)｜[本地调研](openve_research/openve_report.md)

## 7. ReCo-Data

- **数据形式**：`源视频 + 单条指令 + 目标视频`，另有原始物体 mask；任务独立组织。
- **主要任务**：object add、remove、replace，以及暂不优先的 global style。
- **数据集大小**：官方称 500K+ 编辑 pair；Hugging Face 仓库约 1.38 TB。
- **抽查与推荐**：形式最贴合目标，官方展示也是每任务随机 50 条而非挑选样例；但现有审核曾把左右 source/target 方向看反，当前 CSV 不能形成可信通过率。观察中还存在交互逻辑、背景变化和手部/衣物伪影，因此先降级，等人物属性候选按正确方向复核后再决定是否上调。[官方数据](https://huggingface.co/datasets/HiDream-ai/ReCo-Data)｜[本地调研](reco_research/reco_report.md)

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
- **主要任务**：18 类任务，包括 object add/remove/swap 和 global/local stylization。
- **数据集大小**：约 1.95M–2M pair、仓库约 1.60 TB；`object_swap` 为 177,336 pair、约 76.2 GB。
- **抽查与推荐**：`object_swap` 是原子任务，但随机 15 条仅 6 条严格通过、人物相关 2 条，且全部只有 592×336。保留为低优先级通用 object swap 备选，不继续扩大下载。[官方数据](https://huggingface.co/datasets/SENORITADATASET/Senorita)｜[本地调研](senorita_research/senorita_report.md)

## 11. Ditto-1M

- **数据形式**：`源视频 + 指令 + 目标视频`。
- **主要任务**：local editing、global style 和 global freeform editing。
- **数据集大小**：1M+ triplet，仓库约 2.29 TB；其中 local editing 约 530 GB。
- **抽查与推荐**：本地随机子总体只有 4/15 严格通过、人物属性 1/15；主要问题是贴图感、修改越界、背景变化和交互不合理。虽接口正确，但当前质量与人物密度都偏低，不再盲目扩量。[官方数据](https://huggingface.co/datasets/QingyanBai/Ditto-1M)｜[本地调研](ditto_research/ditto_report.md)

## 12. InsViE-1M

- **数据形式**：名义上是 `源视频/图像序列 + 指令 + 目标视频/图像序列`，动态视频与静态图来源混合。
- **主要任务**：宽口径通用编辑，混有对象操作、风格、打光、天气、背景和复合指令。
- **数据集大小**：1,019,570 条 metadata，视频发布约 765 GB。
- **抽查与推荐**：目前只有 metadata 调查，没有可靠的随机视频质量证据；任务首词也不能可靠代表原子操作。按质量优先原则排在最后，只有完成语义分类、去重和真实视频抽查后才考虑上调。[官方数据](https://huggingface.co/datasets/wyh6666/InsViE)｜[本地调研](insvie_research/metadata_report.md)

## 当前建议

第一梯队先保留 **InsAVE remove、CoinVE 和 GOKU remove**：它们的限制不同，但已有较明确的视频质量证据。**EgoEdit** 先完成人工审核；**OpenVE** 先去重和过滤错误指令；**ReCo** 只复核筛出的人物属性候选，不再依据旧 CSV 判断质量。TripVVT、ViViD、Eevee 的视频本身较好，但单列为参考图/重建条件分支，不混入纯文本原子编辑主池。
