# GOKU-2M 数据笔记

- 论文：https://arxiv.org/abs/2606.30599
- 数据集：https://huggingface.co/datasets/bigfacing/GOKU-2M
- 许可证：CC BY-NC 4.0。
- 官方将约 2M 个编辑 pair 分在两个 gated 仓库，总发布量约 9.65 TB；本轮使用的第一个仓库约 5.11 TB。
- 每个任务由一个 metadata tar 和若干可独立解包的未压缩视频 tar 组成，case 内直接提供 `source_video`、`edited_video` 和 `instruction_en`。
- `swap_alter` 有 32 个视频分卷，总计约 519.29 GiB；随机命中的 `part05` 为 24,782,745,600 bytes（23.08 GiB），metadata tar 为 443,934,720 bytes（423.37 MiB）。

## 任务组织结构

GOKU-2M 逻辑上是一个数据集。官方因为总量较大拆成两个 gated Hugging Face 仓库，**不是按照 remove / 非-remove 划分**：

```text
bigfacing/GOKU-2M/
├── add/                 # 添加物体
├── remove/              # 删除物体
├── swap_alter/          # 替换物体或改变对象属性
├── reference_add/       # 根据参考图添加物体
├── reference_swap/      # 根据参考图替换物体
├── style_transfer/      # 全局风格转换
└── camera/              # 镜头运动编辑

Goku-2M/GOKU-2M/
├── subject_movement/    # 改变主体动作或运动
└── multi_task/          # 多步骤组合编辑
```

第二个官方仓库是：https://huggingface.co/datasets/Goku-2M/GOKU-2M 。本地 `goku_research/remove/` 只是第二轮 remove 抽查的结果目录，不代表官方把 remove 单独发布为另一个数据集。

普通单任务的物理结构为：

```text
<task>/
├── <task>.jsons.tar
├── <task>.videos.part01.tar
├── <task>.videos.part02.tar
└── ...

解压后：
├── videos/<case_id>/source.mp4
├── videos/<case_id>/edited.mp4
└── jsons/combine_json/<case_id>_all.json
```

JSON 直接记录 `source_video`、`edited_video`、`instruction_en`、分辨率和 fps，因此普通监督形式是：

```text
源视频 + 指令 -> 目标视频
```

`reference_add` 和 `reference_swap` 额外包含 `reference.jpg`，监督形式是“源视频 + 指令 + 参考图 -> 目标视频”。`multi_task` 包含有顺序的多个步骤、中间视频和最终视频，不应和 atomic pair 直接混合。

对当前 atomic 微调：`add`、`remove` 可直接作为候选任务；`swap_alter` 还需根据指令和画面拆成 `object_replace`、`attribute_change`、`background_or_region_replace`。第一轮暂不纳入 reference、style、camera、subject movement 和 multi-task。

本轮固定官方 revision，在 32 卷中随机选 `part05`，再从卷内完整 case 随机抽 15 对。视频 tar 中有 8,262 个 source/edited 完整 case；其中两个对应 annotation 是 0-byte JSON，故最终可抽总体为 8,260。

30 个原始角色 MP4 均可完整解码；每对都是 127 帧、25 fps、5.08 秒，source/target 规格完全一致。12 对为 1248×720，2 对为 1272×720，1 对为 720×744。15 个 comparison 均转为 H.264 `avc1`、`yuv420p` 并启用 faststart。

原始角色 MP4 只作为抽取和验证的中间文件，comparison 及 ZIP 校验完成后已删除；23.08 GiB 视频 tar 和 metadata tar 也不在项目中长期保存。
