# OpenVE-3M 数据笔记

- 论文：https://arxiv.org/abs/2512.07826
- 数据集：https://huggingface.co/datasets/Lewandofski/OpenVE-3M
- 许可证：CC BY-NC 4.0。
- 固定 revision 的官方仓库实际提供 6 个 spatially-aligned 类别、107 个视频分卷，约 2.03 TiB；camera/creative 类别未出现在文件树中。
- `local_change.csv` 有 479,256 条记录，其中 230,808 条的 source 和 target 都位于 local-change 归档。
- 视频是连续 gzip tar 的分卷，后续分卷不能单独解压，也不能按 tar member 随机 Range 提取。

## 任务组织结构

OpenVE-3M 官方设计为 8 类，其中 6 类保持 source/target 空间和运动对齐，2 类允许镜头或运动发生较大变化：

```text
OpenVE-3M/
├── spatially aligned
│   ├── local_add          # 局部添加对象
│   ├── local_remove       # 局部删除对象
│   ├── local_change       # 局部替换或属性变化
│   ├── background_change  # 更换背景
│   ├── global_style       # 全局风格转换
│   └── subtitles_edit     # 字幕编辑
└── non-spatially aligned
    ├── camera_edit        # 多镜头或镜头编辑
    └── creative_edit      # 允许新动作和较大内容变化
```

官方定义为 8 类；本轮固定 revision 的实际文件树只提供了前 6 个 spatially-aligned 类别，因此 camera/creative 没有进入可下载总体。

OpenVE 使用“CSV 边 + 共享视频文件”，而不是 GOKU 那种每个 case 自包含的 JSON：

```text
csv_files/
├── local_add.csv
├── local_remove.csv
├── local_change.csv
└── ...

videos/
├── local_add.tar.gz.00
├── local_add.tar.gz.01
├── local_remove.tar.gz.00
└── ...
```

CSV 以分号分隔，每行字段为 `video;prompt;original_video`，对应：

```text
original_video + prompt -> video
```

任务 CSV 和视频的物理目录并非严格一一对应。例如 `local_remove.csv` 可能引用 `local_add/` 中的视频，同一组视频还可能形成正向、反向或多条改写指令。因此整合前必须按无向视频 pair 去重，再检查编辑方向和指令是否与画面一致。

对当前 atomic 微调：`local_add`、`local_remove` 可在清洗后映射为 `atomic_add`、`atomic_remove`；`local_change` 需要进一步拆成 `object_replace` 和 `attribute_change`。background、style、subtitles、camera 和 creative 第一轮暂不混入。

本轮固定读取 `local_change_new.tar.gz.00` 开头 2,147,483,648 字节，确认 4,189 个完整 MP4 member；与 metadata 连接后得到 39 条可完整提取记录。第一批以 seed `20260821` 随机抽 15 条；第二批排除这 15 条后，以 seed `20260822` 从剩余 24 条随机抽 15 条。无需扩大到 4 GiB。

每批曾抽取 30 个 source/target 角色文件，各有 24 个唯一 MP4。合计 30 条 metadata 记录只有 29 个唯一有向 pair、19 个唯一无向 pair、38 个唯一 MP4。60 个角色文件均通过解码检查；每批均为 13 对 1280×720、2 对 720×1280，每对 source/target 的分辨率、帧数和 fps 完全一致。独立角色文件在 comparison 校验完成后已作为重复中间文件删除。

30 个 comparison 均转为 H.264 `avc1`、`yuv420p` 并启用 faststart，便于 VS Code 和本地播放器直接打开。
