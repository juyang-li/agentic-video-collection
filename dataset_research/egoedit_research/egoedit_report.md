# EgoEditData 随机 pair 阶段结论

## 结论

EgoEdit 的发布结构和编辑粒度适合作为 **object replacement 的原子监督候选**：本次 15 条完整随机样本的 instruction 都只要求替换一个被手持或操作的物体，没有 CoinVE 式的多任务串联。

但它不能补齐当前的 add/remove 缺口，也不是人物属性数据。发布的 93,644 条训练编辑边中未识别出独立的纯 add/remove 子总体；`Remove...` 开头的 15 条指令也全部是“remove X and replace with Y”。第一视角画面虽然大量出现手臂和手，但编辑目标是物体，因此按本项目口径，15/15 都不属于人物属性编辑。

当前判断：**保留为 atomic object-replace 辅助集，不作为人物属性主数据，也不按官方概述强行拆出 remove。**

## 可获取性与组织

- 作者镜像已授权并可读取，固定 revision 为 `a9bd3faafe51a581b359acd7c6ad5c432508ff60`。
- data table：37,716 行；edit table：93,644 行；全部编辑行都能连接到 source/target URL 和非空 instruction。
- 8 个视频 ZIP 合计 245,014,955,013 bytes（约 245.0 GB / 228.2 GiB），共有 37,319 个唯一 MP4 成员。
- data table 行数比唯一视频多 397，是 URL 重复，不是归档缺文件。
- ZIP member 使用 stored method，服务器支持 Range；本次只拉取随机命中的 30 个视频，共 153,674,111 bytes，没有下载全量 ZIP。
- Snap Inc. Non-Commercial license，需要与项目后续数据使用方式单独核对。

## 抽样

- annotation SHA256：data table `81d0f5bb6fe03d49f8a34a922f306c33b356419f3283945380400df8c5f08efe`；edit table `9a21c8f12f519377365a4bcc8385bc3139e9554630458cccd76417dd33a89a8c`。
- population：93,644 条完整连接的编辑边。
- seed：`20260901`。
- method：对 metadata 行号无放回均匀随机抽 15 条。
- 覆盖：15 个不同 scene key、30 个不同视频文件。

这是完整发布编辑表上的随机样本，不是官方展示样例，也没有按人物、编辑质量或对象类别挑选。

## 技术检查与预览发现

- 30 个原始 MP4 全部可解码；均为 81 帧、15 fps、5.4 秒且无音轨。
- 每对帧数、帧率和时长一致；6/15 对分辨率完全一致，其余 9/15 的 source/target 分辨率不同。
- 15 个左右对照文件均已转为 H.264/AVC、`yuv420p`、fast-start，并通过完整解码。
- 初步六帧预览中，13 组的对象替换与动作背景大体对应。
- `c8c6` 是明显发布数据错配：指令要求防滑垫替换为蓝绿色清洁布，target 却是另一段卧室置物架视频。
- `06142` 的画笔→螺丝刀替换在前段成立，但 target 后段比 source 更早转向浴缸区域，动作轨迹对应较弱。

以上是定位问题用的预览记录，不代替用户人工审核。`egoedit_audit.csv` 仍按项目规则保留默认值，观看 MP4 后只需修改失败或存疑项。

## 适用边界

- 优点：单物体 replacement、真实第一视角交互、手遮挡和大相机运动丰富；数据关系能通过表连接精确表达。
- 缺点：不是人物属性；没有可用的独立 add/remove 分栏；部分 target 画幅变化，随机样本也出现错配和轨迹偏离。
- 决策：可作为 object-replace 和交互遮挡的补充监督；若主线坚持人物属性 add/remove/replace，不建议因为已获授权就直接扩量下载 245 GB。

官方来源：[EgoEdit code](https://github.com/snap-research/EgoEdit)、[授权数据镜像](https://huggingface.co/datasets/liguang0115/EgoEdit)。
