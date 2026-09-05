# Lance removal 500-step 消融评测

> 日期：2026-08-25  
> 归档状态：评测已完成；旧 `lance_remove_pilot/` 训练目录及 checkpoint 已于 2026-09-05 删除。本目录只保留历史结论与已有评测产物，不能直接继续训练或重新推理。轻量训练脚本、配置和 manifest 的删除前快照位于 Git commit `69735dc`；checkpoint、媒体与 runtime 副本未进入 Git。当前工作以 [`../../lance_general_edit_finetune/README.md`](../../lance_general_edit_finetune/README.md) 为准。

## 目的

在原始 Lance 已冻结的 15 条 OmniEdit-Bench Spatial 样本上，对比：

1. 原始 Lance base；
2. 300 InsAVE + 200 GOKU、训练 500 steps 的 final checkpoint；
3. 500 InsAVE-only、训练 500 steps 的 final checkpoint。

推理统一使用 base 评测的 30 steps、seed 42、CFG 4.0 和同一个 `benchmark_all.json`。四联 comparison 固定为：

```text
SOURCE | BASE | MIXED | INSAVE
```

## 目录

```text
lance_remove_ablation/
├── README.md
├── run.json                 # 历史运行配置，其中旧 checkpoint 路径已失效
├── comparison_config.json
├── loss_analysis.json
├── loss_comparison.svg
├── configs/                 # 自动生成的 pending config
├── technical/               # 两个 checkpoint 的完整解码记录
├── outputs_raw/             # mixed / InsAVE 原始模型输出
├── comparison/              # 15 个四联 MP4、指令和 inventory
├── contact_sheets/          # 稀疏帧观察图
├── scores.csv               # 三个模型的人工评分入口
├── preliminary_observations.csv # 五帧接触表的初步观察，不替代正式评分
└── logs/                    # 原始推理日志
```

推理、完整解码、comparison 和 contact sheet 使用 `../lance_tools/` 中的通用脚本；以后新增 checkpoint 只需增加模型输出目录和 comparison panel 配置，不再复制 base 专用脚本。

## 归档与复现状态

原始 500-step checkpoint 已随 `lance_remove_pilot/` 删除，旧复跑入口也已移除，不能从当前工作区重新生成这两组模型输出。已有 comparison、contact sheet、loss 统计和技术清单仍可用于核查历史结论。Git commit `69735dc` 只恢复轻量代码与 metadata，不恢复 checkpoint 或媒体。

## Loss 的正确解读

每个 optimizer step 只有一个随机 3 秒视频片段，并带随机扩散 timestep，因此 raw loss 天然会明显振荡，单个节点上升不代表训练反向。应比较固定窗口均值和 rolling mean，同时用固定生成评测判断实际编辑能力。

| 实验 | 前 100 step 均值 | 后 100 step 均值 | 变化 | 500-step 线性斜率 |
| --- | ---: | ---: | ---: | ---: |
| mixed 300+200 | 0.11169 | 0.08149 | -27.0% | -4.42e-5 / step |
| InsAVE 500 | 0.12808 | 0.11085 | -13.5% | -2.70e-5 / step |

两条曲线的整体斜率和窗口变化均为下降，gradient norm 全程有限且非零，因此没有训练发散证据。mixed 的聚合 loss 更低不能直接解释成模型更好：其 193 个 GOKU step 的平均 loss 是 0.06412，而 307 个 InsAVE step 是 0.12497，数据源难度差异会显著改变总体均值。

原始逐步曲线和统计分别见 [`loss_comparison.svg`](loss_comparison.svg) 与 [`loss_analysis.json`](loss_analysis.json)。

## 固定 15 项评测结果

所有新输出均为长边 848、短边 480、12 FPS（按源视频保留横竖方向），mixed 与 InsAVE-only 各 15/15 完整解码。四联视频统一为 1920×540、12 FPS，面板顺序固定为：

```text
SOURCE | BASE | MIXED | INSAVE
```

基于每条视频 5 个等间隔帧的初步人工观察，当前结论是：**没有证据表明 300+200 混合优于 500 InsAVE；两组 final checkpoint 都出现明显的非 removal 能力退化。** 相对 base 的综合观察计数如下：

| 结果标签 | mixed 300+200 | InsAVE 500 |
| --- | ---: | ---: |
| better | 0 | 2 |
| similar_failure | 3 | 3 |
| different_failure | 2 | 2 |
| worse | 10 | 8 |

这里的 `similar_failure` 表示两者都未完成指令，`different_failure` 表示失败方式不同、无法可靠排序；它们都不是成功。逐项依据保存在 [`preliminary_observations.csv`](preliminary_observations.csv)，正式盲评仍应填写 [`scores.csv`](scores.csv)。

主要现象：

- 3 条 removal 中，base 为 0/3，mixed 仍为 0/3；InsAVE-only 在“移除女人”上取得 1 个清晰成功，但玻璃罩和透明杯仍失败。
- InsAVE-only 在“kiwi 换完整橙子”上也比 base 更接近目标；这是第二个 `better`，但不属于 removal。
- 3 条 object adding 中，两组微调均出现目标延迟出现、形态不稳定或后段消失，base 整体更稳定。
- replacement 中，两组在“天鹅换鸭子”后段让目标退化/消失；appearance 多数变为不执行编辑。
- clothes 中 mixed 经常不执行衣服替换；InsAVE-only 在两个样本后半段出现人物融化或完全消失。

这说明 loss 的下降更多反映训练目标被优化，尚未转化成稳定的编辑质量提升。final checkpoint 似乎倾向于抑制或延迟编辑，同时存在时序退化。由于冻结 benchmark 每类只有 3 条、removal 也只有 3 条，这是一项方向性诊断，不是有统计效力的最终结论；特别是它不足以单独判断 GOKU 域内泛化。

## Go/No-Go

当前对直接扩到 1K 给出 **No-Go**：先不要继续增加同配方训练量。更有信息量的下一步是：

该结论现仅作为历史负结果保留，不再安排 step 249 补测、旧配方续训或 1K 扩量；后续实验统一进入通用编辑微调主线。

完整四联视频位于 [`comparison/`](comparison/)，可分发压缩包为 [`lance_remove_ablation_comparison_15_h264.zip`](lance_remove_ablation_comparison_15_h264.zip)。

## 通用评测脚本

本轮没有复制 base 专用评测逻辑。以下通用脚本仍由当前评测工作复用：

- `../lance_tools/run_video_edit.sh`：按模型路径、benchmark config 和 run 名称生成缺失输出；
- `../lance_tools/verify_outputs.py`：逐文件完整解码并写技术清单；
- `../lance_tools/build_comparison.py`：根据 JSON 面板配置生成 N 联 comparison、评分表和 ZIP；
- `../lance_tools/build_contact_sheets.py`：从 comparison inventory 生成稀疏帧接触表；
- `../lance_tools/analyze_training_metrics.py`：从任意训练 run 的 CSV 生成统一 loss 统计和曲线。
