# Real-person drift stress benchmark v1

这是独立于 `real_person_multiturn_benchmark_v2` 的真实人物多轮漂移专项集，目的不是扩大常规覆盖面，而是直接观察模型能否撤销早期编辑、恢复指定历史状态，并保留不应撤销的中间编辑。

> **当前结论（2026-10-16）**：回撤/历史状态恢复任务继续保留为 benchmark，但现有 Lance Base、CoinVE step677 和 gtfresh step1150 尚不能可靠完成它；旧生成结果无效，相关输出、日志及本地/远端审核包均已删除。本目录只保留定义、源视频、检查点和复现脚本。

当前模型也未针对真实相机运动编辑训练，因此这里的 camera 结果不作能力结论；后续有效评测移除 `camera_id`，先聚焦模型确实训练过的编辑轴。新的 Base+gtfresh History-ViT 比较从头推理，不复用本实验的无效生成结果。

## 数据规模

- 12 个新的 IVEBench short 真实视频，与原 30 视频 benchmark 的 SHA-256 重叠数为 0；
- 每条 4 轮，共 48 条指令；
- 3 类结构各 4 条：重复编辑/撤销、历史中间状态恢复、延迟回撤并保留中间编辑；
- 每模型 24 个重点漂移检查点；三个模型共 72 个专项人工审核单元。

四条重复循环分别测试墨镜、毛线帽、明显的全胡须和围巾。动作不出现在第一轮，只在后续轮使用清晰可见的挥手、竖拇指、交叉双臂或恢复原动作。`benchmark.jsonl` 保存完整链；`turns.csv` 和 `checkpoints.csv` 分别是轮次与重点比较的平面表。

## 视频筛选

筛选复用 3 个已经冻结的 v1 高质量备用样本，再用非个人 `LIJUYANG_ICLR` 凭据通过 OpenRouter 调用 `qwen/qwen3.8-omni-flash` 补充 9 个。固定 Alibaba provider、关闭 fallback、`data_collection=deny`，只做三帧适用性判断，不让模型生成指令。实际报告费用为 `$0.00766551`。

两个旧备用样本（骑马背面、直升机旁低头）在最终人工三帧复核时被剔除。所有 12 个源视频均要求成人真实人物、单一明确主体、脸和上半身清楚、镜头稳定。指令由固定模板编写，因此没有额外的指令生成 API 消耗。

## 历史三模型推理（结果已删除）

比较对象：

1. `lance_base`；
2. `coinve_real_step677`（仅真实 CoinVE 微调）；
3. `gtfresh_step1150`（真实 CoinVE + 师兄合成数据）。

三者统一使用 30 diffusion steps、shift 3.5、text/ViT CFG 1.0、seed 42 和同一训练预处理。每轮输出自回填为下一轮输入。共生成 144 个视频。

adapter 加载不是仅检查文件存在：

- CoinVE：504/504 张量与模型参数匹配，252 个 LoRA-B 非零；
- gtfresh：1016/1016 张量匹配，503 个 LoRA-B 非零。

三个 checkpoint 同时改变了数据与若干训练配方，因此结果只能解释为现有 checkpoint 的实际横向比较，不能视为严格的数据单变量消融。

## 历史人工审核包（已废弃并删除）

以下名称仅记录曾经的交付，不再作为有效结果；本地与百度网盘文件均已删除：

- 本地：`review/real_person_drift_stress_three_model_human_review_20261009.zip`
- 百度网盘：`/apps/bypy/lance_review/real_person_drift_stress_three_model_human_review_20261009.zip`
- SHA-256：`33a627c299ea052d6313ce54d3d947ea5b13df7428a8a220f09365ab9546a58e`

审核包有 12 张卡、156 个 H.264/yuv420p/faststart 视频，其中 144 个是生成结果。全部视频均已完整解码验证；离线 Chrome 验收覆盖 72 个默认标注、四张卡的四模型行播放、CSV/JSON 导入导出，并确认没有外部请求。

每个重点检查点按三个模型分别填写：本轮指令完成、历史状态恢复、无残留编辑、中间编辑保持、人物背景稳定、时序一致、视觉质量和总体无漂移。默认值均为 `1`，失败改 `0`，存疑改 `2`。

## 复现入口

- `scripts/screen_sources.py`：可续跑的非个人 OpenRouter 视频筛选；
- `scripts/build_benchmark.py`：冻结并规范化 12 条链；
- `scripts/run_model.py`、`scripts/launch_models.py`：三模型自回填推理；
- `scripts/build_review.py`：完整解码并构建离线审核网页；
- `scripts/test_review.py`：离线 Chrome 验收；
- `scripts/finalize_review.py`：校验、SHA-256 和 ZIP 封装。
