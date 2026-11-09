# 第二级完整 Pair 筛查

本级直接判断 `source + instruction -> target` 是否编辑成功。它替代原先计划中的 Stage 1.6，不再额外维护一套基于对象词表和 DINO 差异图的间接 pair 规则。

## 当前生产任务（2026-09-06）

首批 4K 已完成 Gemini-only 筛选，权威结果见 [Stage 2 正式报告](../3_lance_lora_finetuning/STAGE2_PRODUCTION_REPORT.md)。下一批从新完成的 18,662 条 Stage-1 survivor 中按原配额筛出新增 4,000 条，暂不加入 CoinVE；沿用同一 Gemini、prompt 和视觉输入。512 条小批已完成并核验，保留 282 条，记录费用 $2.937315；按实测通过率更新总预算为 $45–55，上限 $55。用户已更换余额充足的 API key 并批准上传，但新 key 被 OpenRouter 服务条款 HTTP 403 拦截，最小文本调用同样失败，当前已停止等待访问权限恢复；此次新 key 的失败请求未计费。实时进度和续跑入口见 [下一批 4K](artifacts/stage2_next4k_20260906/README.md)。下方早期 Qwen 和本地模型流程保留为历史实验记录。

## 当前二分类评测结论（2026-09-04）

人工明确标签现共 230 条（176 accepted、54 rejected）；人工 uncertain 无论备注倾向均排除。旧 139 条拆为 train/validation，本轮新完成的 91 条作为未参与调优的 test。Gemini 3.8 Flash 使用冻结 provenance-aware v12 prompt 后，test 的坏样本召回率为 84.6%、accuracy 为 82.4%、好样本误杀率为 18.5%；remove 子集召回率为 90.5%。完整数据构成、划分、历史 Qwen72B v12 可比范围、prompt 选择和分组结果见 [`systematic_binary_dataset_report.md`](./systematic_binary_dataset_report.md)。

## 系统二分类人工基准 v1（2026-09-03）

第二级模型比较改用同一套人工基准，人的审核入口保留 `accepted / rejected / uncertain` 三项，但最终计分集只保留明确的 `accepted / rejected`；`uncertain` 无论备注偏正面还是偏负面都不再代判，直接排除。每个审核页将 SRC、TARGET、INS 和数据来源一体展示，只填写一次整体结论。已知单侧生成样本按 provenance 重排为 `REFERENCE_VIDEO -> GENERATED_VIDEO`；反向 add 按实际生成方向作为 remove 审查，EgoEdit 两侧均为 generated 时保持训练方向并明确标示。ReCo 的训练 src/tar 明确、逐 pair 生成 lineage 未发布，因此标为 `generation_side_unresolved`：instruction following 和 content preservation 按 `SRC_VIDEO -> TARGET_VIDEO`，visible artifact 检查两侧。

数据分成两个用途，不能混用：

- `development`：复用此前人工明确判为 usable/unusable、且仍属于当前任务范围的 139 条，映射为 111 accepted、28 rejected。其中 remove/add/replace 为 79/14/46，EgoEdit/GOKU/ReCo 为 24/57/58。它们可以用于提示词、阈值和模型选择，但不能作为调过这些规则后的盲测成绩。旧 uncertain 全部排除；尤其没有采用将 holdout40 的倾向性备注重映射成 keep/delete 的后处理标签。
- `test_v1`：从当前 4,003 对中排除 782 个曾被人工或模型触碰的 ID 后，以固定种子 `20260903` 新抽 300 条，且与历史集合零重合。构成为 GOKU 反向 add-remove 50、GOKU remove 50、ReCo remove 100、ReCo add 40、ReCo replace 45、EgoEdit replace 15；实际审查操作为 remove/replace/add = 200/60/40，remove 占 66.7%，EgoEdit 占 5%。在人工结果回来前不对这 300 条运行候选模型。

两部分最多形成 439 条明确二分类记录；在新审核尚未排除 uncertain 前，remove 候选为 279 条，占 63.6%，EgoEdit 为 39 条，占 8.9%。正式横向比较只报告 `test_v1`，development 结果单列。若 test_v1 最终 human rejected 数量过少，不回填或重抽 v1，而是另建 removal-heavy challenge set 并单独报告。

核心指标固定为：

- 坏样本召回率：`human rejected 且 model rejected / 全部 human rejected`；
- 好样本误杀率：`human accepted 且 model rejected / 全部 human accepted`。

同时给出 reject precision 和 accuracy 作为辅助，并按 audit operation、dataset、branch 分组。待比较模型在计分前也必须输出二分类 accepted/rejected，不能用 uncertain 覆盖率规避错误。总体指标对应刻意偏重 remove 的诊断分布，不解释为 4,003 对自然分布下的淘汰率。

冻结清单和历史明确标签位于 `artifacts/systematic_binary_benchmark_v1/`。抽样脚本为 `scripts/build_systematic_binary_benchmark.py`，历史标签整理脚本为 `scripts/build_systematic_binary_development_labels.py`。人工 CSV 回传后执行：

```bash
python3 2_vlm_filtering/scripts/finalize_systematic_binary_benchmark.py \
  --human-review /path/to/stage2_systematic_binary_benchmark_v1_review.csv
```

该命令要求 300/300 全部完成，保存原始 uncertain，但只把二分类项写入 `labels/test_v1_binary_labels.csv`，并与 development 合并成 `labels/binary_label_registry.csv`。不同模型的全量二分类结果可用 `scripts/evaluate_systematic_binary_models.py` 统一计算上述指标。

盲审包已上传百度网盘：

```text
/apps/bypy/lance_review/stage2_systematic_binary_benchmark_v1_review_bundle.tar
```

2026-09-04 已按 provenance-aware 规则订正并覆盖上传：大小 `348,006,400` bytes，SHA256 为 `bd532ce219477105645f6c0e629d71b5e5b213e3f44b6d61ed80f6799eaff3a5`。300 个 sample ID、顺序和 survey version 均未变化，页面逐 pair 显示数据集/分支、repo revision、sample/pair ID、provenance 状态及判定依据，不显示任何待比较模型结论。覆盖解压到原目录可沿用浏览器中已有记录；操作前仍建议先导出 JSON 备份。

同步的构成与逐 pair provenance 明细表位于：

```text
/apps/bypy/lance_review/stage2_systematic_binary_benchmark_v1_composition.xlsx
```

大小 `48,536` bytes，SHA256 为 `cea7e3faae3762238c62c960bdf0259dc1112bf97a3699d7c9047e3d7c146aa9`。

## 当前输入与顺序

第一级冻结 pilot 不重算。`1_specialist_filtering/artifacts/filter_run_v2/stage1_next_input.csv` 当前包含 4,003 对：3,737 条原 Stage 1.1 accepted、266 条 uncertain。相较旧 4,018 入口，新增排除了 15 条被旧 metadata 解析误判成 person/object 的 GOKU 衣物 add/remove。处理顺序为：

1. Stage 1.2 对其中 683 条 ReCo robot/CG 打分；未经人工校准时全部只能是 uncertain。
2. OpenRouter `qwen/qwen3-vl-32b-instruct` 同时查看 source、target 和 instruction，尝试判断编辑是否成功。
3. Stage 1.2 或 Qwen 任一硬失败即拒绝；所有适用阶段明确通过才接受。
4. 原 Stage 1.1 uncertain 满足上一条时允许转正。

Stage 1.5 与 Stage 1.6 均不执行。

## 当前 Qwen3 输入与输出

程序完全使用 CPU 解码和拼图，不申请本地 GPU。当前实验输入固定为 5 张全图 pair（单侧长边 640）和 2 张时序 ROI 联系表；联系表内部含 9 个归一化时刻的同坐标局部 pair（单侧长边 640）。已知单侧生成数据统一重排为左侧 `REFERENCE_VIDEO`、右侧 `GENERATED_VIDEO`；EgoEdit `edited→edited` 保持训练顺序并检查两侧。显式 `generation_side_unresolved` 保持 `SRC_VIDEO -> TARGET_VIDEO`，其中 instruction following/content preservation 按训练方向、visible artifact 查两侧；普通 `unknown` 仍直接报错，不猜测。原始 `TRAIN_INSTRUCTION` 不改写，另传 `AUDIT_OPERATION` 与 `AUDIT_RELATION`。当前 provenance-aware 中文提示词见 [`judge_prompt_timecard_v12_provenance_aware.txt`](./judge_prompt_timecard_v12_provenance_aware.txt)，执行脚本为 [`scripts/run_openrouter_qwen3_judge.py`](./scripts/run_openrouter_qwen3_judge.py)。

2026-09-04 之前的 32B/72B、PAL4VST 和 VEFX 实验结果保留为历史调研，不追改原始输出；其中涉及 ReCo 且把 target 当作唯一生成侧的分数不能直接用于当前系统基准或全量筛选。后续模型运行必须读取订正后的 manifest 和 v12 provenance-aware 路由。

模型必须输出严格 JSON，包含：

- 总决策 `accepted / rejected / uncertain`；
- `instruction_following`、`visible_artifact`、`content_preservation`；
- F1-F5 全图逐帧检查与 T1-T9 局部时刻检查；
- reason codes 和三项可追溯证据。

任一逐帧检查为 fail 会机械覆盖总判断。调用固定 `data_collection=deny`、`allow_fallbacks=false`，按 `sample_id` 即时落盘并支持断点续跑；API key 只从环境变量或权限为 600 的密钥文件读取。模型原始结论不会直接删数据，必须经过下述人工校准门槛。

## 当前 Qwen3 校准流程

历史 233 条人工 pair 标签重新应用当前 metadata/结构规则后剩 123 条当前范围样本。旧 36 条提示词开发集去掉 4 条当前已排除项后，固定为 32 条 dev；其余 91 条最初固定为未触碰 holdout。2026-08-31 按人工缺陷标签从中冻结并运行了 38 条定向伪影实验，剩余 53 条仍未调用 Qwen3。对应文件为：

- `artifacts/qwen3_openrouter_v1/calibration_dev.csv`
- `artifacts/qwen3_openrouter_v1/calibration_holdout.csv`
- `artifacts/qwen3_openrouter_v1/calibration_split_summary.json`
- `artifacts/qwen3_openrouter_v1/artifact_focus_eval_38.csv`
- `artifacts/qwen3_openrouter_v1/artifact_focus_eval_v2_5_evaluation.json`

dev 失败时不运行代表性 holdout 和 4,003 对全量。上述 38 条只用于回答局部伪影能否被抓到，不具有解锁正式硬判定的效力。切分、定向集构建与评估脚本为 `build_qwen3_calibration_split.py`、`build_qwen3_artifact_focus_eval.py`、`evaluate_qwen3_artifact_focus.py`、`evaluate_qwen3_calibration.py` 和 `apply_qwen3_gates.py`。

## 历史本地 Qwen2.5-VL-7B 流程

运行环境需按 [`requirements.txt`](./requirements.txt) 保持 NumPy 1.23.5 兼容性，并预缓存 Qwen2.5-VL-7B、SigLIP base、DINOv2-small、MUSIQ 和 `yolov8s-worldv2.pt`。当前服务器已完成这些一次性准备。

准备工作已经拆成可恢复脚本：

```bash
cd /scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
python 1_specialist_filtering/scripts/build_stage1_continuation.py
python 2_vlm_filtering/scripts/build_historical_pair_labels.py
python 2_vlm_filtering/scripts/build_calibration_pool.py
```

这会生成 240 条从未人工审核过的模型校准池：120 条 robot/CG、120 条普通 pair；另导出已有的 233 条历史 pair 标签用于提示词开发验证。GPU 作业先各跑 10 条 smoke，通过后再完成模型校准池：

```bash
mkdir -p 2_vlm_filtering/logs
sbatch 2_vlm_filtering/scripts/run_calibration.sbatch
```

模型分数完成后，脚本从 robot 高分/边界/低分各取 20 条，并从普通 pair 的 Qwen accepted/rejected/uncertain 分层取 40 条，生成 100 对新审核清单。审核页面沿用紧凑格式，只展示 SRC、TARGET、instruction 并填写一次整体判断。

```bash
bash 2_vlm_filtering/scripts/build_and_upload_review.sh
```

审核包上传到：

```text
/apps/bypy/lance_review/general_edit_stage1_2_stage2_calibration_100_v1_bundle.tar
```

当前审核包已完成并上传：大小 `117,094,400` bytes，SHA256 为 `d79248343eee6a9b5a385cf19c69a424acea19524145e533c5c6f9e8b384e390`。

将浏览器导出的结果保存为：

```text
2_vlm_filtering/artifacts/vlm_run_v1/general_edit_stage1_2_stage2_calibration_100_v1_review.csv
```

## 硬判定门槛

- Qwen 自动 accepted 的人工可用率至少 95%。
- Qwen 自动 rejected 的准确率至少 90%。
- 单个 operation 的 accepted 可用率低于 90%，该 operation 不开放自动接受。
- 只允许使用未参与提示词修改的 holdout 开启硬判定；dev 结果不能直接解锁全量。
- Stage 1.2 同样要求 accepted 95%、rejected 90%；证据不足时 robot/CG 留在 uncertain。
- mock 输出会被正式决策脚本明确拒绝。

校准门启用后才允许提交全量。Qwen3 程序在检测到默认 4,003 输入时会读取 `qwen3_calibration_gates.json`；状态不是 `enabled` 会在任何 API 调用前退出。`--allow-uncalibrated-full` 只供显式诊断，不得产生正式决策。以下 sbatch 是历史本地 7B 命令：

```bash
sbatch 2_vlm_filtering/scripts/run_full_filter.sbatch
```

全量结果写入 `artifacts/vlm_run_v1/final/`，包含全部决策、accepted/uncertain/rejected、主训练池、辅助训练池和汇总 JSON。所有模型输出支持按 `sample_id` 断点续跑，原始模型响应和触发理由都会保留。

## 文件边界

- `artifacts/vlm_run_v1/`：本级模型分数、校准审核和最终 manifest。
- `artifacts/qwen3_openrouter_v1/`：Qwen3 OpenRouter 程序输出、dev/holdout 切分和校准证据。
- `1_specialist_filtering/artifacts/filter_run_v2/`：结构排除、4,003 对入口及 Stage 1.2 队列。
- `filter_run_v1`、旧人工审核 CSV 与旧 Stage 1.6 队列均为只读历史证据，不会覆盖。

## 当前 Qwen3 校准结果（2026-08-31）

Qwen3 真实调用已完成基础 36 条、多轮定向诊断和 38 条伪影定向 holdout，累计 API 费用 `$0.118950104`，所有正式响应均来自 Alibaba provider，未出现解析错误或重试。32 条当前范围 dev 的逐帧 v2.1 结果为：

- 18 usable / 6 unusable / 8 human uncertain；
- accepted precision `73.68%`，低于 `95%` 门槛；
- rejected precision `20.00%`，低于 `90%` 门槛；
- usable 误拒 `22.22%`；unusable 漏放 `83.33%`。

后续定向版本将局部采样扩为 9 时刻，并强制输出 T1-T9 检查，仍无法稳定识别以下细粒度问题：中段尚未完成 replacement、旧物残留、非编辑小物体消失，以及相邻部件的精确区分。例如模型在清晰的 T4 对照中把 TARGET 仍存在的手机误认成已替换的书，也多次把移除麦克风后保留的黑色防喷罩误认成银色麦克风。

伪影定向实验固定为 38 条：GOKU 15、ReCo 19、EgoEdit 4。其中 7 条带人工明确的 `local_blur / smear_ghost / repair_hole / local_flicker` 标签，20 条为无缺陷、无备注、pair 明确通过的干净对照，另含 12 条人工 hard failure 和 2 条人工仍接受的轻微残影边界样本。v2.5 结果为：

- 明确伪影只有 `1/7` 被 `visible_artifact=fail` 直接识别，伪影漏报率 `85.71%`；若只看最终是否被任一理由淘汰，也只抓到 `2/7`，漏放率 `71.43%`。
- `smear_ghost` 为 `0/2`，`local_blur` 为 `1/6`，`local_flicker` 为 `1/2`，`repair_hole` 为 `1/1`。同一样本可计入多个标签。
- 20 条严格干净对照中，伪影误报 `1/20 = 5%`，任意理由误拒 `2/20 = 10%`；误报不高，但无法抵消严重漏报。
- 10 条非 EgoEdit hard failure 只淘汰 `4/10`。两条人工容忍的轻微残影样本均被接受。
- 未触碰的当前范围样本没有显式 `texture_break` 或 `mask_seam` 标签；本实验只能用修补空洞、涂抹、局部软化和残影作为近似，不能声称已充分验证细纹理断裂或矩形遮罩接缝。

结论：定向推广门为 `not_enabled`，`qwen3_calibration_gates.json` 继续保持 `pending_or_failed`。不运行剩余 53 条代表性 holdout，不启动 4,003 对全量，不生成正式 Qwen 决策或训练 manifest。按本次 v2.5 实测单条费用线性估算，全量约 `$5.6`；在伪影漏报仍然过高时支付这笔费用没有收益。

## Qwen2.5-VL-72B 高风险伪影复核（2026-08-31）

为检查更大 VLM 对残留、遮罩接缝、纹理断裂和时序伪影的识别能力，另从未进入既往人工结果或 VLM 分数的候选中冻结 30 条高风险富集集。它不是自然分布评测集，不能用 rejected 比例估计全量淘汰率：

- GOKU 23、ReCo 7；remove 16、反向 add 14；生成 target/source 各 16/14。
- 7 条为 metadata 已结构排除的 GOKU 服装样本；其余覆盖 GOKU remove 服饰/配饰、反向 GOKU add 的生成 source，以及 ReCo 服饰/人物 remove。
- 12 条叠加 Stage 1.1 边界质量信号。选样脚本为 `build_qwen72b_artifact_risk_30.py`，冻结清单为 `artifacts/qwen72b_artifact_risk_v1/artifact_risk_30.csv`。

OpenRouter 使用 `qwen/qwen2.5-vl-72b-instruct`，provider 为 Nebius，固定 `data_collection=deny`、禁用 fallback。30/30 完成且无解析错误：20 rejected、10 accepted；三项中 instruction following 为 20 fail，visible artifact 为 7 fail，content preservation 全部 pass。正式 30 条费用 `$0.081998`，另有一条 smoke `$0.002775`。这些结果必须等待人工复核，当前不改变任何正式筛选门。

单页离线审核包包含 SRC、TARGET、生成侧、72B 总结果、三项子结果、reason codes 和逐项证据，并保留人工可用/不可用/存疑及 CSV 导出：

```text
/apps/bypy/lance_review/qwen72b_artifact_risk_30_review_bundle.tar
```

远端大小 `29,870,080` bytes，本地 SHA256 为 `3a359385adecb85760ae779a09a33410f4c9c3ba829cb7f28126de89877cba96`。解压后打开 `review_bundle/index.html`。

同一 30 条随后用完全相同的提示词、5 个全图时刻、9 个 ROI 时刻、图像分辨率和 JSON schema 调用 `qwen/qwen3-vl-32b-instruct`，只改变模型。32B 费用 `$0.037733176`，结果为 24 accepted、4 uncertain、2 rejected；visible artifact 为 25 pass、4 uncertain、1 fail。与 72B 的对比为：

- 总决策一致 `11/30 = 36.67%`；19 条分歧中，15 条是 32B accepted / 72B rejected，3 条是 32B uncertain / 72B rejected，1 条是 32B uncertain / 72B accepted。
- visible artifact 一致 `20/30 = 66.67%`；6 条为 32B pass / 72B fail，4 条为 32B uncertain / 72B pass。
- 32B/72B 的总费用约为 `$0.0377 / $0.0820`。模型大小与严格程度高度相关，但在人工结果回来前不能据此断言 72B 更准确。

双模型单页审核包将 19 条总决策分歧置于最前，并并排显示双方三项结果和证据：

```text
/apps/bypy/lance_review/qwen32b_vs_72b_artifact_risk_30_review_bundle.tar
```

远端大小 `29,900,800` bytes，SHA256 为 `f50cd6510ca2858e27340da2e52df693709dd5d9a53003b8f014950b64079a4f`。解压后打开 `comparison_review_bundle/index.html`。

### 72B provenance 路由修正版（2026-09-01）

上一轮人工结果为 14 unusable、9 usable、7 uncertain。旧提示词把训练方向与生成质量方向混在一起，尤其会把生成侧位于 source 的 GOKU 反向 add 按错误方向解释。v3 做了以下修改：

- 只从 `source_provenance/target_provenance` 推导实际生成侧，忽略 `quality_checked_roles`；
- 单侧生成固定重排为 `REFERENCE_VIDEO → GENERATED_VIDEO`；
- `TRAIN_INSTRUCTION` 原文不变，只结构化映射 add↔remove，replacement 使用 `inverse_replace`；
- `edited→edited` 保持训练顺序并检查两侧，unknown provenance 不进入调用。

同一冻结 30 条使用 Qwen2.5-VL-72B 完整重跑，30/30 成功、0 解析错误、0 重试，Nebius provider，总费用 `$0.0870285`。新结果为 20 accepted / 10 rejected，三项结果中 instruction following 为 20 pass / 10 fail，但 visible artifact 与 content preservation 均为 30 pass。与人工明确的 23 条对照：

- 总准确率从旧版 `69.57%` 上升到 `78.26%`；
- usable 误拒率从 `55.56%` 降为 `0%`；source-generated 子集的路由型误拒被明显修正；
- unusable 召回从 `85.71%` 降为 `64.29%`，14 条坏样本抓到 9 条、漏放 5 条；
- 新版 10 条拒绝全部来自 instruction following，`visible_artifact=fail` 为 `0/30`。因此该版本解决了方向误判，但没有证明能可靠捕捉遮罩、涂抹、残影或纹理断裂，仍不能开启全量硬筛。

审核包将 5 条人工 unusable / 新版 accepted 的漏放样本置于最前，随后是抓到的坏样本、正确接受样本和人工 uncertain；每条同时展示训练 source/target、provenance 路由、旧/新 72B 三项证据与上一轮人工备注：

```text
/apps/bypy/lance_review/qwen72b_audit_normalized_v3_review_bundle_v2.tar
```

远端大小 `29,962,240` bytes，SHA256 为 `4875cdc4b057d24a4b775e39fde35568a47debecb03013220d61fe1da6d439b0`。解压后打开 `audit_v3_review_bundle/index.html`。v2 修复了旧/新模型字段名不一致导致的对比区域渲染中断，并在页面顶部突出显示“旧版 → 新版 | 人工”结论。同包附带合并 CSV、评估 JSON 和 API summary。旧 v1 包仅保留为历史，不再使用。

为审查输入分辨率和ROI组织，另导出了5个代表性样本的模型视觉输入实例。每例包含实际发送的 F1-F5、R1-R2，及仅用于诊断的9张拆分ROI和红框覆盖图：

```text
/apps/bypy/lance_review/qwen72b_visual_input_examples_v1.tar
```

大小 `9,400,320` bytes，SHA256 为 `b1c63b53ddad69700448e6fc6045bf700bc1ce7eeaff708d5eb80224806f1f65`；解压后打开 `qwen72b_visual_input_examples_v1/index.html`。实例显示当前ROI平均可覆盖约23%到95%的原图，部分ROI几乎等同全图或没有围住instruction指定对象，因此当前细粒度瓶颈主要是局部定位与版式，而非JPEG长边640本身。

### 72B 五时间卡紧缩 ROI 实验 v4（2026-09-01）

按输入版式复核结论，v4 不再发送 5 张独立全图和 2 张 T1-T9 ROI 长图。它固定发送 5 张时间卡，分别覆盖约 0%、25%、50%、75%、100%；每张卡上半部分为 FULL pair，下半部分为同一时刻、同一坐标的 TIGHT_ROI pair。ROI 由差异定位后以 1.35 倍扩展，面积上限为原画面 30%；极小编辑无法可靠定位时使用 30% 中央裁剪，并提示模型 ROI 不准时回到 FULL 判断。输出 schema 只保留 F1-F5，不再要求 T1-T9。提示词为 `judge_prompt_timecard_v4.txt`。

同一冻结 30 条使用 Qwen2.5-VL-72B 重跑，30/30 成功、0 解析错误、0 重试，provider 为 Nebius：

- 输入总计 228,206 tokens，平均 7,606.87 tokens/对；费用 `$0.064649`，比 v3 的 `$0.0870285` 低约 25.71%。
- 14 accepted / 16 rejected；所有 16 条 rejected 同时被报为 `instruction_following=fail` 和 `visible_artifact=fail`，content preservation 为 30 pass。
- 与上一轮 23 条明确人工标签对照：unusable 召回仍为 `9/14 = 64.29%`，没有超过 v3；usable 误拒由 `0/9` 上升为 `4/9 = 44.44%`，总准确率由 `78.26%` 降为 `60.87%`。
- v4 虽然开始显式报告遮罩、涂抹、残影和纹理断裂，但当前自动对照显示提示词/紧缩 ROI 可能过严，不能据此开启 4,003 对全量筛选。4 条可能误杀和 5 条漏放需优先人工复核；7 条旧人工 uncertain 也保留在包中。

对照审核包并排展示 v3 路由版、v4 五时间卡版、上一轮人工标签、三项证据及 source/target 视频，且把 5 条漏放和 4 条可能误杀排在最前：

```text
/apps/bypy/lance_review/qwen72b_timecard_v4_review_bundle.tar
```

远端大小 `30,085,120` bytes，本地 SHA256 为 `7391574736a5f1aee38fef15010997bf78d45417d101df77c8ef696eb61cf964`。解压后打开 `timecard_v4_review_bundle/index.html`。包内 `results/` 同时保留 v4 原始分数、API summary、自动评估、合并审核清单和实际提示词。

### 72B 整体判断提示词 v5（2026-09-01）

根据 v4 误杀复核意见，v5 提示词保存为 `judge_prompt_timecard_v5.txt`；v4 文件和结果保持不变以保证可复现。v5 的改动为：

- 不再使用 0% 条件帧；五个时间点调整为约 12%、25%、50%、75%、100%。
- 五张时间卡只作为整段视频的联合证据，不再要求模型逐张输出 pass/fail；JSON schema 删除 `frame_audit`。
- 单张清楚、严重的问题仍可直接导致拒绝；单张轻微差异而其余时刻自然稳定时允许接受，不采用机械多数投票。
- `visible_artifact` 聚焦实际高频问题：remove 后不合理补全、大片模糊/平滑色块/遮罩轮廓、明显残影或纹理断裂，以及明显粗劣的新增主体。

同一冻结 30 条使用 Qwen2.5-VL-72B 运行，初次调用有 3 条 OpenRouter 临时未返回 choices，定向重试后 30/30 成功、最终 0 解析错误；provider 为 Nebius，总费用 `$0.0606595`。结果为 24 accepted / 6 rejected，content preservation 为 30 pass。与上一轮 23 条明确人工标签自动对照：

- 9 条 usable 全部接受，误拒率从 v4 的 `4/9 = 44.44%` 降为 `0/9`；
- 14 条 unusable 只拒绝 `5/14 = 35.71%`，低于 v4 的 `9/14 = 64.29%`；
- v5 与 v4 在 12/30 条上改变总决策；v5 明显降低了误杀，但可能过度放宽，需要人工重点复核 9 条潜在漏放。

对照审核包并排展示 v4 逐帧版、v5 整体判断版、上一轮人工标签、证据和 source/target 视频：

```text
/apps/bypy/lance_review/qwen72b_holistic_v5_review_bundle.tar
```

远端大小 `30,054,400` bytes，本地 SHA256 为 `2302b49fd8f52a30bc7ec48b21b696652afbd6f53e3bfda798315940ac86f5fe`。解压后打开 `holistic_v5_review_bundle/index.html`。前 9 条是潜在漏放，包内 `results/` 保留原始分数、API summary、评估、审核清单和最终提示词。

### remove 补全召回优化 v6-v8（2026-09-01）

针对 v5 漏掉的大片模糊、人体未补全和头发未补全，v6 将提示词压缩为 4.7 KB，并要求每条 remove 说明目标原区域实际补出了什么。同一 30 条费用 `$0.0605575`，结果仍为 24 accepted / 6 rejected：`pilot-goku_add-b6accfce9a157a30` 的人体缺失可以抓到，但 `pilot-goku_add-5a25b43e48589a91` 的大片模糊及三个帽子/头发样本仍被描述为“自然内容”。

v7 进一步强制输出结构化 removal audit，只在 13 条定向开发集上运行，费用 `$0.02613725`。它仍把模糊团描述成自然背景、把缺失身体描述成自然皮肤、把帽形平滑区域描述成自然头发，甚至放过 v6 已抓到的人体缺失。因此 v7 判定为失败实验，不运行完整 30 条，也不进入候选审核包。

提示词调整已经到达视觉识别瓶颈。当前最好候选 v8 使用较稳的 v6 72B 结果，再叠加两个便宜、可审计的 specialist 信号：

- `cpu_large_local_detail_drop`：五个 ROI 的 generated-reference 中位 entropy delta 不高于 `-0.65`，且 gradient ratio 不高于 `0.85`；用于发现 remove 后大片局部模糊或平滑填充。
- `goku_headwear_remove_reconstruction_risk`：GOKU accessory 的帽子/头饰 remove 直接进入拒绝候选；当前抽样中模型无法可靠区分自然头发与帽形平滑/异常秃顶。

v8 不增加 API 调用。在 23 条明确旧人工标签上，自动对照结果为：

- unusable 召回 `10/14 = 71.43%`，高于 v4 的 `9/14` 和 v5/v6 的 `5/14`；
- usable 误拒 `0/9`；总准确率 `19/23 = 82.61%`；
- 30 条总体为 15 accepted / 15 rejected；新增抓到 `pilot-goku_add-5a25b43e48589a91`、ReCo 大幅细节下降样本，以及三个帽子/头发风险样本；`pilot-goku_add-b6accfce9a157a30` 由 v6 保持拒绝。

审核包先展示仍漏掉的 4 条 unusable，再展示抓到的 10 条；专项新增抓到的样本位于该组前部：

```text
/apps/bypy/lance_review/qwen72b_hybrid_v8_review_bundle.tar
```

远端大小 `30,197,760` bytes，本地 SHA256 为 `91ed0f5692eadd7beb178bb1bcff22f3d66c507523581912888e4f67912d8c77`。解压后打开 `hybrid_v8_review_bundle/index.html`。页面右侧 evidence 显示 CPU 数值和触发原因，包内 `results/` 保留 v6 提示词、v6/v8 分数、评估、审核清单和 specialist 程序。

这些 specialist 阈值是在同一 30 条开发样本上选择的，不能视为独立泛化证据。程序默认入口已回到 v6；全量校准门继续锁定。只有本次人工复核通过且在新的未触碰 holdout 上复现后，才能运行 4,003 对全量。

### 最佳分支路由 v12（2026-09-01）

复核 v8 前四条漏放后发现，其中两个 GOKU 服装样本本来已被上游 `excluded_garment_add_remove` 淘汰；真正会进入 Stage 2 的新增缺口是护目镜/太阳镜 remove 后眼睛被肤色色块抹掉。另一个关键问题是旧请求同时呈现 `AUDIT_OPERATION=remove` 与原始 `TRAIN_INSTRUCTION=Add ...`，72B 的回答确实多次跟随后者而忽略实际审查方向。

v9 保持媒体和 provenance 路由不变，但只给模型一个当前审查动作：原始 add/remove 指令只去掉开头命令词并作为“待移除目标描述”，不再作为第二条冲突命令。严格提示词保存于 `judge_prompt_timecard_v9.txt`。在 4 条明显漏放与 4 条可用对照上，v9 抓到 4/4 坏例，但会误杀不应进入严格分支的项链、耳环和鼻环；眼镜三例的结果则在独立重复调用中保持一致：2 条人体/眼睛未补全坏例均拒绝，1 条可用眼镜例接受。

因此 v12 不用一条提示词机械覆盖所有目标，而采用以下固定路由：

- 已有 Stage 1 结构规则先执行；GOKU 服装 add/remove 不进入 Stage 2。
- 普通样本沿用 v8（v6 72B + CPU 大片细节下降 + GOKU 帽子风险）。
- instruction 命中 glasses/goggles/sunglasses/spectacles/eyewear（及中文同义词）的 remove 审查，额外走 v9 人脸结构补全检查；v9 只覆盖这一窄分支。

组合程序为 `scripts/compose_best_routed_filter_v12.py`。冻结 30 条上的结果：最终管线拒绝 14/14 人工 unusable；对真正通过上游结构规则、会进入 Stage 2 的 16 条明确标签，8/8 unusable 被拒绝、8/8 usable 被接受。唯一按旧人工质量标签统计的 false reject 是一条画质可用的 GOKU 格子衬衫，但它属于已确定整体排除的服装分支，不是 VLM 质量误判。该结果仍是在同一开发集上调校，正式扩量前必须用未触碰 holdout 复验。

最佳版审核包并列显示上版 v8、新版 v12、两版 evidence、旧人工标签和实际生成侧：

```text
/apps/bypy/lance_review/qwen72b_best_routed_v12_review_bundle.tar
```

远端大小 `30,146,560` bytes，本地 SHA256 为 `3bf26ac9d579a59b529809f56b9cb11e79fe86fe664308ca571f404c59565825`。解压后打开 `best_routed_v12_review_bundle/index.html`。第 1 条单独展示上游服装策略造成的表面 false reject；随后优先展示 v8 漏放而 v12 抓到的样本。包内 `results/` 保留组合分数、评估、v9 两次眼镜结果、提示词和组合程序。

### v12 未见 holdout 40（2026-09-01）

为避免继续在原 30 条上过拟合，使用固定种子 `20260901` 从正式 4,003 对中冻结 40 条新审核集。抽样前排除项目中 741 个曾进入人工审核或 VLM 开发清单的 sample ID，剩余 3,742 对媒体完整候选。六个正式分支各分层随机抽 6 条，共 36 条；再从 8 条尚未审核且“眼镜本身是 remove 目标”的候选中随机抽 4 条。最终为 ReCo/GOKU/EgoEdit `19/15/6`，add/remove/replace `13/15/12`，Stage 1 accepted/uncertain `35/5`。

v8 基础 72B 对 40 条调用费用 `$0.08963125`，4 条眼镜 v9 复核费用 `$0.0069585`，共 `$0.09658975`，全部解析成功。v8 为 30 accepted / 10 rejected；v9 将两条眼镜样本从接受改为拒绝，v12 最终为 28 accepted / 12 rejected。该计数不是准确率，必须等独立人工标签返回后才能计算漏报和误杀。

审核包采用盲审页面，只显示 source、target 和 instruction，不提前显示模型结论；v12 分数保存在包内 `results/`，用于收到人工 CSV 后自动对照：

```text
/apps/bypy/lance_review/qwen72b_v12_unseen_holdout40_review_bundle.tar
```

远端大小 `55,162,880` bytes，本地 SHA256 为 `e8b46f20650d86728c988027028fac98d8bcf2252f4e85b9fe90f1bfdac5ed0a`。解压后打开 `review_bundle/index.html`，完成后导出 `qwen72b_v12_unseen_holdout40_v1_review.csv`。

#### holdout 人工结果与结论

40 条盲审已全部回传：原始标签为 24 usable、2 unusable、14 uncertain。按后续约定，将备注“基本可用”的 2 条 uncertain 计为 keep，将明确备注模糊、低质、没删净、preservation 或 instruction 问题的 8 条计为 delete，剩余无备注的 4 条保持 neutral；最终得到 26 keep、10 delete、4 neutral。

在 36 条明确 keep/delete 上：

- 基础 v6：delete 召回 `3/10 = 30.00%`，keep 误删 `5/26 = 19.23%`，准确率 `24/36 = 66.67%`。
- v8：delete 召回 `4/10 = 40.00%`，keep 误删 `6/26 = 23.08%`，准确率 `24/36 = 66.67%`。
- 最终 v12：delete 召回 `4/10 = 40.00%`，keep 误删 `8/26 = 30.77%`，准确率 `22/36 = 61.11%`。

因此原 30 条开发集上的 v12 提升没有在未见 holdout 复现。严格眼镜 v9 在 4 条上拒绝 2 条人工 keep、接受 2 条 neutral，没有提供正向证据；GOKU 帽子硬拒绝也产生一条确认误删。v12 相比 v6 只多抓到一条应删样本，却多误删三条应保留样本，当前禁止扩到 4,003 对。CPU 大片细节下降信号未在 holdout 产生确认 keep 误删，但也没有抓到“右半边人物透明”这一明确漏放，仍需独立验证。

人工结果、基础 v6、最终 v12 和两版完整 evidence 的对照包：

```text
/apps/bypy/lance_review/qwen72b_v12_holdout40_postreview_comparison.tar
```

远端大小 `55,203,840` bytes，本地 SHA256 为 `14db96fe83646dc74c8da2e4f1bb7fae08e20ce0ed3db7b302cb42fd335ac2ed`。解压后打开 `postreview_comparison_bundle/index.html`；排序先展示 v12 漏放，再展示 v12 误删。

### PAL4VST 局部伪影实验（2026-09-01）

为验证专用像素级伪影模型能否补足 72B 对遮罩、涂抹和局部模糊的漏报，使用 PAL4VST 作者官方 `unified.pth` 对冻结的 holdout40 与高风险 30 条一起实验。70 条含 35 usable、24 unusable、11 uncertain；每对按 provenance 路由实际生成侧，取约 12/25/50/75/100% 五个时刻，同时运行 generated 全图、tight ROI 及同坐标 reference 对照，共产生 1,400 行概率指标和 70 张定位热图。H200 正式推理耗时 98 秒，70/70 成功。

结果不能支持 hard gate：

- 全图与 ROI 的固定 OR 指标最好，但 ROC-AUC 只有 `0.601`；同集最佳平衡点召回 `70.8%` 坏样本，同时误杀 `42.9%` 好样本。
- 达到至少 80% 召回时误杀为 `71.4%`；在更自然的随机 holdout40 上 AUC 仅 `0.535`。
- 模型能明显响应 `pilot-goku_add-5a25b43e48589a91` 的大片人体模糊和 `pilot-goku_add-b6accfce9a157a30` 的身体未补全，但也把正常人脸、运动手部、反光物体、透明保鲜膜与复杂纹理高频标成伪影。
- reference 差值没有解决领域偏移，tight ROI 偶尔还会偏离真正编辑对象。PAL4VST 当前只能作为高阈值人工复核候选生成器；未经项目数据上的像素级适配，不进入删除规则。

可复现实验脚本为 `scripts/run_pal4vst_experiment.py`。完整结论、逐帧/逐样本 CSV、评估 JSON 与可视化 HTML 位于 `artifacts/pal4vst_v1/`，入口为 `REPORT.md` 和 `index.html`。现有 Stage 2 决策与 4,003 对输入均未被修改。

### VEFX-Reward-4B pair 专用模型实验（2026-09-01）

使用作者官方 `xiangbog/VEFX-Reward-4B` 和官方 VEFX-Bench 推理代码，对上述冻结 70 条中仍属于当前 Stage 1 范围的 63 条评分。被排除的 7 条均为后来已经整体移除的 GOKU 服装分支；63 条包含 34 usable、18 unusable、11 uncertain。媒体按 provenance 统一为参考侧到生成侧：43 条 forward、16 条交换媒体并反转操作、4 条 EgoEdit `edited→edited` 保持训练方向并单列。模型按官方默认使用 4 fps 和每帧最多 399,360 pixels，63/63 成功，单卡 H200 纯推理约 80 秒；4B 缓存实际占约 13GB。

VEFX-4B 在当前人工标签上没有可用的删除信号：

- 按模型定义以低分视为低质时，IF/RQ/EE/Overall 的坏样本 ROC-AUC 分别为 `0.319/0.278/0.350/0.281`，均明显低于随机水平。
- Overall 要达到至少 80% unusable 召回，会误删 `91.2%` usable；只保留单侧生成 pair 时仍误删 `90.0%`。
- 人工 unusable 的 Overall 均值反而为 `6.091`，高于 usable 的 `5.622`；这个反向现象在 GOKU 和 ReCo 内也存在，不只是 EgoEdit 两侧生成样本造成的跨数据集偏差。
- Transformers 4.57 提示 image processor 默认切换为 fast。另用 checkpoint 对应的 slow image processor 完整复跑 63 条，IF/RQ/EE/Overall 共 252 个值与原运行逐项完全一致，因此结论不是该兼容性差异造成的。

当前不把 VEFX-4B 用作 hard reject，也不依据本批数据把分数方向反过来使用；后者很可能是在同一开发集上利用编辑幅度或数据集风格的偶然相关。官方 32B 需要约 65GB 显存和更大的磁盘权重，而论文中相对 4B 的总体增益有限，因此先不下载 32B；只有其他强模型比较显示 reward-model 路线确有价值时再补测。

可复现实验入口为 `scripts/prepare_model_bakeoff.py`、`scripts/run_vefx_reward.py`、`scripts/run_vefx_4b.sbatch` 和 `scripts/evaluate_vefx_reward.py`。当前范围清单、两次逐条分数、汇总与评估位于 `artifacts/model_bakeoff_v1/`。本实验没有修改 4,003 对输入或任何现有 Stage 2 决策。

### OpenRouter 强模型公开样本连通性测试（2026-09-01）

在没有外发项目数据授权前，只使用 VEFX-Bench 官方公开的 object-removal demo 做接口测试。最初五时间卡请求发现 DeepInfra 每个 prompt 最多接受 4 张图，因此 API 对照固定为四张时间卡，约对应视频 12%、38%、62%、100%；每张仍同时包含 FULL pair 和同坐标 tight ROI pair。四卡提示词为 `judge_prompt_timecard_v6_fourcard.txt`，不改动已有五卡提示词和历史结果。

六个候选均已在严格 JSON schema 下成功返回：

| 模型 | 固定 provider | API 秒/对 | 单对费用 | 63 对线性估算 |
|---|---|---:|---:|---:|
| Qwen3-VL-235B Instruct | DeepInfra | 8.61 | $0.00142 | $0.09 |
| Qwen3.5-397B-A17B | DeepInfra | 5.93 | $0.00177 | $0.11 |
| Gemini 3.7 Flash | Google AI Studio | 4.78 | $0.00489 | $0.31 |
| GPT-4o-2024-11-20 | OpenAI | 6.03 | $0.01480 | $0.93 |
| GPT-5.6 Sol | OpenAI | 6.88 | $0.02310 | $1.46 |
| Gemini 3.1 Pro Preview | Google AI Studio | 17.59 | $0.02802 | $1.77 |

六模型跑完当前 63 条的单样本线性费用约 `$4.66`，低于当前 key 的剩余额度；实际费用仍会随图像 token 和 reasoning 长度变化。Qwen3.5 必须关闭默认 reasoning，否则 768 个 completion token 会全部耗在思考且没有最终 JSON；Qwen3-VL-235B 的 Venice 端点会返回空白结构化内容，固定 DeepInfra 后正常。Gemini 两个型号使用 mandatory reasoning 的 low 档；GPT-5.6 Sol 使用 low 并省略不支持的 temperature。脚本会保存 provider、reasoning token、费用和原始响应，模型间统一使用同一四卡提示词。

公开 demo 没有本项目的人工 usable/unusable 标签，其 accepted/rejected 结果不能用于比较准确率。正式比较仍需在 63 条人工开发集上计算 unusable recall、usable false reject、coverage 和分支混淆；在用户明确同意外发这些抽帧图之前不会调用。公开清单和汇总由 `scripts/prepare_public_openrouter_smoke.py` 与 `scripts/summarize_public_api_smokes.py` 生成，结果位于 `artifacts/model_bakeoff_v1/`。

正式六模型运行器为 `scripts/run_openrouter_bakeoff_matrix.py`。默认费用上限为 `$6`，逐模型串行、模型内部四并发，保存每个模型的 CSV/summary 后才进入下一个模型，并在每个模型结束后重新核算累计费用；中断后按 sample ID 续跑。项目数据入口必须显式提供 `--confirm-project-frame-export`，否则会在读取 API key 和调用网络之前退出。当前 dry-run 已验证 63 条、六模型和 `$4.6626` 线性估算，尚未触发项目数据调用。

## 历史本地 7B 校准结果

H200 作业 `118695` 已成功完成，总耗时 21 分 37 秒：

- Stage 1.2：120/120 完成，0 错误，约 0.50 秒/对。
- Qwen 新校准池：240/240 完成，0 解析错误；原始结论为 193 accepted / 47 rejected。
- Qwen 实测平均 2.31 秒/对、2,788 input tokens/对，峰值显存 17.3GB；按线性外推，4,018 对纯推理约 2.6 小时。
- 当前范围的 128 条历史标签上，Qwen 原始 accepted/rejected 精度均未达硬门槛；`qwen_thresholds_pre_review.json` 因此为 `pending_or_insufficient`。
- 新 100 对盲审清单按 add/remove/replace 覆盖 32/34/34，并包含 Stage 1.2 高分、边界、低分各 20 条。

历史本地 7B 流程同样停在校准门，其原始结论未写入训练 manifest。
