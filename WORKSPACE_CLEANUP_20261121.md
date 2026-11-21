# 工作空间清理与Git纳管：2026-11-21

范围由用户确认：保守清理，整理后统一提交。实际删除仅发生在video_collection内部；未运行新推理、评分API或GPU任务。

## 实际清理

- 删除2336个文件：27个有成功回执的ZIP、1959个可重建浏览文件、350个Python字节码；逻辑大小 **47.626 GiB**。
- model_evaluation的du占用：570.776 → 523.146 GiB，减少 **47.630 GiB**。逻辑文件大小与du按硬链接去重的口径不同；不以共享盘df变化归因本次释放。
- 删除前后4603个已存在的生产清单文件引用均保持存在；24份最终结果／协议文件SHA256不变。最终汇总与整理前d8cd3bc版本逐字节一致，98个数值和20处最佳分数加粗保留。
- 逐文件路径、大小、mtime、链接数、依据与恢复方式见本地 [清理清单](maintenance/metadata/cleanup_20261121.json)；该运行清单不随源码推送。

### 已上传ZIP

依据是对应上传工具的complete／uploaded回执，未追加远端哈希、大小核对或回读。完整回执路径及远端恢复位置保留在本地清单中。

| 本地原路径 | GiB | 恢复依据 |
|---|---:|---|
| `model_evaluation/a849_external_vbench_v3_116/review/a849_v3_candidate28_preview_20261114.zip` | 0.137 | `model_evaluation/a849_external_vbench_v3_116/metadata/supplement_baidu_delivery.json` |
| `model_evaluation/a849_external_vbench_v3_116/review/a849_v3_inputs116_20261114.zip` | 0.511 | `model_evaluation/a849_external_vbench_v3_116/metadata/inputs_baidu_delivery.json` |
| `model_evaluation/a849_external_vbench_v3_116/review/a849_external_vbench_v3_116_scroll.zip` | 1.250 | `model_evaluation/a849_external_vbench_v3_116/metadata/baidu_delivery.json` |
| `model_evaluation/ltx25_distilled_v7_pilot10/review.zip` | 0.537 | `model_evaluation/ltx25_distilled_v7_pilot10/metadata/baidu_upload.json` |
| `model_evaluation/a849_external_vbench_v2_96/review/a849_external_vbench_v2_96_scroll.zip` | 1.113 | `model_evaluation/a849_external_vbench_v2_96/metadata/upload_results_state.json` |
| `model_evaluation/ds_full_eval_20261115/ds_full_eval_20261115_v4_hair24_review.zip` | 0.818 | `model_evaluation/ds_full_eval_20261115/metadata/hair24_baidu_upload.json` |
| `model_evaluation/ds_full_eval_20261115/ds_full_eval_20261115_review40.zip` | 5.045 | `model_evaluation/ds_full_eval_20261115/metadata/baidu_upload.json` |
| `model_evaluation/frozen300_data_ablation_20261115/report/FINAL_RESULTS_300_116_20261116_best_bold.zip` | 0.001 | `model_evaluation/frozen300_data_ablation_20261115/metadata/final_results_best_bold_baidu_upload_20261116.json` |
| `model_evaluation/frozen300_data_ablation_20261115/report/FINAL_RESULTS_300_116_20261116.zip` | 0.001 | `model_evaluation/frozen300_data_ablation_20261115/metadata/final_results_baidu_upload_20261116.json` |
| `model_evaluation/a849_external_vbench_v1/review/a849_external_vbench_v1_smoke_scroll.zip` | 0.025 | `model_evaluation/a849_external_vbench_v1/review/a849_external_vbench_v1_smoke_scroll.baidu_upload.json` |
| `model_evaluation/a849_external_vbench_v1/review/a849_external_vbench_v1_original24_scroll.zip` | 0.168 | `model_evaluation/a849_external_vbench_v1/review/a849_external_vbench_v1_original24_scroll.baidu_upload.json` |
| `model_evaluation/vlm_judge_pilot_20261115/review.zip` | 0.505 | `model_evaluation/vlm_judge_pilot_20261115/metadata/baidu_upload.json` |
| `model_evaluation/vlm_judge_pilot_20261115/review_v2.zip` | 0.290 | `model_evaluation/vlm_judge_pilot_20261115/metadata/baidu_upload_v2.json` |
| `model_evaluation/a849_external_vbench_v3_104/review/a849_v3_inputs104_20261114.zip` | 0.504 | `model_evaluation/a849_external_vbench_v3_104/metadata/inputs_delivery.json` |
| `model_evaluation/a849_external_vbench_v3_104/review/a849_v3_supplement8_preview_20261114.zip` | 0.032 | `model_evaluation/a849_external_vbench_v3_104/metadata/supplement_delivery.json` |
| `model_evaluation/ltx25_distilled_v7_300/ltx25_distilled_v7_300_review.zip` | 8.975 | `model_evaluation/ltx25_distilled_v7_300/metadata/baidu_upload.json` |
| `model_evaluation/frozen300_vbench_20261115/metadata/portable_docs_v2/video_metrics_guides_examples_20261115_214920.zip` | 0.000 | `model_evaluation/frozen300_vbench_20261115/metadata/portable_docs_v2/baidu_upload.json` |
| `model_evaluation/frozen300_vbench_20261115/metadata/readme_delivery_20261115/ssim_psnr_lpips_vbench_readmes_20261115_211140.zip` | 0.000 | `model_evaluation/frozen300_vbench_20261115/metadata/readme_delivery_20261115/baidu_upload.json` |
| `model_evaluation/a849_external_vbench_v4_116/review/a849_external_vbench_v4_116_scroll.zip` | 1.151 | `model_evaluation/a849_external_vbench_v4_116/metadata/baidu_delivery.json` |
| `model_evaluation/benchmark_full10s_20261112/review/full10s_base_a849_80_scroll_20261113.zip` | 1.639 | `model_evaluation/benchmark_full10s_20261112/review/full10s_base_a849_80_scroll_20261113.baidu_upload.json` |
| `model_evaluation/benchmark_full10s_20261112/review/full10s_completed_241outputs_20261113.zip` | 1.955 | `model_evaluation/benchmark_full10s_20261112/review/full10s_completed_241outputs_20261113.baidu_upload.json` |
| `model_evaluation/benchmark_full10s_20261112/review/full10s_completed302_vace_stopped_scroll_20261114.zip` | 2.221 | `model_evaluation/benchmark_full10s_20261112/review/full10s_completed302_vace_stopped_scroll_20261114.baidu_upload.json` |
| `model_evaluation/frozen300_20261113/frozen300_people30_review_20261113.zip` | 0.076 | `model_evaluation/frozen300_20261113/metadata/compact_baidu_upload.json` |
| `model_evaluation/frozen300_20261113/frozen300_20261113.zip` | 7.648 | `model_evaluation/frozen300_20261113/metadata/baidu_upload.json` |
| `model_evaluation/a849_tripvvt_pilot32_20261115/review/a849_tripvvt_pilot32_20261115_scroll.zip` | 0.454 | `model_evaluation/a849_tripvvt_pilot32_20261115/metadata/baidu_delivery.json` |
| `model_evaluation/a849_external_vbench_v3_96/review/a849_v3_inputs96_20261114.zip` | 0.281 | `model_evaluation/a849_external_vbench_v3_96/metadata/inputs_delivery.json` |
| `model_evaluation/a849_external_vbench_v3_96/review/a849_v3_results_snapshot_20261114_2036.zip` | 1.066 | `model_evaluation/a849_external_vbench_v3_96/metadata/upload_results_snapshot_state.json` |

### 浏览副本

| 已清理目录内的媒体 | GiB | 恢复方式 |
|---|---:|---|
| `model_evaluation/ltx25_distilled_v7_300/review/media/` | 8.972 | 既有上传包，或保留的package.py／review.py；不重跑推理或Judge |
| `model_evaluation/ds_full_eval_20261115/review/media/` | 1.931 | 既有上传包，或保留的package.py／review.py；不重跑推理或Judge |
| `model_evaluation/ds_full_eval_20261115/review_hair24/media/` | 0.317 | 既有上传包，或保留的package.py／review.py；不重跑推理或Judge |

## 保留边界

- 正式11K数据集、冻结300输入与GT、参考素材、各模型正式输出、权重、环境、缓存和条件保持原路径；action_full121目录整体保护。
- VLM实际送评frames、参考卡片、原始响应、请求签名及review/evidence保持；VBench source／H3评分投影、分数、协议和原始时序保持。
- 无可靠成功回执的ZIP、用途不明确的文件、旧失败尝试、PID／锁记录均保留；没有按名称批量删除目录。
- 只清理上表三个浏览媒体目录，保留页面、构建脚本和清单；这些页面在恢复／重建媒体前不是可播放的完整审核包。

## 历史变更与纳管

- 整理开始前已有798个tracked路径删除：414个直接落在退役档案明确范围，其余为旧审核展开副本／比较文本和过时方案。它们不是本次新执行的删除。
- 两份已删除的旧dataset方案从整理前HEAD保存到退役档案，保留旧决策历史；退役代码不恢复为当前入口。
- 原已暂存28个VLM源码／Prompt文件，与经审阅的推理、测评、示例源码、依赖和最小退役审计资料一起纳管。媒体、环境、完整分数、运行清单、日志、ZIP和凭据被忽略规则排除。
- 原源码／固定Prompt保持原字节。23处历史尾随空白警告保留，不为格式化破坏已记录的SHA；旧review_v2字符串存在一处Python SyntaxWarning，语法有效。
- Python AST与Shell语法检查通过；通用示例6项CPU测试通过。GPU示例未新增独立验收，不声称通过。
- 当前README与AGENTS已指向最终300／V4-116及正式VLM；未把历史submitted作业重新启动。

## 核对与同步

新经验文档及当前索引的本地链接已检查；补齐原通用指南缺少的用时报告样式。最终数据表和上传回执均保留。提交按历史纳管、经验文档、清理记录分批组织；使用现有origin/main正常推送，不强推。GitHub保存源码与说明，不包含本地全量运行数据。
