# Lance/GOKU 退役档案（2026-11-12）

本目录是永久删除 Lance/GOKU 本地实现与媒体后的最小审计档案。只保留关键结论、成功上传回执和删除清单；不保留可复跑源码、模型、输出或审核媒体。百度网盘交付以已有 `uploaded` / `complete` 回执为准，未重新下载或复核远端哈希。

删除前，CoinVE＋TripVVT 正式 11K 的 5,202 个外部媒体引用已用同文件系统硬链接迁入 `dataset_research/coinve_tripvvt_11k_20261108/media/{coinve,tripvvt}/`。迁移逻辑大小 40,731,540,560 字节；源和目标均按清单 SHA256 校验，11,000 条样本和 24,200 个媒体字段通过全量验证。详细回执见该数据集的 `metadata/media_internalization_20261112.json`。

## 永久删除范围与删除前大小

| 路径 | 字节 |
| --- | ---: |
| `lance_general_edit_finetune/` | 230,169,206,784 |
| `dataset_research/goku_research/` | 125,898,752 |
| `model_evaluation/ivebench_lance/` | 31,380,406,272 |
| `model_evaluation/lance_base/` | 1,955,086,336 |
| `model_evaluation/lance_remove_ablation/` | 58,187,776 |
| `model_evaluation/lance_tools/` | 73,728 |
| `model_evaluation/gtfresh1150_vs_coinve677_20260927/` | 888,061,952 |
| `model_evaluation/history_vit_ablation_progressive_v1/` | 94,208 |
| `model_evaluation/history_vit_ablation_v1/` | 110,592 |
| `model_evaluation/ivebench_history_vit_ablation_nonrollback_nocamera_v1/` | 130,355,200 |
| `model_evaluation/real_person_base_vs_gtfresh1150_20260929/` | 155,234,304 |
| `model_evaluation/real_person_drift_stress_v1/` | 5,468,160 |
| `model_evaluation/real_person_multiturn_benchmark_v1/` | 58,187,776 |
| `model_evaluation/real_person_multiturn_benchmark_v2/` | 53,579,776 |
| `model_evaluation/omni_baselines_20260926/lance/` | 306,835,456 |

合计删除前逻辑占用 265,345,846,272 字节。由于正式 11K 新建的是硬链接，其中被引用的数据块会继续由 11K `media/` 持有。

## 保护范围

- `model_evaluation/ltx25_action_full121_20261112/` 整体。
- 所有 LTX、VACE、A849 runtime、模型权重、缓存、条件和运行元数据。
- `model_evaluation/ltx25_vace_a100_20261111/{runtime,cache,conditions}`。
- CoinVE＋TripVVT 正式 11K 的媒体、清单、reader 和元数据。

`records/` 保存从待删目录抽取的 README、状态摘要、上传回执及 SHA sidecar。它们仅用于结论和交付追溯，不构成可复跑环境。

## 验收与空间记录

- 删除前共享文件系统：已用 2,821,370,494,976 字节，可用 178,629,505,024 字节，使用率 95%。
- Lance/GOKU 删除后：已用 2,595,652,624,384 字节，可用 404,347,375,616 字节，使用率 87%；15 个目标路径全部不存在。
- 审核包、展开副本、抽查媒体、字节码和会话转录清理后：已用 2,572,392,165,376 字节，可用 427,607,834,624 字节，使用率 86%。相对开始净释放 248,978,329,600 字节。
- 11K 最终验收：`manifest.jsonl`、`manifest.csv`、`crops.csv` 各 11,000 条；split 为 10,000 / 500 / 500；24,200 个媒体字段全部为 `media/...` 且存在；`dataset_reader.iter_samples` 可遍历全部样本。
- 当前 `action_full121` 是 `model_evaluation/` 下唯一保留的本地审核 ZIP。其余历史 LTX/VACE 清单、上传回执和 runtime 保留，已上传审核媒体已删除。

共享 WekaFS 存在其他并发任务。清理结束后观察到新的 `minimax_h3_ref2va90_local_20261112` 等产物持续写入，后续 `du` / `df` 可能回升；这些数据不在本次授权范围内，未做删除。
