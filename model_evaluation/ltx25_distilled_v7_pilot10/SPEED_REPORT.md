# LTX2.5 distilled＋DataMake V7 单阶段实验

历史运行比较；软件版本、缓存、资源竞争可能不同，非严格同场基准。两边decode_seconds均包含VAE、编码写盘、完整视频解码验证；新报告另列VAE、编码写盘和验证时间。主要比较denoise_seconds + decode_seconds；另列不含验证的辅助口径。首条保留。四卡仅可理想外推四个独立worker，未实测。仅UE，不外推CoinVE/TripVVT。

| 样本 | 状态 | 旧dev阶段(s) | 蒸馏去噪(s) | VAE(s) | 写盘(s) | 蒸馏阶段(s) | 配对加速 |
|---|---|---:|---:|---:|---:|---:|---:|
| identity_01 | complete | 392.764 | 23.914 | 1.974 | 1.003 | 27.296 | 14.389 |
| identity_02 | complete | 394.730 | 22.643 | 1.453 | 1.413 | 26.233 | 15.047 |
| single_piece_01 | complete | 409.514 | 23.625 | 1.526 | 1.036 | 26.694 | 15.341 |
| whole_outfit_01 | complete | 410.307 | 24.190 | 1.499 | 2.355 | 29.748 | 13.793 |
| action_01 | complete | 659.693 | 39.488 | 1.462 | 0.963 | 42.368 | 15.570 |
| action_02 | complete | 651.646 | 39.073 | 1.480 | 1.492 | 42.630 | 15.286 |
| site_01 | complete | 422.178 | 24.988 | 1.469 | 1.721 | 29.321 | 14.399 |
| hair_01 | complete | 416.882 | 23.709 | 1.482 | 1.543 | 27.654 | 15.075 |
| hair_color_01 | complete | 374.517 | 22.246 | 1.499 | 0.936 | 25.092 | 14.926 |
| camera_01 | complete | 428.062 | 24.325 | 1.554 | 1.876 | 28.889 | 14.818 |

完成配对：10/10；10条总体加速比：14.90661780382524。
现有配对汇总：{"paired_count": 10, "historical_stage_sum_seconds": 4560.293605115265, "distilled_stage_sum_seconds": 305.92409794963896, "speedup": 14.90661780382524}
排除首条辅助统计：{"paired_count": 9, "historical_stage_sum_seconds": 4167.529286978766, "distilled_stage_sum_seconds": 278.6284617865458, "speedup": 14.957299266043627}

模型加载时间、逐条总耗时、显存峰值及失败原因见 speed_report.json；软件版本见 runtime.json。

可见差异尚待人工查看；不将技术检查通过等同编辑效果通过。
