# 模型推理与测评

更新：2026-11-21。当前正式结果均已完成；旧README中的作业号和submitted文字是历史记录，不表示现在仍在运行。

## 结果与经验入口

- [最终300／116结果](FINAL_RESULTS_300_116.md)：300六模型的GT、VLM、VBench和V4自建116四模型的VLM、VBench；最佳分数已加粗。
- [测评经验](METRIC_EVALUATION_LESSONS.md)：正式口径、对齐、并行、失败处理、实测速度和清理边界。
- 外部通用指南：[SSIM／PSNR／LPIPS](SSIM_PSNR_LPIPS_README.md)、[VBench](VBENCH_README.md)。

## 已完成实验

| 入口 | 内容 |
|---|---|
| [冻结300输入](frozen300_20261113/README.md) | UE200、CoinVE75、TripVVT25，完整输入与GT |
| [蒸馏基座300](ltx25_distilled_v7_300/README.md) | distilled＋V7、8步，无LoRA |
| [四模型GT](frozen300_gt_metrics_20261115/README.md) | LTX、A849、H3 PDD8、VACE LightX2V4；1200输出 |
| [四模型VBench](frozen300_vbench_20261115/README.md) | 7200项，全量原始Temporal Flickering，不做静态筛选 |
| [纯合成／纯真实消融](frozen300_data_ablation_20261115/README.md) | 两组各300，5400项新增GT／VBench分数 |
| [最终V4自建116](a849_external_vbench_v4_116/README.md) | 464输出、2784项VBench；非旧V3候选集 |
| [正式VLM](ds_full_eval_20261115/README.md) | DeepSeek最终Prompt，300六模型＋116四模型，4748项含重复评分 |

运行清单、原始响应、逐视频结果和验收记录保留本地，由各实验README定位；GitHub保存源码、说明、固定Prompt和依赖，不提供完整媒体仓库。历史交付ZIP按上传回执从百度网盘恢复；重建浏览包使用保留脚本，不重跑评分API。

## 复用与历史

- [模型入口](inference_models/README.md)、[LTX/VACE百样本阶段](ltx25_vace_a100_20261111/README.md)的runtime、缓存和条件继续保留。
- [完整121帧动作复测](ltx25_action_full121_20261112/README.md)整体保护。
- V1／V2／V3、早期VLM试评、先导10条和旧100条为历史实验；保留原始记录，不作为当前待执行队列。
- [Lance/GOKU退役档案](../retirement_lance_goku_20261112/README.md)只用于结论与交付追溯，不恢复执行入口。
