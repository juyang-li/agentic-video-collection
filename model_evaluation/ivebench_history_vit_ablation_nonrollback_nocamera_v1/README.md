# IVEBench History-ViT ablation（非回撤、无 camera）

本实验用于判断 Lance 多轮编辑中，更早历史视频的 ViT 表征是否影响当前输出。数据来自任务中心构建的 IVEBench real-person multi-turn benchmark v2，不是此前误用的 UE/DataMake validation 链。

## 数据与条件

- 父 benchmark：`../real_person_multiturn_benchmark_v2/benchmark.jsonl`，SHA-256 `1f9edd319806e546eef367d82facc14cdb2240b872ad41135e980cd2fe62998a`。
- 整链排除 rollback/history-reference 任务，并删除 21 个 camera turn；33 个后续 turn 中对已删除运镜状态的引用已逐项清理，审计见 `camera_dependency_rewrites.json`。
- 冻结派生 benchmark：24 条链、65 turns、41 个 R2+ checkpoint；SHA-256 `bdc9122c0c67b4e3789426e2c0d16f283cfca09acc5c46fc57275105d0d3d50f`。
- 模型：Lance Base、gtfresh step1150。参数为 30 steps、shift 3.5、text/vit CFG 1.0、seed 42。
- 条件：normal、历史 ViT zero、历史 ViT matched Gaussian。历史文本、最新 direct-source ViT 和 VAE 条件保持不变；消融输出不向后传播。

## 完成与验收（2026-10-17）

- 两模型重新生成 130 个 normal；四组消融生成 164 个输出。
- 294 个推理视频完整解码通过；164 个 receipt 的 hook、历史文本、direct-source ViT/VAE 审计通过。
- 无 hook fixed-history smoke 在两模型的 R2 与末轮均与 normal 逐帧 RGB 完全一致。
- 人工审核包包含 41 张卡、164 个审核单元和 318 个去重视频；离线 Chromium、CSV/JSON 导入导出、ZIP CRC 和单顶层目录验收通过。
- 不含 GT 或 Gemini。差异指标只描述相对同模型 normal 的变化，语义结论以人工审核为准。

## 交付

- ZIP：`review/ivebench24_nonrollback_nocamera_history_vit_ablation_base_gtfresh_human_review_20261017.zip`
- 大小：64,385,001 bytes
- SHA-256：`6599dac4406403bb547ea12cc0105616c2fabc1b14d4123437f7d5ccdf946ace`
- 百度网盘：`/apps/bypy/lance_review/ivebench24_nonrollback_nocamera_history_vit_ablation_base_gtfresh_human_review_20261017.zip`
- 远端大小和 MD5 已与本地一致；同名 `.sha256.json` 为外置验收与上传收据。
