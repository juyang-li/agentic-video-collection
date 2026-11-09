# OpenVid T2V 审阅包报告

> 完成日期：2026-09-04  
> 状态：50 条审阅包已冻结、校验并上传百度网盘。2026-09-05 用户完成快速人工审阅，认为整体可用；未回传逐条结构化标签。

## 使用决定

OpenVid 已通过 T2V 候选池的初步人工验收，但首轮 Lance LoRA 不混合原计划的 10% T2V，只用冻结编辑池的 train split。若 VBench 显示开启 LoRA 后基础画质或动态明显退化，下一轮再从 OpenVid 引入约 5%–10% replay。

## 抽样

- 来源：`finetrainers/OpenVid-1k-split`，对应 OpenVid-1M 的 1K 高分切分。
- 固定 seed：`20260904`。
- metadata 门槛：时长 3–10 秒，`temporal_consistency_score >= 0.995`，排除 caption 明确标注的 animation/cartoon/game/news/watermark 等内容。
- 50 条分层配额：human 15、vehicle 10、animal 5、scene 10、object 10。
- 为避免完整下载 6.56GB tar，先通过 HTTP Range 扫描 metadata/header，再只下载最终 50 个冻结视频 offset。

## 校验

- 50 条、50 个唯一 sample ID，分层配额全部满足。
- metadata 时长范围：3.403–9.218 秒；temporal consistency 范围：0.995020–0.999479。
- 50/50 可解码首帧和末帧；50/50 转为浏览器兼容的 H.264、CRF17、yuv420p、faststart、无音频代理。
- 50/50 代理 SHA256 与 manifest 一致；manifest 同时保留 OpenVid source key 和原始 MPEG-4 文件 SHA256。
- 审阅包无 `.part` 文件；HTML 中 50 个媒体路径均存在。

## 产物与上传

- 本地目录：`artifacts/openvid_t2v_review_v1/`
- 本地归档：`artifacts/openvid_t2v_review_v1_bundle.tar`
- 百度网盘：`/apps/bypy/lance_review/openvid_t2v_review_v1_bundle.tar`
- 远端确认大小：`318576640` bytes
- SHA256：`b1c617db69f507417c82e434c4d701d05394e80d572097cac37e22652008035f`
- 百度文件 ID：`21134264566522`

解压后直接用 Chrome/Edge 打开 `openvid_t2v_review_v1/index.html`。完成后导出 CSV 或 JSON；服务器回传目录为：

```text
/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/3_lance_lora_finetuning/artifacts/openvid_t2v_review_v1/review_results/
```
