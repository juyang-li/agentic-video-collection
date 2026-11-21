# 300 对冻结测评集与跨服务器交接

交付入口为 `package/README.md`、`package/inputs.jsonl`、`package/ground_truth.jsonl`。300 对为 UE 200、CoinVE 75、TripVVT 25；旧 100 对原身份保留，新增仅取师兄当前正式 test。外部 96 条不纳入。数据与文档包不含权重、环境、推理脚本或旧输出。

在 login 节点使用已有 PyAV 环境：

```bash
model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python model_evaluation/frozen300_20261113/scripts/build.py
python model_evaluation/frozen300_20261113/scripts/document.py
python model_evaluation/frozen300_20261113/scripts/select_review.py
model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python model_evaluation/frozen300_20261113/scripts/review.py --compact
python model_evaluation/frozen300_20261113/scripts/verify.py
python model_evaluation/frozen300_20261113/scripts/verify_review.py
python model_evaluation/frozen300_20261113/scripts/compact_package.py
python model_evaluation/frozen300_20261113/scripts/package.py
```

抽样、原路径、完整解码 PTS 缓存和上传回执保留于本地 `metadata/`。构建器遇到身份变化或 SHA256 不符立即失败，不按效果补样。媒体和清单不进入 Git；本目录不修改既有 H3/VACE 作业或自动监控。不向师兄服务器提交任务。

A849 新批次已提交：`201941`（testqueue、4×H200），全部重新推理，输出独立保存在 `outputs/a849/`。输入预检通过：300条文本缓存与475份可复用输入媒体缓存已核验，100份真实视频新窗口缓存由GPU作业重新编码。新批次独立监控入口为 `scripts/watch_a849.py`，15分钟检查，仅testqueue到期/抢占/节点故障续跑；失败保留证据并停止盲目重试。状态见 `metadata/a849_submission.json`，不能将submitted写成推理完成。

审核展示按最新要求精简为30对（8个合成人物各3对＋CoinVE3＋TripVVT3），交接数据仍完整保留300对全部媒体。精简审核ZIP与完整数据ZIP分别上传，不能将30对审核包当作完整测评输入。

上传入口（login节点）：`scripts/upload.sh` 只传30对精简审核ZIP；`scripts/upload_full.sh` 传完整300对交接ZIP。两者仅上传ZIP，完成记录分别是 `metadata/compact_delivery.json` 和 `metadata/delivery.json`，不另做远端回读、大小或哈希核验。

交付已完成，上传器均返回complete：`/apps/bypy/lance_review/frozen300_20261113.zip`（完整300对、765个原媒体、模型文档及30对HTML审核页，8,212,424,765字节）；`/apps/bypy/lance_review/frozen300_people30_review_20261113.zip`（独立30对精简审核包，81,184,707字节）。没有额外远端核验。本地快速审核入口：`review_subset/index.html`；完整交接目录入口：`package/README.md`。

2026-11-14：201941每卡完成第一条后因最终日志输出缺少json导入退出，已核验4条MP4哈希与121帧完整解码。显式导入已修复，旧/新worker仅允许本次纯日志变化兼容，参数/ID/输出哈希改变仍拒绝复用。续跑任务202855（testqueue、4×H200），保留全部编码缓存与4条已完成结果。监控修复了Invalid job id误判为调度器不可用的问题，并重新启动。
