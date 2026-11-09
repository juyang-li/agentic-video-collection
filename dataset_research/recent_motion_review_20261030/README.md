# 近期人物动作数据：无评分连续浏览包

交付 `review/recent_motion_datasets_scroll50_20261030.zip`，解压后打开 `index.html`，直接向下滚动。无评分表、无前后翻页按钮；顶部可跳到各组。

- X-Dance：12条独立动作视频，附人物参考图与官方pose。
- RealisDance-Val：24条人物运动，附官方同来源参考图；本轮视频均为1920×1080或1080×1920。
- Wan-Move / MoveBench：11条人物子集，原始832×480，单列补充区。
- Wan-Animate / Wan-Animate-2：3条官方输入示例，明确区别于完整数据集。

共50条、62个视频（含12个pose），逐个通过完整解码；浏览器确认整页50卡可滚动、无评分控件、62视频元数据可读、四组代表片段可播放，零外部请求。原始分辨率、帧率、完整时长与GT含义逐条标注。

筛选时间窗为2025-04-30至2026-10-30；[RESEARCH.md](RESEARCH.md)解释公开日期、项目热度、数据下载量、实际输入协议，以及MotionPair-60K等暂需申请的候选。原始API快照见metadata/research_facts.json。

已上传百度网盘：`/apps/bypy/lance_review/recent_motion_datasets_scroll50_20261030.zip`。远端文件大小和文件ID已确认，校验sidecar同时上传；未进行整包回读。

ZIP大小：836,203,558 bytes（约797.5 MiB）。SHA256：`01d029bc1a779a3bf56bd4406c34aeaa1eee69bdf043f38a673e06420474c248`。最终上传结果见metadata/delivery_receipt.json。按用户要求不进行网盘整包回读。
