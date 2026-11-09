# CoinVE 50 对任务均衡审核

完整解压 ZIP，用 Chrome / Edge 打开 index.html，无需服务器或联网。

- 五个展示组各 10 对，保留所有复合指令和 Stylize；同一对可包含多类任务。
- 这不是全库均匀随机样本，类别占比不能外推全库。任务统计及完整指令见 TASK_REPORT.md。
- 排除旧 CoinVE-4K 的 source/target 共享媒体，固定官方 revision、metadata SHA256 和 seed 20261108。
- source 文件名跨度至少 151 帧，技术检查要求共同可用时长至少 5 秒、尺寸/帧率匹配。两侧完整片段均保留；同步只播放共同时间段。
- 完整分辨率和帧率保留，浏览器副本 H.264/yuv420p/faststart；无音轨。原始媒体保存在包外 originals/，清单保留映射和原始 SHA256。
- 空格同步播放，左右键翻页；1 接受、0 拒绝、2 存疑。支持备注及 CSV/JSON 导入导出。
- 默认 human_decision=1、reviewed=0；默认值不是人工通过。没有进行模型质量判定，也未因编辑效果差而换样。
- sample_manifest.csv 为来源和任务清单；review_manifest.csv 为浏览器媒体与原始 SHA256 映射；media_inventory.json 保存转码和完整解码记录。
- 本批用于决定后续新增 6k 的任务范围和质量门槛，不绑定 Lance 或 LTX2.5 的训练输入格式。
