# 查看 15 组随机 pair

优先打开 `comparison/*.mp4`：左侧是 source，右侧是 target，画面左上角有标签；同名 `.txt` 保存 instruction。

comparison 使用 H.264/AVC (`avc1`) + `yuv420p` + fast-start，尺寸 1920×540、101 帧。若 VS Code 仍受本机 codec 限制，下载上一级的 `ditto_random_pairs_15_h264.zip` 后用系统播放器查看。

`contact_sheets/*.jpg` 是无需视频解码器的六帧静态对照。未经转码的独立 source/target MP4 已在 comparison 校验完成后删除，以免重复保存。
