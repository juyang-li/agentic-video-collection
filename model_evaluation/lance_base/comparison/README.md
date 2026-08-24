# Lance 左右对比视频

按编号依次打开 15 个 MP4：左侧是 `SOURCE`，右侧是 `LANCE OUTPUT`；同名 TXT 是编辑指令。

- 编码：H.264 / `yuv420p` / fast-start，便于 VS Code 和本地播放器打开。
- 画面：1920×540，每侧固定 960×540，保持原始长宽比并补黑边。
- 时间：源视频按完整相对进度对齐模型输出，不是只截取源视频开头。
- 评分：把人工结果填写到上一级 [`scores.csv`](../scores.csv)。

如果 VS Code 仍不能播放，下载上一级的 `lance_base_comparison_15_h264.zip` 到本地解压查看。
