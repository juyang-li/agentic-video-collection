# History-ViT ablation v1

> **已废弃（2026-10-16）**：该实验误用了回撤/历史恢复专项 benchmark。该 benchmark 继续保留，但当前两个模型本身无法可靠完成任务，因此生成结果无效，不能用于判断历史 ViT 的作用。生成视频、日志、指标以及本地/远端审核包均已删除；定义与脚本仅留作参考。

当前模型未针对真实相机运动编辑训练，后续有效评测移除 `camera_id`。替代实验使用普通连续编辑链，并为 Base 与 gtfresh 从头生成结果，不复用本实验输出。

在冻结的 `real_person_drift_stress_v1` 24 个重点检查点上，比较 Lance Base 与
gtfresh step1150 是否依赖历史编辑链。每次只生成当前检查点，历史视频固定使用原始
正常推理结果，避免消融结果自回填造成混杂。

正式条件共四个：

- `vit_zero_text_keep`：历史 ViT embedding 置零，保留历史指令；
- `vit_gaussian_text_keep`：历史 ViT embedding 替换为逐块同均值/方差高斯噪声，保留历史指令；
- `vit_zero_text_zero`：历史 ViT embedding 与历史指令 embedding 同时置零；
- `vit_gaussian_text_zero`：历史 ViT embedding 替换为高斯噪声，历史指令 embedding 置零。

最新直接编辑源的 ViT/VAE、当前指令、序列结构和扩散 seed 均保持不变。人工审核为
主结论，不调用 Gemini。

## 历史运行记录（不作为实验结论）

- 曾生成 Base 与 gtfresh 的 240 个 checkpoint 输出和 192 组比较，现均已判定无效并删除本地产物。
- 曾上传的百度网盘 ZIP 与 SHA sidecar 已删除，不应继续审核或引用旧副本。

