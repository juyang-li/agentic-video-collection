# 后续任务队列

2026-11-13用户将主评测扩至96条并要求身份/发型去背景，本待办同步迁至[96条版本队列](../a849_external_vbench_v2_96/TASK_QUEUE.md)。以下24条记录仅保留历史。

## 待执行：官方 LTX2.5 Ingredients IC-LoRA，当前24条

- 用户于2026-11-13确认权重为官方 Ingredients，并要求先加入任务队列，不立即补跑。
- 状态：queued；未下载新权重、未提交推理作业。当前三模型72条评测继续完成。
- 范围：沿用当前冻结24条source、编辑语义和实际参考图，不换题；身份照片保持完整原尺寸，只将头发区域填白，不裁剪。
- 冻结清单SHA256：`7586dbed5d17830d0d42fe77a973fc5b4850787e35e03dcf7178f513406a829a`。
- 权重：`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`；执行前固定revision、文件SHA和官方工作流。
- 官方Ingredients是静态参考拼图条件生成，不能假设与现有V7 source视频编辑接口相同。执行前准备并直接检查source抽帧及已有参考构成的拼图和目标描述；不使用GT，不将A849输出作为输入。
- 使用官方适配的推理流程；将实际分辨率、帧数、采样及共同4秒评分窗口写入独立协议。结果单列，不能冒充只换LoRA的配对消融。
- 完成后保留24条生成记录、相同六项VBench逐视频及分组指标、直接视觉观察，并加入无评分控件的离线滚动审核包。
- 所有计算仅用testqueue；届时复用可用分配或自行提交sbatch。

官方说明：https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
