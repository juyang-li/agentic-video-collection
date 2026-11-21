> 此为初轮构图诊断：任务02–04的原参考有裁顶，未完全满足完整头部要求，不能单独作为正式身份淘汰依据。严格复测见相邻 a849_external_vbench_v3_identity_fullhead；全部原结果保留。

# v3 Identity 优先诊断

四条任务、三个模型共十二条输出，独立于 v2 和 v3 主测清单。参考取自固定版本 CelebA-HQ1024，1024×1024 原图不裁剪或缩放；人物前景和自然头发按位保留，背景 RGB128。来源及分割记录在相邻 v3 主目录 `metadata/reference_processing.json`。

沿用 v2 三模型参数，共同0–4秒、49帧、12fps及原画布。先直接检查 source 时间序列、实际面部 mask 与分割参考，再启动复测。`review/*_face_detail.jpg` 是按 source mask 位置提取的放大诊断视图，不改变模型实际输入。

去留规则：至少三条任务在三个模型中都没有清晰面部身份迁移，才移出主测，将四个名额补为发色；改变肤色、复制发型或发色本身不计为清晰面部身份迁移。缺失或未审核的输出不能计为失败。结果写入 `metadata/identity_review.json` 后，由 v3 `scripts/freeze.py` 核验十二条输出哈希并执行规则。

十二条均已生成并直接审核。第1条VACE有较清晰迁移；第2、3、4条三个模型均无清晰迁移，触发移出主测的规则。最终结论见 `metadata/identity_review.json`；`identity_review_partial.json` 仅为过程记录。所有失败和低质量有效结果均保留，不调用 Gemini 或其他 VLM API。GPU 仅 testqueue；CPU 审核页和联系图在 login 生成。
