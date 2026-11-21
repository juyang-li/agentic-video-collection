# A849 v3：104 条、四模型

本轮保留原有 v3 的96条输入与已有结果，增加4条 CelebA-HQ 发型参考任务和4条完整头部 Identity。比较 LTX 基座、A849、原40步 VACE及新增 MiniMax-H3 Ref2VA PDD八步。共同时间范围仍为原视频0–4秒；不通过跨版本均值差声称模型进步。

冻结入口为 `metadata/inputs.jsonl`、`freeze.json`、`generation_gate.json`。`input_origins.json`记录原任务和修改来源，`reuse_outputs.json`记录300条父任务结果的精确输入复用。新增发型任务为 `hair_celeb_01..04`，保留原发型指令及source-only mask，仅替换为四张非美国总统/副总统的CelebA-HQ参考。Identity引用已直接审核的完整头部版本，保留自然头发、RGB128背景以及历史失败诊断。

LTX基座采用DataMake dev V7条件协议，与A849仅LoRA不同：40 steps、seed42、CFG3、STG1 block28、rescale0.7、ref_cfg1；不代表官方Ingredients路径的最佳能力。VACE保留40 steps、CFG5、flow_shift5、BF16 transformer和FP32 VAE，mask内RGB128。

H3严格沿用 `qwen3.8/reports/accelerated_editors_portable_20261114/README.md` 的PAI Ref2VA八步PDD权重及输出头、SGLang固定版本、TP2/Ulysses2、shift12/3、lossless/max reference。原生107帧24fps、1344×768或768×1344；评分直接抽取生成帧0,2,…,96得到49帧12fps，像素无损核验。原生视频含音频保留，浏览副本不用于评分。输入有原生时序对齐末端补齐，评分不冻结或重复生成帧。分辨率和编码差异在报告中注明。

GPU入口只允许testqueue。沿用203237，达到调度器时限后由依赖作业203468续跑；实际信息见 `metadata/execution_state.json`。CPU准备、投影、原始闪烁评分、汇总、审核打包和上传在login执行。中断聊天不主动释放分配；调度器时限仍生效。

主要入口：

- `scripts/gpu_pipeline.sh`：复用、补跑4Hair、H3及五项GPU质量/官方静态筛选，可按精确回执续跑。
- `scripts/h3_derive.py --loop`：login生成共同时间点评分视频与浏览副本。
- `scripts/visual.py contacts/record/sync`：直接审核的证据、SHA绑定记录及父任务观察复用，不调用外部VLM。
- `scripts/watch_completion.py --upload`：login完成六项指标、三个统计范围、四模型离线审核及本地验收；直接视觉审核未完成时保持待审核，全部完成后仅上传ZIP。

正式指标固定为 Subject Consistency、Background Consistency、Aesthetic Quality、Imaging Quality、Temporal Flickering、Motion Smoothness。闪烁分别报告全量原始值和source加四模型共同官方静态子集；没有共同任务为不适用，缺失评分留空。报告范围为原96、新增8和总104。状态以 `metadata/completion_watch.json`、`final_validation.json`和最终上传回执为准，本README不代表整批已经完成。
