# A849 测评集 v3

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。

本目录执行用户的 v3 更新计划，保留相邻 v2 的清单、288 条输出、旧指标与报告，不覆盖历史结果。

正式主测共96条：Hair color16、Hair16、Background16、TripVVT24、通用文字编辑24，不含动作或camera任务。完整头部参考的4条Identity、三模型共12条复测已生成并直接审核：任务02、03、04在三个模型中均未呈现清晰参考面部身份迁移，第1条VACE有较清晰迁移。因此按预设规则将Identity移到独立诊断，并用hair_color_13–16补足四个主测名额。

正式Identity诊断位于`../a849_external_vbench_v3_identity_fullhead/`。初轮02–04参考存在裁顶，九条输出保留在`../a849_external_vbench_v3_identity/`作为历史构图诊断，不用于正式去留；有效且输入未变的01三条输出原样复用。完整头部复测期间，其余已冻结任务继续生成；这一执行顺序偏差及修正证据见`metadata/identity_protocol_correction.json`。未根据模型效果更换其他任务。

身份参考采用固定版本 CelebA-HQ1024，直接查看原图和分割图，背景 RGB128，保留的前景掩膜内像素（包括自然头发）按位保持，不再涂白头发。Hair 的参考及原输入协议保留。发色模板取自师兄训练清单，字段为 `axis=hair_color`、`mode=attribute`、`references=[]`；英文目标色使用实际指令中的 black、blue、blonde、chestnut brown、pink、red、white。多人发色和换衣任务明确目标人物。

通用文字编辑固定为面部 4（旧 native_03、20、23、24）、配饰 6、添加物体 4、移除物体 4、服装／局部属性 6。十八条官方任务保留 OmniEdit 原指令，六条配饰为明确标记的自建任务，无 GT。添加区域根据 source 空间规划，移除对象采用 source-only SAM2 跟踪，配饰按每帧实际面部位置规划对应编辑区域。

正式六项及顺序固定为 Subject Consistency ↑、Background Consistency ↑、Aesthetic Quality ↑、Imaging Quality ↑、Temporal Flickering ↑、Motion Smoothness ↑。Dynamic Degree 仅保留在 v2 历史结果。Temporal Flickering 调用固定版本官方函数，分别报告全量原始分数及 source 与三个模型均通过官方静态筛选的共同任务；没有共同有效任务时记为不适用。全量原始分数受正常运动影响，上升不能直接解释为质量改善；不重复帧或冻结视频。

已核验复用53个任务、159条输出，余129条需新生成。输入与实际生成协议一致的有效输出按 SHA 和输入契约核验后复用；原生成 sidecar 保持原字节。native_03沿用v2推理前中性称谓修正，VACE实际辅助prompt以conditions及metadata/vace_prompt_corrections.json为准，不恢复清单中旧的woman/Her辅助描述。更换参考、指令或 mask 后重新生成。维持三模型的 v2 参数、共同 4 秒、49 帧、12fps 及横竖画布。有效低质量输出不剔除，缺失评分不填零。v3 属于调整后的测试集，不以 v2/v3 均值差声称模型进步。

CPU 下载、准备、原始闪烁评分、聚合和打包直接在 login 执行；GPU 分割、缓存、推理、质量指标和静态筛选只允许 `testqueue`，GPU 入口检查实际分区。身份诊断、缓存及mask准备已在 testqueue 作业202767完成。该交互作业后续被其他模型占用，正式生成已改投独立 testqueue 作业203237；其他工作未停止。

主要入口：`freeze.py`冻结主测；`input_review.py`、`gate.py`及`audit_reuse.py`审核输入和复用；`full.sbatch`在testqueue生成并执行官方GPU指标/静态筛选；`score.py raw`在login计算原始闪烁；`aggregate.py`生成逐视频/分组报告；`review.py --stage all --pack`打包；`validate.py`验收。完整头部修正由`freeze_fullhead_diagnostic.py`准备，直接审核后由`adopt_identity_decision.py`核验全部12条输出、执行去留规则并确认正式主测清单未改变。

输入审核页及运行中对比页已生成，生成、质量评分和最终打包仍在继续；实时状态见metadata/execution_state.json与metadata/completion_watch.json。静态筛选未完成时状态为pending，配对n为null，不作零共同样本结论。最终交付入口为`review/all/index.html`及同目录ZIP；冻结输入审核见`review/inputs/index.html`，逐视频六项分数见`metadata/per_video_metrics.csv`，编辑效果与分数联表见`metadata/per_task_report.csv`，分组报告见`metadata/report.md`。以`metadata/final_validation.json`确认最终完成状态。

冻结输入审核ZIP已按用户要求上传到百度网盘`/apps/bypy/lance_review/a849_v3_inputs96_20261114.zip`（正式96条及4条独立Identity诊断输入，无模型输出）。上传器返回complete，回执见`metadata/inputs_delivery.json`；未额外远端核对或下载回读。解压后打开包内index.html。
