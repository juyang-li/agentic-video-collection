# 冻结10秒资产重跑

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。

沿用原100条样本ID、完整指令与参考身份，不重新抽样。80条UE冻结真实10秒SOURCE、motion和审核GT；20条CoinVE/TripVVT原片不足10秒，保留ID、列入例外并暂缓，不补帧、不换样。GT仅审核。

当前资源：199810上的A849与LTX2.5基座均已完成80/80，现运行四卡VACE；199975运行H3；199814按用户要求不再用于本任务。分配过期后先尝试空闲的199975，不可用则仅在testqueue自行sbatch四卡H200，每30分钟检查；稳定推理按15分钟检查。

模型与协议：

- 基座与A849：121帧、12fps、40步、seed42、CFG3、STG1（block28）、rescale0.7、ref_cfg1，1280×736或736×1280。复用身份、像素时间轴、权重和tensor核对后的完整训练缓存（SOURCE/motion为16 latent帧，图片为1帧）。
- “基座”是LTX2.5未微调权重＋DataMake V7参考条件协议的同协议消融，不是官方开箱即用的多参考编辑流程。参考→SOURCE→target按训练入口拼接；A849使用相同入口并加载step849 LoRA。不能据此代表官方最佳参考编辑能力。
- VACE：官方Wan2.1-VACE-14B，121帧、12fps、40步UniPC、CFG5、flow shift5、seed42，沿用原720p横竖幅。SOURCE-only分割/跟踪，editable区域按官方repaint规则置灰；动作使用完整motion姿态及SOURCE外观参考。头部解析修复及失败尝试保留，不读取GT制作条件。
- H3：本地官方Ref2VA，无Context-IR；TP2＋Ulysses2四卡、16CPU、50次原生更新、seed42、flow12/audio3、lossless、不量化、不用加速LoRA或torch.compile。原生24fps、243帧约10.125秒，审核共同0–10秒。SOURCE/motion由冻结121个时刻无损扩展，保持时间轴，只在10秒后补原生对齐尾端。VisualVAE按官方原生FP16 autocast解码。
- IC-LoRA补充试验：官方Ingredients权重已核验；多视频SOURCE＋静态拼图联合输入实测出现参考照/参考演员混入输出，已停止并标为无效协议。旧结果和环境保留，不计入有效对照。补跑输入方式等待用户选择，详见iclora/README.md。

A849已完成80/80，通过冻结身份、参数、文件哈希、视频容器和逐文件完整解码记录检查。生成墙钟约3.60小时（包含共享资源及中途调整，不当作纯四卡吞吐基准）；单卡PyTorch峰值分配约48.25GiB。详情见metadata/technical_audit_a849.json。

H3已通过8任务组首条的实际请求、编码路由、50步记录和243帧解码检查。分组加权估算四卡约3.61条/小时，80条生成约22.1小时，属阶段性估算。已测动作样本约29.9分钟；首测最高原生峰值约76.8GiB/卡。CPU绑定已从1修正为16；样本不同，不作单变量因果比较。

技术检查不代表人工质量通过。H3抽帧发现运镜/动作保持问题；IC-LoRA失败协议有明确参考泄漏证据，观察和截图均保留。尚未宣称10秒相对4秒已证实改善或截取就是原因。

SOURCE/参考/审核GT冻结映射、短片例外、完整指令、缓存和逐条运行证据分别在assets/、metadata/、logs/、outputs/、h3/、iclora/。不修改师兄训练代码，不自动清理、提交或推送。本轮尚未制作或上传最终审核ZIP；已授权上传目录为/apps/bypy/lance_review/，交付时只上传ZIP，以上传器成功回执为准。

阶段审核包已由上传器返回complete：/apps/bypy/lance_review/full10s_base_a849_80_scroll_20261113.zip。仅基座/A849共160条输出、80条样本；其余模型仍在运行或待决定。仅上传ZIP，无额外远端复核。

2026-11-13续跑记录：白发hair_color_10的121帧SOURCE-only蒙版改用CPU SAM2正负点与可见发色约束；五个遮挡/端点抽帧检查保留于metadata/hair_color_10_repair_overlay.jpg，121帧文件身份回执为metadata/white_hair_mask_repair.json。先前语义欠覆盖与白背景误选的尝试均保留，续跑入口会核验并复用新蒙版。未修改已运行/完成样本条件。

后续授权（2026-11-13）：待H3和VACE均80条完成，再执行ltx_task_loras/对应任务适配器补测；详见该目录README与metadata/authorized_followup.json。分配过期按用户要求sbatch testqueue四卡H200并休眠等候。

2026-11-13 11:17：199810已到期，VACE完成26条，续跑sbatch任务200694（testqueue、account=prj0000000275、4×H200、18小时）已进入PENDING。199975继续H3。监控已补显式项目账号，并按30分钟周期等待分配；不使用199814。

2026-11-13 16:56：199975到期，H3完成55/80；自动提交201588（testqueue、prj0000000275、4×H200、18小时）等待续跑。VACE200694仍排队，完整26条保留。原running记录仅表示到期中断样本，不表示当前GPU进程仍活跃；新分配会按既有续跑入口归档/重跑未完成条目。两项等待分配按30分钟检查。

已完成结果快照上传成功：/apps/bypy/lance_review/full10s_completed_241outputs_20261113.zip；数量{'ltx_base': 80, 'a849': 80, 'vace': 26, 'h3': 55}。仅ZIP，以上传器complete为准，未额外远端复核。

2026-11-14 用户要求停止VACE并上传已完成结果：已取消job200694；VACE完成62/80，处理中记录仅表示中断，不再自动续跑VACE。依赖VACE全80完成的后续任务不触发；frozen300 A849保持运行。停止清单见metadata/inference_stop_vace_20261114.json。

用户停止VACE后的审核包上传成功：/apps/bypy/lance_review/full10s_completed302_vace_stopped_scroll_20261114.zip；数量{'ltx_base': 80, 'a849': 80, 'vace': 62, 'h3': 80}，合计302条输出。无评分滚动审核，含完整prompt、SOURCE、参考、GT和输出，不含VACE遮罩/overlay/prepared input视频。只上传ZIP，以上传器complete为准。
