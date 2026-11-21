# A849 v3：116 条候选，四模型

当前池保留 88 条输入不变的任务，并准备 16 条 Celeb Hair 和 12 条 Identity 候选，待用户选择。不是已选定的最终测试集。原有 v2、v3_96、104 的清单、失败诊断和输出均保留。

|分组|数量|
|---|---:|
|Hair|24（旧8 + Celeb16）|
|Identity|12|
|Hair color|16|
|Background|16|
|TripVVT|24|
|通用文字编辑|24|

旧 Hair 仅保留 1、2、3、4、6、8、9、10；在新 Hair01–16 中与前8条 Celeb Hair 交替排列，另外8条为 Hair17–24。所有审核 ID 按前缀连续编号、按字母排序。旧 ID 只作来源追踪，见 metadata/renumbering.json。

通用文字编辑24条的原视频均来自固定版本OmniEdit-Bench；18条使用官方原指令，6条配饰指令为自建并单独标记。分为面部4、配饰6、添加物体4、移除物体4、服装/局部属性6。面部旧ID为native_03、20、23、24，在新包对应native_01、08、09、10。逐条官方记录、版本与指令核对见metadata/native_provenance_audit.json。

28 张新参考取自固定版本 CelebA-HQ 1024；两组参考人物全局互不重复，依据经官方校验和核实的 CelebA identity 标签核对，不做脸部识别。直接检查原图、RGB128 去背景图、source0–4秒和编辑区域，不使用 VLM API。人物前景保持原像素，Identity 保留自然头发。之前有问题的00017、00011及分割削掉头顶的候选已移出推荐池。参考图仍有少量分割边缘不规则和原图耳饰/手部，详见输入审核记录。

四模型为 ltx_base、a849、vace、h3_pdd8，共464条输出。前3模型保持40步、seed42、49帧12fps和原横竖画布。H3 严格使用师兄 README 的 PDD8 蒸馏合并权重与8个融合输出头、TP2/Ulysses2四H200、shift12/3、lossless/max reference，沿用共同4秒。原生107帧24fps；评分从实际生成帧0,2,…,96提取49帧，不插帧、不重复或冻结输出，完整原生视频另存。

六项正式指标：Subject Consistency、Background Consistency、Aesthetic Quality、Imaging Quality、Temporal Flickering、Motion Smoothness。全量原始闪烁分受正常运动影响；另报官方静态筛选下 source 与四模型共同通过的任务及配对均值。没有共同任务为不适用，未完成不是零。Dynamic Degree 仅保留历史结果。调整测试集后不以 v2/v3 均值差声称模型进步。

输入清单已冻结：metadata/inputs.jsonl；输入审核与像素核验：metadata/input_review.json、reference_pixel_audit.json；H3预检：metadata/h3_preflight.json。完整来源、复用和模型协议均有 SHA 绑定的回执。重复 source 的不同参考编辑不计作独立视频。

GPU仅使用testqueue。203237于2026-11-14 18:20:22开始，2026-11-15 00:20:24达到六小时时限；续跑203468实际于2026-11-15 03:25:49开始，05:44:29完成全部464条推理，06:01:25完成GPU质量评分与官方静态筛选，按用户要求保留allocation至09:25:55达到六小时时限，以上均为北京时间。未主动取消作业。CPU派生、审核、验收、打包和上传均在login执行。当前无剩余GPU工作。

离线审核ZIP只上传百度网盘/apps/bypy/lance_review/。已交付a849_v3_inputs116_20261114.zip（全部冻结输入）和a849_v3_candidate28_preview_20261114.zip（此前48个新增候选输出的阶段预览）。当前全部464个输出及2784项评分已完成，464个结果均已直接检查并记录，最终四模型ZIP在验收后交付。有效低质量输出和历史失败诊断完整保留，116条仍是待用户选择的候选池。

完整版已交付：百度网盘/apps/bypy/lance_review/a849_external_vbench_v3_116_scroll.zip，1342179375字节（约1.25GiB）。解压后打开a849_external_vbench_v3_116/index.html。验收证据见metadata/completion_requirements_audit.json；共116条候选、464个结果、2784项有效分数，尚未替用户选择最终子集。
