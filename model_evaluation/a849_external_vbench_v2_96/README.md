# A849 外部编辑评测：96条扩展版

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。

用户于2026-11-13要求从24条扩至96条，并去除身份及发型参考图的背景。本目录是新版本入口；v1保留原始输出、协议和已交付审核包，不覆盖历史证据。

目标配额：TripVVT服装24（上装/下装/dresses各8）、身份16、发型16、无人背景16（室内外各8）、原生人物编辑24。先直接检查source全时段抽帧和实际参考图，再冻结清单、prompt、mask方案并推理。不调用Gemini或其他VLM API，不依据模型结果换题，自建任务无GT。

身份参考：完整原图、不裁剪、不缩放，保留人物前景；头发填RGB255，背景填中性灰RGB128，其余前景像素保持原值。发型参考：单张正面照片，去除背景为RGB128，保留原发型与发色。实际分割结果须直接看图，记录处理掩码、原始尺寸和像素验证。

三模型目标共288条输出。v1中只有任务完整输入记录与新版完全相同的有效输出允许复用；原始生成契约不改写。身份和发型组参考已变化，必须重新生成。统一原4秒、49帧、12fps及原推理协议，六项VBench逐视频、分组和A849−base配对差值；缺失不填0。

这里的ltx_base是同版dev权重在DataMake V7条件协议下不加载A849的对照，不是官方Ingredients工作流，也不代表官方最佳参考编辑能力。96条任务对应83段不同source；不同编辑可共用视频，详见metadata/source_clusters.json，不能当作96段独立视频。原生人物编辑24条含外观、服装、颜色及5条表情任务，不改变身体动作或镜头。

所有计算任务仅testqueue；优先复用job199811，失效后自行sbatch。无需smoke关卡；正式生成仍须完成source、参考及mask的直接检查和训练来源重合检查。审核包为固定导航、无评分控件的离线滚动页，VACE辅助视频仅前两条。

官方Ingredients IC-LoRA按用户要求只加入后续队列，范围同步扩至96条，当前不启动。

完整结果审核ZIP已按用户明确授权、直接CPU上传至百度网盘`/apps/bypy/lance_review/a849_external_vbench_v2_96_scroll.zip`，上传器返回complete，回执见metadata/results_delivery.json；未额外远端核对或下载回读。该包已包含source、参考、prompt、全部288条输出和评分报告。

2026-11-13结果：96条任务、三模型288条输出全部生成成功；六项VBench共1728个逐视频分数齐全，288条输出已直接视觉检查。A849相对V7基座的六项全组均值均上升，但分组存在退化，且身份白发泄漏、发色复制和编辑未完成等问题仍在；dynamic_degree上升只表示动态视频比例增加。完整结果和限制见[报告](metadata/report.md)。

交付入口：[无评分控件滚动审核页](review/all/index.html)、[全部结果审核ZIP](review/a849_external_vbench_v2_96_scroll.zip)、[original审核ZIP](review/a849_external_vbench_v2_96_original96_scroll.zip)、[逐视频指标](metadata/per_video_metrics.csv)、[分组/配对指标](metadata/group_metrics.json)、[逐条视觉记录](metadata/visual_observations.json)、[最终验收记录](metadata/final_validation.json)。ZIP解压后打开根目录index.html。original96网盘上传尚待用户对扩展媒体范围的明确授权，不能将本地打包视为网盘交付完成。
