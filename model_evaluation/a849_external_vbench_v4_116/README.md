# A849 v4 116

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。

本轮已完成并上传：116条任务、464个四模型结果、2784项六维评分及464条直接视觉审核。最终验收通过，580次完整解码、116次H3原生评分帧像素核对通过；静态共同子集2条。完成记录见 `metadata/finalize_state.json`、`metadata/final_validation.json` 和 `metadata/baidu_delivery.json`。

只读进度入口：`python scripts/status.py`。它分别统计32条候选、16条发色／背景扩充、入选8条的另外两模型补测，以及最终评分和上传回执。H3原生和派生浏览／评分文件只计作一次模型推理。

| 分组 | 最终数量 | 变更 |
|---|---:|---|
| Hair | 24 | 保留原有24条 |
| Hair color | 24 | 原16＋新8；师兄原模板；7色各1＋栗色1 |
| Background | 24 | 原16＋新8；4张原参考各新增2条 |
| TripVVT | 24 | 原24移除06/08/09/13/19/21/22/23；其余16＋新选8 |
| Native | 20 | 面部4、配饰6、添加物体4、服装／局部属性6 |

Identity12和object_removal4移出新版主测，旧v3完整116条清单、464视频、报告与诊断保持原样。92条未变任务的368输出和同媒体指标直接复用。最终四模型116×4＝464输出、2784项六维分数。

TripVVT预筛目录为 `../a849_tripvvt_pilot32_20261115`。用户确认使用本地项目test留出集，严格排除A849已知训练谱系重叠；不能将该标签写成官方test，实际官方split逐条保留。固定seed20261115，上装10、下装10、连衣裙12，先看source时间序列与参考/GT再冻结输入，32条均生成A849和H3。GT和官方caption只用于审阅。

新选8条必须满足A849正确对象换衣成功且内容保持良好。H3可以差或失败；优先A849明显更好，其次双方成功且相近，同档按冻结ID。无最终品类硬配额。若32条不足8条，报告缺额，不放宽条件或自动增加第33条。保留全部32条输出、逐条审核和选择原因；这是看过输出后的展示选样，不以v3/v4均值差声称模型进步。

新增发色／背景16条独立冻结在 `../a849_v4_extensions16_20261115`。H3沿用README的Ref2VA PDD8（含8 fused heads），TP2×Ulysses2；基座与A849保持40步，VACE按逐视频生成回执配置。共同source为0–4秒、49帧12fps，seed42；H3保留原生107帧24fps，评分取真实生成帧0,2,…,96，无输出插帧或冻结。仅使用testqueue，接入用户授权的202896，不取消其主allocation。

六项固定指标：Subject Consistency、Background Consistency、Aesthetic Quality、Imaging Quality、Temporal Flickering、Motion Smoothness。全量原始闪烁分数受正常运动影响，不能直接解释为质量改善；静态子集重新取source＋四模型官方StaticFilter共同通过的任务，列ID、数量和均值，无共同样本记不适用。缺失不填零，低质量有效输出保留。

审阅页按字母顺序、组内连号排列，保留旧ID／候选ID映射，含完整指令、source、参考与TripVVT完整caption/GT。VACE辅助视频只展示全包前两条。完成直接审核、协议与媒体校验后，只上传预筛ZIP和最终ZIP到已授权百度目录，以上传器成功回执为完成依据。

最终ZIP：`/apps/bypy/lance_review/a849_external_vbench_v4_116_scroll.zip`。解压后打开 `index.html`，输入审核入口为 `input_audit.html`。32条候选包：`/apps/bypy/lance_review/a849_tripvvt_pilot32_20261115_scroll.zip`。主allocation202896按用户要求保留，完成流水线不取消allocation。
