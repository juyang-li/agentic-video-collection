# 100 条单轮：LTX-2.5 / A849 / VACE

## 最终交付（2026-11-12）

三个模型各100条正式输出及6条换参考诊断已完成，seed42、40步、49帧/12fps。输入ZIP与结果ZIP均由上传器返回complete，仅上传ZIP；没有额外远端核对或回读。

- 输入：`/apps/bypy/lance_review/a100_inputs_scroll_20261111.zip`
- 三模型对比：`/apps/bypy/lance_review/a100_results_scroll_20261111.zip`（约1.36GiB）
- 结果包706段视频完整解码通过；解压后打开`index.html`，另含独立换参考诊断页。

正式输出全数保留，未按效果换样。协议问题的早期尝试独立归档，不混入正式结果。VACE官方gray128填充、源侧分割及pose适配已经记录；粗mask、浅色马尾漏分及多人物遮挡等局限在包内明示，不等同于人工质量通过。

正式生成中位耗时：基座160.9秒、A849161.4秒、VACE933.7秒；仅正式生成累计约35.5 GPU小时，不含加载、条件准备和历史修复。详细协议与计时见`metadata/final_inference_summary.json`。四条头发条件改用源头部ROI逐帧解析，短暂空帧采用最近有效源mask，记录于`repair_hair_cpu.py`及condition.json；未读取GT。

下文保留运行流程与历史过程记录。

固定 seed：抽样 `20261111`，推理 `42`。仅复用师兄正式 test_manifest；冻结结果在 `metadata/selection.jsonl`，模型输入单独在 `metadata/inputs.jsonl`，不含 GT 或 target caption。

已冻结 100 条、100 个不同 SOURCE：UE 八组各 10 条，CoinVE 10 条（两编辑 7、三编辑 3），TripVVT 10 条（upper_body 4、lower_body 3、官方 dresses 3）。单件服装 shirts 3、full_body 3、skirts 2、pants 2。采样不读取模型输出。TripVVT source 含合成换装，不把真实数据集来源等同于原生实拍。

统一 0–4 秒含端点的 49 帧，12fps；横屏 1280×736、竖屏 736×1280，保持比例补边；49 个时刻编码播放为 49/12 秒。镜头组保留原指令，只评价前四秒，不要求完成十秒环绕。输入审核包含完整指令、全部参考、SOURCE、GT、来源、官方 caption 和抽样说明，无评分控件。

## 运行入口

当前仅使用用户指定的 `testqueue` 作业 `199052`。所有计算入口检查分区；GPU 入口另外检查 jobid。`watch_allocation.py` 每次查询后 sleep 1800 秒，分配后自动通过显式 `--partition=testqueue` 的 srun 接入，先执行 prepare（输入打包及只上传 ZIP、VACE 环境下载、独立 UE49 帧缓存），再执行 smoke。用户随后明确授权输入打包/上传直接在登录节点执行，不等GPU；用 `A100_ALLOW_LOGIN_PACKAGE=1 bash scripts/package_upload.sh`。推理、编码和条件准备仍只使用testqueue。

```bash
python scripts/watch_allocation.py
# 以下均须在199052分配内部运行；每个stage只运行一次，可复用已完成样本。
bash scripts/launch_stage.sh prepare
bash scripts/launch_stage.sh smoke
bash scripts/launch_stage.sh controls
bash scripts/launch_stage.sh second
bash scripts/launch_stage.sh all
```

`smoke` 为 identity_01、whole_outfit_01、action_01、camera_01；覆盖横竖幅，共 12 个正式输出。`controls` 只替换身份或整套服装参考、保持 source/text/seed 固定，共 6 个额外诊断输出。`second` 为其余六组的首条。扩大执行前要求 agent 写入有具体观察的 `metadata/gate_smoke.json` 和 `gate_second.json`，并由 gate.py 验证文件和协议字段。解码成功不会自动通过协议门槛，也不需要再次向用户请求批准。

启动阶段每 15 分钟检查实际视频和条件；连续两组没有编辑变化时先复核输入和脚本，最多作两次有证据的修正。协议正确后效果弱的输出如实保留，不换样、不按效果筛选。稳定后每半小时简查。参考控制和模型输出均不属于人工质量通过。

## 模型实现

LTX 两组复用师兄原生 `load_inference_models` / `generate_latent` / 卷积 `VideoDecoder`，v7 参考→源→目标，40 步、CFG3、STG1/block28、rescale0.7、ref_cfg1。基座不传 checkpoint；A 为 mixed720p_sequential_visual_dropout step849。只为 UE source/motion 另外编码49帧，复用经身份和哈希校验的原生图片参考与文本缓存，以及真实数据已有49帧缓存。不改原代码、正式清单、训练配置或历史缓存。

VACE 使用独立环境，固定 Diffusers 0.35.2 / WanVACEPipeline、官方 Wan2.1-VACE-14B-diffusers 的下载 commit、BF16、VAE FP32、UniPC flow_shift5、40步、CFG5、conditioning_scale1。依赖版本、模型revision和pipeline代码SHA写入runtime记录；共享读取原生环境已验证的Torch构建，不修改原环境。

VACE mask 为 255 可编辑、0 保持；DINO＋SAM2 从 SOURCE 首/中/末关键帧分割并跟踪，各编辑区域联合。局部条件无检测时记录失败，不以整帧假冒局部mask；新增物体的显式插入ROI需在首测中核对。场景/背景编辑反转人物保护mask；发型扩展32像素、发色扩展6像素，其余12像素。动作参考使用官方 PoseBodyFaceVideoAnnotator，源0/24/48帧提供外观参考；其余任务保留原参考顺序。官方VACE逐次前插参考latent，实际latent顺序反转，在protocol_audit单列记录。

适配prompt只使用原指令中的源上下文、已有合法参考及目标要求，不读取完整GT或TripVVT官方target caption。CoinVE完整有序指令保留，一次生成完成。手工语义查询、插入ROI和prompt模板属于待验证条件，不能提前视为已正确。

`effect_probe.py` 生成 source/输出首中末联系图与逐帧差异，辅助发现复制源视频；差异大小本身不代表编辑完成。正式扩展与交付前必须检查完整视频。每条独立状态保存协议审计、耗时、显存和异常，单条失败继续其余条目。

官方实现依据：[固定版本 WanVACE](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/wan/pipeline_wan_vace.py)、[VACE pose/composition](https://github.com/ali-vilab/VACE/blob/main/vace/annotators/composition.py)。

交付：`review/a100_inputs_scroll_20261111.zip` → `/apps/bypy/lance_review/`，以基础上传器返回成功为准，仅传ZIP，不额外核对远端大小或整包回读。完整输出后用 `package.py --kind results` 制作 SOURCE/参考/GT/三个模型对比包。

2026-11-11：输入审核ZIP已在登录节点完成并上传，基础上传器返回 complete；远端 `/apps/bypy/lance_review/a100_inputs_scroll_20261111.zip`。本地210段视频全部完整解码、100条媒体引用检查通过。没有额外远端大小核对或整包回读。

运行更新：199052 于2026-11-11 22:20:46获配C64Y574，4×H200、56CPU，到期2026-11-12 16:20:46。UE49帧独立缓存90份已完成；LTX基座/A849各四条首测完整解码通过，生成中位耗时约163/164秒（含解码）。VACE镜头首条约918秒；其余首测与换参考诊断正在执行。

首测输入修复：身份改用源人物脸/裸露四肢分区＋SAM2；整套穿搭扩展到衣物、鞋及可被新袖长/裤长覆盖的四肢，保护头部。分区模型为 [SegFormer ATR人体分区](https://huggingface.co/mattmdjaga/segformer_b2_clothes)，revision在runtime记录；关键帧mask已看过，旧漏鞋mask输出保留在metadata/protocol_attempts且不计正式样本。官方pose补齐Matplotlib并将ORT会话限定4CPU线程，不修改原官方算法或训练环境。模型差的合法结果仍保留。

关键协议修正：官方VACE重绘预处理需把mask白区RGB先填为128，再编码。首次只传原RGB＋mask形成额外reactive内容约束，对应身份/服装及其换参考旧输出已归档为非正式协议诊断；修正后保持样本、mask、参考、seed42及40步等参数重新验证。动作仍用官方pose control；镜头按既定方案保留raw RGB全画面control，二者旧正确输出通过显式代码SHA兼容记录复用。最终审核显示实际灰填充输入视频，避免只凭mask名义判断协议。
