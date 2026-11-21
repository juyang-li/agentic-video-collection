# LTX2.5 Base（distilled DiT）＋DataMake V7：300条

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。 review/media浏览副本已清理（VLM另含review_hair24/media），本地审核页需重建或恢复后播放；实际送评证据保留。

本批对冻结300对全部重新推理：UE 200、CoinVE 75、TripVVT 25。10条先导结果不复用。独立目录保存结果，不改动A849、历史dev或共享DataMake源码。

## 固定模型与协议

完整官方BF16权重 `ltx-2.5-22b-distilled-transformer-bf16.safetensors`，revision `49b093a39c19169a8a81721dcc541e07e945232f`，DiT SHA256 `31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4`。沿用原生trainer环境、Gemma4文本包及conv VAE。

实际8次ancestral Euler更新；sigma为 `[1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]`。CFG1、STG0、rescale0、ref_cfg1、eta1、s_noise1；初始seed42、步级噪声seed10042、解码seed42。

不加载A849、蒸馏LoRA或其他LoRA。完整蒸馏DiT已提供少步模型权重；蒸馏LoRA用于匹配dev基座的另一条少步适配路线。已查阅官方资料未明确披露本权重具体的蒸馏训练算法，不将其认定为DMD/DMD2/consistency distillation；ancestral Euler是采样器。来源：[官方模型说明](https://huggingface.co/Lightricks/LTX-2.5)、[官方工作流说明](https://docs.ltx.io/open-source-model/usage-guides/text-to-video/)，核对日期2026-11-14。

单阶段、原画幅、BF16，无量化、compile、offload、上采样、音频生成。V7物理token/位置顺序为有序参考→SOURCE→生成区；参考/SOURCE是干净条件，仅生成区更新和提取。每一步验证条件区不变与数值有限。

输入清单逐字节复制自 `frozen300_20261113/package/inputs.jsonl`，ID、完整prompt、参考顺序、SOURCE和时间轴均固定。画布为1280×736或736×1280，12fps；UE121帧，真实视频按清单49–97帧，不变速/补长。图片保持V7原生尺寸与最少edge padding。CoinVE可无参考；TripVVT official garment恰好一张。

SOURCE/参考及文本缓存仅在CPU验证媒体、时间索引、shape、有限值和SHA256后只读复用。100份真实视频窗口缓存沿用A849已生成的独立缓存。相同packed Gemma和feature extractor的pre-connector特征可复用，蒸馏video connector重新加载，历史connector输出不复用。GT和审核caption只进入package进程。

## 执行与验收

CPU预检：`scripts/preflight.py`；小张量单图/双图/motion/official garment/无参考测试：`scripts/test_cpu.py`。完整权重哈希、缓存核验和实际依赖版本记录在metadata。实现与共享依赖源码哈希冻结于 `metadata/implementation.json`，GPU worker加载前检查。

`run.sbatch`仅申请testqueue的4张H200、32 CPU、384GB、2小时。四个独立常驻worker，每卡加UUID锁，初始显存检查低于2GiB。worker0先运行identity_01、whole_outfit_01、action_01、coinve_01、tripvvt_01、coinve_11，覆盖单图/双图/motion/无参考/官方服装图/49帧。六条全部通过后写smoke_gate，其余worker才加载模型。剩余样本按冻结序号mod4分片，六条门槛结果计入300条，不重复运行。

任何技术失败停止整批、终止其他worker，落盘失败及未运行原因；效果差照实保留。无自动换seed/采样协议、无盲目重试。基础设施中断需人工检查后显式恢复；worker的resume路径仅接受合同相同且哈希/完整解码均通过的complete结果，失败样本不自动重试。当前初次提交入口禁止使用已有输出。

每条 `outputs/<id>.mp4` 和JSON，汇总 `results.jsonl`。成功要求8次更新、条件保护、有限值、正确尺寸、12fps和逐条冻结帧数完整解码。状态分开记录：`metadata/run_status.json`（推理）、`package_status.json`（打包）、`delivery_status.json`/`upload_execution.json`（交付）。不能将submitted或uploading写作完成。GPU作业退出即释放资源。

## 报告与交付

模型加载单列；每条缓存加载、去噪、VAE解码、编码写盘、完整解码验证、总处理耗时和显存峰值均记录。decode_seconds含VAE/写盘/验证，另有分项。四worker实测处理墙钟包含六条门槛阶段，排队、模型加载、打包上传不混入；保留每个worker首条并附排除首条辅助统计。历史dev仅核对同输入/时间轴后给出可配对子集的比较，不虚构300条dev速度或套用先导14.9倍；历史环境和资源竞争有差异，非严格同场基准。

登录节点独立 `scripts/delivery.py` 后台控制器每60秒检查Slurm是否退出；不持续占用对话。退出后用CPU重新验收全部输出并生成完整300条无评分滚动HTML/ZIP。包含完整指令、SOURCE、有序参考、GT和本批输出，支持导航/筛选/冻结区间同步播放/原件入口；TripVVT caption只用于审核。编辑效果仍待人工评判。ZIP不含VACE中间媒体、模型、缓存或凭据。

正式审核包：`ltx25_distilled_v7_300_review.zip`。300条全部成功并通过离线链接、媒体SHA和ZIP CRC检查后，按用户明确批准，后台自动上传至：

`/apps/bypy/lance_review/ltx25_distilled_v7_300_review.zip`

仅上传ZIP，支持断点续传，以上传器回执为准，不另做远端整包回读。开始上传后不需要对话轮询。推理失败只生成带明确状态的本地partial审核包，不自动上传正式包。上传失败保留日志与可恢复状态，不重新运行推理。

本批已提交：作业203465（testqueue、4×H200），提交于2026-11-14 21:10新加坡时间。登录节点后台交付控制器PID2193455；提交时处于Priority排队，调度器暂估11月15日00:37:30启动，此估计可能变化。submitted/pending不代表已开始GPU推理。

## 百度交付完成与上传修复

2026-11-14T22:41:53.486447+08:00：百度create接口返回成功，已上传完整300条审核包至 `/apps/bypy/lance_review/ltx25_distilled_v7_300_review.zip`，9,636,715,992字节。原ZIP SHA256保持 `f90113ac1af0b276a86cba6b28d1e9a6ace1d6124fb2dfc11e36369d7bfed361`，未重跑推理或改包。

旧4MiB分块共2298块，0–2047已确认；诊断重传2047成功，2048及2297返回31299/Invalid param part_id，因此停止旧上传并保留metadata/baidu_upload_4mib_stalled.json。独立recover_upload.py改为8MiB、1149块，使用推荐c.pcs.baidu.com服务器、4个持久连接、短超时、有界重试和即时错误日志。1149块MD5均匹配，最终create回执路径/字节数匹配；回执见metadata/baidu_upload_8mib.json、upload_completion_audit.json。所有阶段状态已同步complete。未做额外远端回读。

原始推理源码身份记录不改写；恢复上传工具独立于冻结推理入口。历史4MiB上传入口不能直接用于本包，恢复时使用scripts/recover_upload.py。
