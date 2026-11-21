# LTX2.5 distilled＋DataMake V7 单阶段实验

固定10条UE输入，与 `benchmark_full10s_20261112/outputs/ltx_base` 的完整10秒历史结果配对。仅新增蒸馏推理，不补跑dev、不扩大300条，不修改A849作业、历史JSON或共享DataMake源码。

样本：identity_01、identity_02、single_piece_01、whole_outfit_01、action_01、action_02、site_01、hair_01、hair_color_01、camera_01。执行顺序先 identity_01、whole_outfit_01、action_01，逐条技术检查通过才继续后七条；技术失败立即停止并给余下样本写明确未运行原因，不自动重试。

蒸馏DiT SHA256 `31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4`，revision `49b093a39c19169a8a81721dcc541e07e945232f`。单阶段原画幅，121帧/12fps，1280×736或736×1280，BF16，无LoRA。V7参考→SOURCE→生成区；CFG1、STG0、rescale0、ref_cfg1。使用本地官方固定9个sigma、8次更新的ancestral Euler，eta1/s_noise1，初始seed42，步级噪声seed10042。

文本仅复用同Gemma和feature extractor的**pre-connector**缓存；重新加载蒸馏DiT中的video connector。完整权重及projection逐tensor哈希证据见metadata/weights.json。SOURCE/图片/motion沿用已哈希验证的VAE缓存，GT只进入review打包代码。

## 蒸馏权重、训练方法与蒸馏 LoRA

核对日期：2026-11-14；范围为本次 LTX-2.5 10条实验及当日公开官方资料。

本次直接使用官方完整蒸馏 DiT `ltx-2.5-22b-distilled-transformer-bf16.safetensors`，采用官方固定8步、CFG=1配置。没有自行训练蒸馏模型，也没有加载蒸馏 LoRA、A849或其他LoRA。这里的蒸馏主要体现为减少推理步数；完整蒸馏DiT仍为22B规格，不能将它理解为本次做了参数压缩或量化。

已查阅的官方 LTX-2.5 模型说明没有明确给出该权重的具体蒸馏训练配方，因此暂不能认定为DMD、DMD2或consistency distillation。`ancestral Euler` 是推理采样器，不能据此反推出蒸馏训练算法。后续若找到官方明确披露，再补充算法名称与来源。

官方另提供 `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`，用于dev-transformer工作流。蒸馏LoRA以低秩权重增量的形式，让匹配的dev基座适应少步采样；它不能独立运行，仍需加载完整基座。官方两阶段代码还可让第一阶段使用dev采样、第二阶段加载蒸馏LoRA做少步细化。完整蒸馏DiT则直接提供蒸馏后的模型权重，本次采用此路线。不能默认“dev＋蒸馏LoRA”与完整蒸馏DiT逐参数或输出等价，也不能从本次结果推断“A849＋蒸馏LoRA”的兼容性或编辑效果；后者需要独立配对验证。

来源：

- [官方LTX-2.5模型说明：完整蒸馏DiT与dev工作流用蒸馏LoRA](https://huggingface.co/Lightricks/LTX-2.5#model-family--checkpoints)
- [官方工作流说明：直接加载蒸馏transformer与base＋蒸馏LoRA的使用区别](https://docs.ltx.io/open-source-model/usage-guides/text-to-video/)
- 本地官方实现：`runtime/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py` 的 `distilled` 参数说明，以及 `ti2vid_two_stages.py` 的 `stage_2` LoRA组装；路径相对于本README前述共享NATIVE项目。

## 入口和状态

- `scripts/prepare.py`：登录节点完整哈希和配对/视频解码。
- `scripts/audit_timeline.py`：时间轴、缓存采样帧序列、历史manifest及参考顺序验证。
- `scripts/test_cpu.py`：小张量单图/双图/motion/无参考四种V7拼接、提取和真实官方随机采样器的条件保护检查。
- `scripts/controller.sh`：后台控制器，由pilot.sbatch申请独立testqueue单卡作业，8 CPU、96GB、2小时。原分配202767在检查后出现另一占有全部4张GPU的step，故不复用。A849位于202855，完全不接触。
- `scripts/run_gpu.sh`：Slurm独占step内再加worker锁和node GPU UUID锁，检查单张H200且已用显存低于2GiB。
- `scripts/worker.py`：模型常驻串行10条；每步验证条件区不变和数值有限；每条输出完整解码121帧。
- `scripts/finalize.py`：GPU推理结束后在同一batch作业内，落盘失败/未运行原因，自动生成报告和审核包。
- `metadata/run_status.json`、`outputs/<id>.json`：总体与逐条状态；`logs/worker.log`：GPU日志；`logs/package.log`：打包日志。

完成或失败后生成 `review/index.html` 和 `review.zip`，均只包含SOURCE、有序参考、GT、dev、成功的蒸馏输出和审核证据；不含VACE、模型、缓存或凭据，不自动上传。无评分连续滚动页面支持任务导航、0–10秒同步播放和完整视频入口。人工可见差异需要查看后记录，技术通过不代表人工通过。

速度见 `SPEED_REPORT.md`、`metadata/speed_report.json`。模型加载单列，保留首条，并附排除首条辅助统计。主要口径为两边denoise+decode，其中decode均含VAE、编码写盘和完整解码检查；另列VAE、写盘、检查分项及排除新检查时间的辅助加速比。十条未全部成功时不填十条总体加速比。所有依赖版本、GPU、显存峰值和源码哈希落盘。历史比较受环境、缓存、竞争影响，并非严格同场基准；四卡未实测；结论仅适用于本次10条UE。

稳定运行后后台自行处理和退出GPU step，不需要对话持续轮询。现有interactive allocation本身由用户持有，不取消；本实验独立batch作业退出后释放其单卡分配。
