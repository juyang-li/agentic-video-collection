# 4×H200：VACE / MiniMax H3 的10秒推理速度与配置

给师兄及其 Agent 核对推理速度用。2026-11-14 从现有完成记录重新统计，没有启动新推理、调整环境或申请GPU。“10秒视频”指共同时间轴0–10秒；原生容器因端点和对齐会略长。

## 先看速度和并行方式

| 模型 | 四卡怎么用 | 统计样本 | 每条平均耗时 | 中位耗时 | 4卡稳定吞吐量折算 |
| --- | --- | ---: | ---: | ---: | ---: |
| VACE | **每卡一个常驻pipeline，各生成不同样本**，同时4条 | 已完成62条 | 4003.97秒（66.73分钟） | 4034.09秒（67.23分钟） | **约3.60条10秒视频/小时** |
| MiniMax H3 | **四卡协同一条**，TP2＋Ulysses2，单请求串行 | 已完成80条 | 983.20秒（16.39分钟） | 871.95秒（14.53分钟） | **约3.66条10秒视频/小时** |

两者的“一条耗时”不能直接按四倍换算：VACE单条占一张卡，H3单条占四张卡。VACE的单卡平均吞吐量约0.90条/小时；H3这里没有测单卡配置。

这是由逐条记录折算的**持续生成能力**，不是提交Slurm到拿到全部结果的墙钟速度。排队、模型加载、预处理、失败重试、作业到期中断、打包和上传都不计入主表；VACE实际分片尾部也可能不满4路。不要用跨多次分配的最早/最晚时间相减推算模型纯速度。

### 耗时口径与限制

- VACE用 `generation_seconds`：CUDA同步后围住 `pipe(...)` 计时，包含pipeline内部文本/条件编码、去噪和VAE输出，**不只是DiT去噪**；不含外部准备mask/pose、读取SOURCE、预览写盘、MP4保存及完整解码检查。若改用记录的 `completed_unix-started_unix`，62条平均4041.48秒，四卡约3.56条/小时，仍不含模型加载和独立条件准备。
- H3用客户端 `total_seconds`：从提交前记录起到轮询结束、下载和完整输出解码校验，平均983.20秒；服务返回的 `inference_time_s` 平均979.59秒。服务只加载一次，不逐条重启。
- VACE只统计用户停止时完成的62条，**不是80条全任务均衡测量**：camera为0条，hair_color只有2条。4个中断样本不计入吞吐量，也不把它们耗掉的时间摊入成功样本。
- H3统计全部80条，8组各10条；不是之前只跑少量首条时的估算。参考数量与动作输入影响很大，见下表。
- 仅验证H200，不承诺A100、H100或其他设备的速度。这里两个协议的帧率、帧数、画幅、模型和条件路径不同，不是同算量benchmark，也不能据此比较模型质量。

计算方法使用均值：VACE `4×3600 / mean(generation_seconds)`；H3 `3600 / mean(total_seconds)`。不要将H3再乘4。也不要用中位数得到的“典型样本吞吐量”代替整批均值。

### 不同任务的差异

每条时间均为组内均值；VACE列是单卡一条，H3列是四卡一条。四卡条/小时仍按上面的并行方式折算。

| 任务 | VACE完成数 | VACE单条分钟 / 四卡条每小时 | H3完成数 | H3单条分钟 / 四卡条每小时 |
| --- | ---: | ---: | ---: | ---: |
| identity | 10 | 63.91 / 3.76 | 10 | 14.02 / 4.28 |
| single_piece | 10 | 67.76 / 3.54 | 10 | 14.58 / 4.11 |
| whole_outfit | 10 | 67.62 / 3.55 | 10 | 14.68 / 4.09 |
| action | 10 | 71.37 / 3.36 | 10 | 30.02 / 2.00 |
| site | 10 | 63.63 / 3.77 | 10 | 14.48 / 4.14 |
| hair | 10 | 67.37 / 3.56 | 10 | 14.61 / 4.11 |
| hair_color | 2 | 60.43 / 3.97 | 10 | 14.08 / 4.26 |
| camera | 0 | 未测 | 10 | 14.61 / 4.11 |

H3非动作70条平均866.38秒，四卡约**4.16条/小时**；动作10条平均1800.99秒，约**2.00条/小时**。H3动作条件是SOURCE视频＋完整motion视频，而非静态图。VACE动作输入为pose视频＋3张SOURCE外观参考。不能拿两者动作单条时间直接比较单卡性能。

## VACE：实际配置

| 项目 | 本次使用值 |
| --- | --- |
| 模型 | `Wan-AI/Wan2.1-VACE-14B-diffusers` |
| revision | `db79b90c60bbb45ceec9e41b9d5a4df934538ac4` |
| 框架 | diffusers `WanVACEPipeline`，0.35.2 |
| Python / torch / transformers | 3.12.16 / 2.10.0+cu126 / 4.57.1 |
| GPU并行 | 4个独立进程，每进程仅可见1张H200，每进程常驻加载模型、连续处理分片 |
| 输出 | 1280×736或736×1280，121帧，12fps，采样时刻0–10秒，容器121/12秒 |
| 采样 | `UniPCMultistepScheduler`，flow_shift=5，40步，guidance_scale=5，conditioning_scale=1，seed=42 |
| 精度 | transformer BF16，VAE FP32 |
| 模型驻留 | `pipe.to('cuda')`；当前worker未调用model/sequential CPU offload、显式VAE tiling/slicing或 `torch.compile` |
| 文本 | 完整adapted_prompt，max_sequence_length=512，先检查token数，无静默截断 |
| 参考 | 按输入顺序传reference_images；本批通常0–3张，不做参考替换诊断 |
| 重绘 | 白mask可编辑，送入pipeline前可编辑SOURCE RGB填128，黑区保留原像素 |
| 动作 | 官方pose适配，SOURCE第0/60/120帧作外观参考；不把pose置灰 |
| CPU | 入口 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`；四卡作业配置16 CPU；ORT pose池限制4/1线程 |
| 显存记录 | 62条 `peak_allocated_bytes` 最大约69.23GiB/卡；是PyTorch allocated口径，非NVML总占用 |
| 加载 | 完成记录中的model_load_seconds范围23.49–133.54秒；受缓存等影响，单独列出，不摊入每条生成计时 |

实际worker关键调用（摘录，不是完整启动器）：

```python
vae = AutoencoderKLWan.from_pretrained(modelpath, subfolder='vae',
    torch_dtype=torch.float32, local_files_only=True)
pipe = WanVACEPipeline.from_pretrained(modelpath, vae=vae,
    torch_dtype=torch.bfloat16, local_files_only=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.)
pipe.to('cuda')
# video/mask/refs均提前准备；普通重绘的video已经按mask把RGB填128。
with torch.inference_mode():
    generated = pipe(video=video, mask=mask, reference_images=refs or None,
        prompt=adapted_prompt, negative_prompt=negative_prompt,
        height=h, width=w, num_frames=121, num_inference_steps=40,
        guidance_scale=5., conditioning_scale=1.,
        generator=torch.Generator(device='cuda').manual_seed(42),
        output_type='pil', max_sequence_length=512).frames[0]
```

四卡入口用 `CUDA_VISIBLE_DEVICES` 将4个worker分开，各取 `inputs[shard::4]`，**不是torchrun/DDP把同一条VACE分到4张卡**。首次加载一次后循环处理多条；不要每条重新 `from_pretrained`。

本次VACE62条已由用户停止并释放其4卡分配，不能为复现本文数字自行恢复旧任务。

## MiniMax H3：实际配置

| 项目 | 本次使用值 |
| --- | --- |
| 模型 | 本地 `MiniMaxAI/MiniMax-H3`，**H3-Base Ref2VA**；没有Context-IR，不调用付费API |
| 权重revision | `0919fb684102ef115126612f3fbb84d1027e7317` |
| 框架 | SGLang multimodal生成，commit `fa179312e94c764f8637724ea2a364d302888906`，editable安装 |
| Python / torch / transformers / diffusers | 3.12.16 / 2.14.1 / 5.19.0 / 0.37.0 |
| flashinfer-python | 0.7.4 |
| 拓扑 | num_gpus=4，tp_size=2，ulysses_degree=2，ring_degree=1，encoder_parallel=auto |
| 并发 | batching_max_size=1；客户端上一条completed并下载校验后才提交下一条 |
| 输出 | 短边768，1344×768或768×1344；24fps，10秒请求原生输出243帧（10.125秒） |
| 采样 | 50次原生去噪更新，flow_shift=12，audio_flow_shift=3，seed=42，quality=lossless |
| 精度 | DiT原生BF16/FP32混合路径；text encoder BF16；VisualVAE FP32驻留、原生FP16 autocast解码 |
| 精度/加速开关 | quantization=None，enable_torch_compile=False，cache_dit_config=None，lora_path=None，enable_cfg_parallel=False |
| attention | server_config记录attention_backend=None、component_attention_backends={}；保留当前H3原生选择，不声称另配了dense attention |
| offload | server_config记录dit_cpu_offload=False、text_encoder_cpu_offload=False |
| 服务模式 | performance_mode=speed，warmup_mode=off；speed名称不能据此推断启用了compile或缓存，要看实际开关 |
| CPU | OMP/MKL各4，TOKENIZERS_PARALLELISM=false；服务step使用16 CPU。历史有1 CPU绑定问题，已改为16，证据见下表 |
| 显存记录 | 本批完成结果 `server_response.peak_memory_mb` 最大78938；保留服务字段原口径，不与VACE的PyTorch allocated直接比较 |

当前SGLang源码已应用兼容补丁：C++模板函数命名查找、VisualVAE嵌套权重识别、缺少memfd_create时普通解码回退。安装曾用 `SGLANG_BUILD_RUST_EXTS=none` 关闭可选Rust扩展。不要在师兄已有环境盲目重新安装、打补丁或升级；先比对环境及版本。

历史服务启动参数如下；这是配置摘录。运行时必须在获授权的4×H200 `testqueue`分配内，将输出路径及端口换成自己的独立值，不直接执行历史任务脚本。

```bash
sglang serve --model-path "$H/runtime/models/H3/Ref2VA" \
  --served-model-name MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 --tp-size 2 --ulysses-degree 2 --encoder-parallel auto \
  --performance-mode speed --enable-torch-compile false --warmup-mode off \
  --batching-max-size 1 --host 127.0.0.1 --port 30121 \
  --output-path "$OUT/server"
```

请求固定项：`task=ref2va`、`num_outputs_per_prompt=1`、`num_inference_steps=50`、`flow_shift=12`、`audio_flow_shift=3`、`seed=42`、`quality=lossless`、`reference_resize_mode=max`、target短边768/时长10秒。SOURCE本身就是第一项video condition；动作时再传完整motion video，图片则单独按顺序编号。结构化prompt的标签必须对应condition顺序。

### H3主要时间在哪里

80条perf记录中的阶段平均值：

| 阶段 | 平均秒数 |
| --- | ---: |
| 文本编码 | 6.20 |
| 参考视频/图片视觉编码 | 6.46 |
| 去噪 | **951.73** |
| 输出解码 | 2.29 |
| 服务总时间 | 979.59 |
| 客户端提交到下载校验总时间 | 983.20 |

阶段时间不覆盖全部服务开销，不要求加和等于总耗时。identity组去噪平均814.64秒，动作组1753.98秒；动作变慢主要体现在记录的去噪阶段，不能简单归因于多读了一个MP4。这里是阶段耗时观察，不是某个单变量的因果试验。

历史 `cpu_affinity_correction.json` 记录服务step可用CPU从1改到16、推理参数未改；不同样本及运行阶段同时变化，不能从这些记录声称“加CPU必然带来某倍提速”。师兄Agent应检查实际子进程的 `Cpus_allowed_list`，不能只看sbatch申请数。

## 师兄Agent定位慢在哪里：先对齐这些条件

1. **先确认耗时的含义。** 分开记录模型加载、SOURCE/ref编码、去噪、VAE解码、写盘；CUDA计时前后同步。VACE的generation_seconds不是纯去噪，H3客户端时间也不是纯去噪。
2. **确认硬件与并行拓扑。** 核对实际GPU名称、可见设备数、显存占用及每个worker所在GPU。VACE本配置是4个独立pipeline；H3是TP2＋Ulysses2的一个服务。不要将“申请4张卡”视为已经用上了正确并行方式。
3. **核对帧数、空间大小、参考长度/数量、步数、CFG与dtype。** 不能只比较文件名里的“10秒”：H3是243帧24fps，VACE121帧12fps；同模型改变空间/时间token数量也会改变代价。
4. **检查是否逐条重载模型、是否实际启用了CPU offload或额外切片。** 本文模型常驻GPU，批内不重新加载。若师兄为了显存采用另一条路径，应在对照中写明，不能直接套用这里的速度。
5. **检查CPU绑核与I/O。** H3实际子进程是否只看到1核，是否每次从网络加载权重，是否把SOURCE/ref重复编码；分别观察阶段计时，不靠总时间猜瓶颈。纯CPU检查放login；GPU测试只允许testqueue，先复用现有记录，不为查本文配置提交新作业。
6. **优先核对实际运行参数与版本，保留相同输入。** 本文没有验证任何新加速方法。降低分辨率/步数/参考数量、换dtype、量化、compile或近似attention都会改变对照条件，不能包装成相同协议下已经验证的提速。

## 本地文件位置与证据

完整的四模型用法及最小调用示例见同目录 [README.md](README.md)。下列路径供Agent直接读取；只读查看，不运行固定历史输出目录的脚本。

| 内容 | 完整路径 |
| --- | --- |
| VACE环境 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace` |
| VACE模型snapshot | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/hf_cache/models--Wan-AI--Wan2.1-VACE-14B-diffusers/snapshots/db79b90c60bbb45ceec9e41b9d5a4df934538ac4` |
| VACE版本、工具位置和pipeline签名 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/runtime/vace_ready.json` |
| VACE推理与四卡入口 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/vace_worker.py`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/run_vace.sh` |
| VACE62条逐条耗时/配置/显存 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/outputs/vace` |
| VACE采样进度日志示例 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/logs/all_vace_0.log` |
| VACE停止清单 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/metadata/inference_stop_vace_20261114.json` |
| H3真实权重目录 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/models/H3/Ref2VA` |
| H3真实环境 / SGLang源码 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/sglang/python` |
| H3依赖锁 / 必要补丁 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/metadata/requirements.lock.txt`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/metadata/sglang_build_portability.patch` |
| H3实际服务配置 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/metadata/server_config.json` |
| H3服务启动 / 客户端计时 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/scripts/serve.sh`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/scripts/client.py` |
| H3请求 / 阶段耗时 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/metadata/requests`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/metadata/perf` |
| H3CPU绑核修正记录 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/metadata/cpu_affinity_correction.json` |
| H3全部80条结果记录 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/outputs` |

H3的10秒任务 `h3/runtime/{env,models,sglang,bin}` 是软链接，分别指向 `minimax_h3_ref2va90_local_20261112/runtime/` 下同名目录。上表给出的环境与权重是实际复用位置；不要复制软链接后以为模型文件已经复制完毕。

### CPU重算统计（不加载模型、不提交GPU作业）

```python
from pathlib import Path
import json
from statistics import mean, median
R = Path('/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112')
for model, subdir, key, concurrency in [
    ('VACE', 'outputs/vace', 'generation_seconds', 4),
    ('H3', 'h3/outputs', 'total_seconds', 1),
]:
    records = [json.loads(p.read_text()) for p in (R / subdir).glob('*.json')]
    records = [d for d in records if d.get('status') == 'complete']
    values = [d[key] for d in records]
    print(model, 'completed=', len(values), 'mean_s=', mean(values),
          'median_s=', median(values), 'four_gpu_videos_per_hour=', concurrency*3600/mean(values))
    for group in ['identity', 'single_piece', 'whole_outfit', 'action', 'site', 'hair', 'hair_color', 'camera']:
        # 不用glob('hair_*.json')分组，否则会把hair_color混进hair。
        times = [d[key] for d in records if d['id'].rsplit('_', 1)[0] == group]
        if times:
            print(group, len(times), mean(times)/60, concurrency*3600/mean(times))
```

最后注意：此前“VACE四卡约15条/小时”来自**49帧、12fps、约4秒**的另一批记录，不适用于这里121帧的10秒视频。本页3.60/3.66条每小时取自10秒完成记录；完整样例本次未重新运行。
