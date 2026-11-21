# 四模型推理使用说明（当前共享集群）

本文面向在当前共享集群复用模型的师兄。文件与 API 按 2026-11-14 本地状态核对。先用已准备的 `identity_01` 跑通，再换自己的 SOURCE、英文编辑指令和参考。**GT 和 target caption 均不作为推理输入。** 本文不涉及数据集流水线和历史任务续跑。

| 模型 | 当前实际使用方式 | SOURCE / 图片 / 动作参考 | 已验证硬件与固定参数 |
| --- | --- | --- | --- |
| MiniMax H3 | 本地 H3-Base Ref2VA，SGLang，无 Context-IR | SOURCE 也是 video condition；图片、动作分别为 image/video condition | 4×H200 协同一条，TP2＋Ulysses2；768短边，24fps，50步，seed42，flow12/audio3 |
| VACE | Wan2.1-VACE-14B，diffusers `WanVACEPipeline` | SOURCE＋mask；图片传 `reference_images`；动作先转 pose | 每进程1×H200；1280×736或736×1280，12fps，40步 UniPC，CFG5，flow shift5 |
| LTX2.5 基座 | 官方 22B dev BF16 权重＋DataMake V7 | 有序图片/动作 latent → SOURCE latent → 待生成区域 | 每进程1×H200；同上画幅、12fps，40步，CFG3，STG1/block28，rescale0.7，ref_cfg1，seed42 |
| A849 | 同一 LTX2.5 基座加载 step849 LoRA | 与基座完全相同 | 与基座完全相同 |

LTX/VACE 均验证过49帧和121帧；下文统一复用121帧（0–10秒端点，容器时长121/12秒）。H3验证过4秒请求→107帧、10秒请求→243帧；不能把请求秒数当成实际帧数。

10秒视频的四卡速度、计时口径与慢速排查见 [4×H200：VACE / H3 配置与吞吐量](H200_4GPU_VACE_H3_PERFORMANCE.md)。

## 文件位置速查

下列均为绝对路径。代码示例中的 `E`、`N`、`H`、`V` 只是缩写，分别对应表中的目录；不是新的安装位置。

| 名称 | 完整位置 |
| --- | --- |
| 本项目模型评测根目录 `E` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation` |
| LTX/DataMake 根目录 `N` | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5` |
| H3 本地运行根目录 `H` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112` |
| VACE 保留运行根目录 `V` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111` |
| 共用的已准备10秒输入 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/metadata/inputs.jsonl` |
| 该输入的 LTX 缓存索引 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/metadata/native_cache_index.json` |
| 该输入的 VACE 条件 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/conditions/identity_01/condition.json` |
| 该输入的 H3 请求 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/metadata/requests/identity_01.json` |

## 运行约定与验证范围

纯 CPU 文件检查在 login 执行。需要 GPU 的操作（包含 VAE/Gemma 编码）只能在 `testqueue` 分配内执行，不要伪造 `SLURM_JOB_PARTITION`。下列 `salloc` 只是供使用者申请资源的示例，本次文档编写没有提交它们；排队时等待，不换分区。

```bash
# login；H3需要4张卡，其余模型的最小示例需要1张卡
salloc --account=prj0000000275 --partition=testqueue --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=02:00:00
# H3改为 --gres=gpu:4 --cpus-per-task=16 --mem=192G --time=04:00:00
# 在分配上通过 srun --pty bash 进入计算节点；执行GPU代码前检查：
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
nvidia-smi --query-gpu=name,memory.total --format=csv
```

以上未指定集群未核实的 GPU constraint；必须确认获得的是 H200、卡数正确且卡空闲，再运行下面示例。本文只引用 H200 实测，不承诺其他设备可运行或有相同速度/显存。

输出使用自己可写的共享目录，示例默认 `/scratch/prj0000000275/$USER/model_inference/<时间戳>`，可替换 `OUT`；不要指向已有任务的 outputs/cache/conditions。代码均用独占创建避免覆盖。环境和模型路径通常无需替换。

本次核验覆盖路径存在性、软链接、源码 API/参数、固定配置、准备输入及缓存、Python/Bash 示例语法；已在login用原生环境成功读取并collate `identity_01` 缓存（SOURCE `[1,128,16,23,40]`、文本 `[1,1024,4096]`、1张参考）。**没有重新执行完整示例，没有加载 GPU 模型或新增推理**。显存来自已有完成记录：10秒协议 LTX基座/A849 的 PyTorch峰值分配约48.25GiB/卡，VACE约69.23GiB/卡；H3首测记录最高原生峰值约76.8GiB/卡。不同记录口径不能直接横比，也不是显存上限保证。证据分别为 `benchmark_full10s_20261112/outputs/{ltx_base,a849,vace}/*.json` 和该任务 README 中的 H3 首测记录。

## MiniMax H3

### 用的是什么、文件在哪里

使用 `MiniMaxAI/MiniMax-H3` 的本地 **H3-Base Ref2VA**，权重 revision `0919fb684102ef115126612f3fbb84d1027e7317`。没有 Context-IR，不调用付费 API。SGLang commit 为 `fa179312e94c764f8637724ea2a364d302888906`，当前依赖锁记录 torch2.14.1、transformers5.19.0、diffusers0.37.0、flashinfer-python0.7.4。

| 文件 | 完整位置 |
| --- | --- |
| 权重（含 text encoder、DiT、VisualVAE、audio VAE、processor） | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/models/H3/Ref2VA` |
| Python 环境 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env` |
| SGLang 源码（editable安装） | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/sglang/python` |
| 服务启动入口 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/sglang` |
| pipeline配置 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py` |
| sample配置 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py` |
| 必要兼容补丁 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/metadata/sglang_build_portability.patch` |
| 依赖锁 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/metadata/requirements.lock.txt` |
| 权重核验记录 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/metadata/weights_verified.json` |

10秒任务的runtime只是复用链接：

| 软链接 | 实际位置 |
| --- | --- |
| `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/runtime/env` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env` |
| `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/runtime/models` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/models` |
| `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/runtime/sglang` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/sglang` |
| `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/runtime/bin` | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/bin` |

必要补丁已在当前源码应用：C++模板函数命名查找修复、VisualVAE嵌套权重完整性识别、缺少 `memfd_create` 时走原有普通解码回退。安装时 `SGLANG_BUILD_RUST_EXTS=none` 关闭可选Rust扩展；这些不是权重修改。不要在现有环境重复安装或重复打补丁。

精度采用原生 DiT BF16/FP32 路径；VisualVAE以FP32驻留、按官方FP16 autocast解码。quantization为None，不能给CLI传 `quantization=bf16`。固定单请求、50次原生更新、lossless，不启用 torch.compile、近似注意力、跳步缓存、量化或加速LoRA。

### 怎样使用 ref

`conditions` 是有序数组。第一项 SOURCE 使用 `type=video, role=reference`，对应结构化 prompt 的 `<Video 1>`；后续动作视频继续编号 `<Video 2>` 等。图片单独从 `<Picture 1>` 开始编号，多个视角按原顺序逐项加入。所有 video 项带 `start_time_seconds=0.0`，本地文件用 `file:///...` URI，路径必须能从服务所在节点读取。

prompt包含 `subject_definitions`、`summary`、`retention_analysis`、`detailed_description`、`overall_soundscape`、`non_diegetic_music`。在 definitions 中逐项说明参考类型和视角；在 detailed_description 中放完整英文编辑指令，不因参考多而截断或改写掉子句。SOURCE本身是参考视频条件，不是另有一个专用source字段。动作参考直接作为video条件，不转pose；说明只迁移动作、不复制参考演员/场景。图片不是GT，`target`字段只描述输出尺寸/时长。

当前10秒适配先按冻结121个12fps时刻读取SOURCE/motion，再用 `min(120,(j+1)//2)` 映射为243帧24fps，10秒后仅保持端点；等比例补边至1344×768或768×1344。中间MP4使用 `libx264rgb CRF0`，逐帧像素核验。4秒输入则由49个时刻扩展至107帧。参考图片由 `reference_resize_mode=max` 走原生预处理。换视频时需重做时间采样、画幅和无损编码，再更新条件URI及prompt编号；不能仅改duration而复用旧视频。

适配逻辑完整入口：`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/scripts/prepare.py` 的 `encode(item, record, role)` / `prepare(item)`。它会写历史目录，供阅读和在自己目录重用逻辑，不能直接执行来准备新数据。

### 怎样运行

先在4×H200的 `testqueue` 计算节点启动独立服务。端口示例 `30121` 可替换，先确认未被占用；不要使用历史任务的30011/30012端口。

```bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
export H=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112
export OUT="/scratch/prj0000000275/$USER/model_inference/h3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
printf 'OUT=%s\n' "$OUT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$H/runtime/bin:$H/runtime/env/bin:$PATH"
unset PYTHONPATH
export TRITON_CACHE_DIR="$OUT/triton" FLASHINFER_WORKSPACE_BASE="$OUT/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$OUT/server"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
"$H/runtime/env/bin/sglang" serve \
  --model-path "$H/runtime/models/H3/Ref2VA" --served-model-name MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 --tp-size 2 --ulysses-degree 2 --encoder-parallel auto \
  --performance-mode speed --enable-torch-compile false --warmup-mode off \
  --batching-max-size 1 --host 127.0.0.1 --port 30121 --output-path "$OUT/server"
```

服务前台常驻；另开同一计算节点的终端，设置相同 `H` 和刚才打印/记下的 `OUT`（不要重新生成时间戳）。先查询 `/v1/models` 就绪，再提交、查询和下载。下面客户端复用10秒identity请求，只读历史输入，保存自己请求及结果。客户端与服务必须同节点，`127.0.0.1` 不跨节点。

```bash
export BASE=http://127.0.0.1:30121
"$H/runtime/env/bin/python" - <<'PY'
import json, os, time
from pathlib import Path
import requests
out = Path(os.environ['OUT'])
base = os.environ['BASE']
s = requests.Session()
s.trust_env = False
r = s.get(base + '/v1/models', timeout=10)
r.raise_for_status()
request_file = Path('/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/metadata/requests/identity_01.json')
body = json.loads(request_file.read_text())
# 换输入时在此替换prompt/conditions/target；conditions路径须在服务节点可见。
with (out / 'request.json').open('x') as f:
    json.dump(body, f, ensure_ascii=False, indent=2)
r = s.post(base + '/v1/videos', json=body, timeout=(10, 180))
r.raise_for_status()
job_id = r.json()['id']
with (out / 'submission.json').open('x') as f:
    json.dump(r.json(), f, indent=2)
print('request_id:', job_id, flush=True)
deadline = time.monotonic() + 7200
while True:
    r = s.get(base + '/v1/videos/' + job_id, timeout=(10, 60))
    r.raise_for_status()
    result = r.json()
    if result['status'] == 'completed':
        break
    if result['status'] in ('failed', 'cancelled'):
        raise RuntimeError(result)
    if time.monotonic() > deadline:
        raise TimeoutError(f'保留submission.json，稍后按同一ID查询：{job_id}')
    time.sleep(5)
with (out / 'result.json').open('x') as f:
    json.dump(result, f, indent=2)
with s.get(base + '/v1/videos/' + job_id + '/content', stream=True, timeout=(10, 180)) as r:
    r.raise_for_status()
    with (out / 'identity_01.mp4').open('xb') as f:
        for chunk in r.iter_content(4 * 1024**2):
            if chunk:
                f.write(chunk)
print(out / 'identity_01.mp4')
PY
```

断开客户端不代表服务任务取消：用 `submission.json` 的id继续 GET，不要盲目重复POST。原生结果保存在自己的server目录，也可经content端点下载。

### 已有脚本如何复用

`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/scripts/serve.sh` 固定job199810、端口30011和历史输出；同目录 `client.py --stage pilot|first|all` 固定历史清单与输出，没有通用的输入/输出/URL参数。10秒版本位于 `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3/scripts/serve.sh`、`client.py`，同样属于任务入口。阅读其请求保存、重连和解码核验逻辑即可，不直接运行历史run/continue/resume脚本。

## VACE

### 用的是什么、文件在哪里

官方 `Wan-AI/Wan2.1-VACE-14B-diffusers` revision `db79b90c60bbb45ceec9e41b9d5a4df934538ac4`；当前diffusers0.35.2、torch2.10.0+cu126、transformers4.57.1。VAE FP32、transformer BF16，UniPC flow_shift5，40步、CFG5、conditioning_scale1、seed42；每进程1×H200，四卡运行是四份样本并行。

| 文件 | 完整位置 |
| --- | --- |
| Python环境 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace` |
| 模型snapshot | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/hf_cache/models--Wan-AI--Wan2.1-VACE-14B-diffusers/snapshots/db79b90c60bbb45ceec9e41b9d5a4df934538ac4` |
| 实际pipeline源码 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/lib/python3.12/site-packages/diffusers/pipelines/wan/pipeline_wan_vace.py` |
| 配置/版本/工具记录 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/runtime/vace_ready.json` |
| 推理示范入口 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/vace_worker.py` |
| 条件生成入口、mask规格 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/prepare_conditions.py`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/vace_specs.py` |
| GroundingDINO snapshot | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/hf_cache/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb` |
| SAM2源码 / checkpoint | `/scratch/prj0000000275/jcy/sam2`、`/scratch/prj0000000275/jcy/sam2_ckpt/sam2.1_hiera_base_plus.pt` |
| mask分割/跟踪工具 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/omni_baselines_20260926/vace/generate_vace_masks.py` |
| 人体解析包装器 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/human_parsing.py` |
| 人体解析snapshot | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/hf_cache/models--mattmdjaga--segformer_b2_clothes/snapshots/584abc1e1d260e23c0fc627c5217a09b2b461046` |
| 官方pose源码 | `/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/tools/VACE/vace/annotators/pose.py` |
| pose权重目录 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/hf_cache/models--ali-vilab--VACE-Annotators/snapshots/63512cbbe4e3a98e1636356ef2230da84649f617/pose` |

pose目录含 `yolox_l.onnx` 和 `dw-ll_ucoco_384.onnx`。HF snapshot中的文件是软链接，实际数据在各自 `models--…/blobs/`；复制时需跟随链接或连同blobs目录保存。三个环境的 `bin/python` 最终都解析到 `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer/bin/python3.12`，但它们是不同环境：**必须通过各自环境的bin/python启动**，不能用最终真实解释器路径替代VACE/H3环境。

### 怎样使用 ref

`video` 为等比例缩放补边后的SOURCE帧列表，`mask`为同画幅、同帧数的RGB/PIL帧列表，白255可编辑、黑0保持。普通重绘必须先将白区的SOURCE RGB置为128，黑区保持原像素；只给原始RGB和mask是错误约束。mask从SOURCE分割/跟踪得到，不读GT。头发、服装、背景需根据编辑目标准备不同mask，不能用同一mask覆盖所有任务。

图片按原参考顺序传 `reference_images=[PIL.Image,...]`，没有图片则None。当前pipeline逐个前置latent，内部最终前缀顺序与参数列表相反；保持调用端清单顺序，不手工再反转。完整英文指令保留于 `adapted_prompt`，最大512 token，先核查长度避免静默截断。

有动作参考时：对motion视频采样到同一12fps时间轴，用官方 `PoseBodyFaceVideoAnnotator.forward(motion_frames)` 生成pose帧，pose作为 `video`，mask全白，SOURCE的第0/60/120帧作为外观 `reference_images`（49帧协议用0/24/48）。此时SOURCE不再作为RGB video；pose不得再填128。`prepare_conditions.py` 的 `pose_model(ready)` 限制ORT CPU池为4/1线程。camera任务是历史明确的raw SOURCE控制例外，同样不套用重绘置灰；下面identity示例只走普通重绘。

换SOURCE、编辑部位、motion后重新准备采样、mask/pose；换图片后重新读入图片；换prompt后更新完整 `adapted_prompt` 并检查token数。条件清单包含 `mask_frames`、哈希、可选 `pose_video` 和 `adapted_prompt`；它不是把GT转换出来的输入。

### 怎样运行

在1×H200 `testqueue`节点，将以下Python块保存到自己的 `vace_one.py`。复用prepared identity条件，不会向历史conditions写预览文件。

```python
import json, os, sys
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION') == 'testqueue'
assert os.environ.get('SLURM_JOB_ID')
E = Path('/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation')
T = E / 'benchmark_full10s_20261112'
sys.path.insert(0, str(T / 'scripts'))
from media import video_frames, write_preview
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler
assert torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
out = Path(os.environ['OUT'])
out.mkdir(parents=True, exist_ok=False)
item = next(json.loads(x) for x in (T / 'metadata/inputs.jsonl').read_text().splitlines()
            if json.loads(x)['id'] == 'identity_01')
c = json.loads((T / 'conditions/identity_01/condition.json').read_text())
assert c['status'] == 'complete' and not c.get('pose_video')
video, _ = video_frames(item, item['source'])
mask = [Image.open(p).convert('RGB') for p in c['mask_frames']]
assert len(video) == len(mask) == 121
refs = [Image.open(r['absolute_path']).convert('RGB') for r in item['references']]
erased = []
for frame, m in zip(video, mask, strict=True):
    pixels = np.asarray(frame).copy()
    pixels[np.asarray(m)[:, :, 0] > 127] = 128
    erased.append(Image.fromarray(pixels))
ready = json.loads((T / 'runtime/vace_ready.json').read_text())
p = ready['vace']['path']
vae = AutoencoderKLWan.from_pretrained(p, subfolder='vae', torch_dtype=torch.float32,
                                      local_files_only=True)
pipe = WanVACEPipeline.from_pretrained(p, vae=vae, torch_dtype=torch.bfloat16,
                                      local_files_only=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.)
pipe.to('cuda')
assert len(pipe.tokenizer(c['adapted_prompt'])['input_ids']) <= 512
w, h = item['canvas_wh']
with torch.inference_mode():
    frames = pipe(video=erased, mask=mask, reference_images=refs or None,
        prompt=c['adapted_prompt'],
        negative_prompt='blurry, excessive noise, compression artifacts, flickering, distorted proportions, deformed facial features, extra limbs, disfigured hands, unnatural transitions',
        height=h, width=w, num_frames=121, num_inference_steps=40,
        guidance_scale=5., conditioning_scale=1.,
        generator=torch.Generator(device='cuda').manual_seed(42),
        output_type='pil', max_sequence_length=512).frames[0]
assert len(frames) == 121
write_preview(frames, out / 'identity_01.mp4', 12)
print(out / 'identity_01.mp4')
```

```bash
# 计算节点；将路径替换为自己保存的脚本
export OUT="/scratch/prj0000000275/$USER/model_inference/vace_$(date +%Y%m%d_%H%M%S)"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python "$HOME/vace_one.py"
```

### 已有脚本如何复用

10秒 `vace_worker.py --stage smoke|second|all --shard 0..3 [--control]`：smoke固定identity_01/action_01，second固定其他组首条，all遍历历史清单并四路分片，control为换参考诊断。它读取固定ROOT并写固定outputs，不能传任意SOURCE/输出目录。`prepare_conditions.py --stage ... --shard ... [--only ID]` 根据同任务mask规格生成条件，也会写历史目录。较早 `ltx25_vace_a100_20261111/scripts/` 入口还包含job199052检查。不要把run_vace.sh、launch_stage.sh或旧worker当成新任务通用启动器；复用模型加载、条件生成函数和上述独立调用。

## LTX2.5 基座

### 用的是什么、文件在哪里

官方 `Lightricks/LTX-2.5` revision `49b093a39c19169a8a81721dcc541e07e945232f` 的 **22B dev**，不是distilled或IC-LoRA Ingredients。视频生成使用原生LTX-2代码，DataMake扩展V7参考位置/文本缓存/条件组装；不生成音频、不使用upscaler。当前原生解释器实读为Python3.12.16、**torch2.13.0**、accelerate1.15.0、peft0.21.2、transformers5.14.1、safetensors0.8.0。历史checkpoint依赖记录为torch2.10.0+cu126，与当前环境存在差异；本文未调整环境，也未在当前torch版本重跑GPU推理，历史耗时/显存不构成对此版本的验证；上游源码revision `e98f6d927d83c92ef2be6aec76c8440b19306e39`。

| 文件 | 完整位置 |
| --- | --- |
| 当前真实权重目录 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/models/LTX-2.5-native` |
| DiT | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/models/LTX-2.5-native/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| Gemma文本包 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/models/LTX-2.5-native/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| conv VAE | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/models/LTX-2.5-native/vae/ltx-2.5-video-vae-conv-bf16.safetensors` |
| 原生Python环境 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer` |
| 上游LTX-2源码 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/repos/LTX-2` |
| 共用训练/LoRA配置 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/configs/training/mixed720p_sequential_visual_dropout/train.yaml` |
| 负提示词与guidance设置 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/configs/inference/stronger_guidance/inference.json` |
| 权重revision / 校验记录 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/models/native_model_revision.json`、`/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/verification/native_weights_verified.json` |
| 路径迁移表 / 解析器 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/path_relocations.json`、`/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/scripts/common/project_layout.py` |
| 10秒缓存读取 / 推理参考入口 | `/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/native_cache.py`、`/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts/native_worker.py` |

权重记录里的 `local_dir` 仍是旧的 `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/models/LTX-2.5-native`；该旧地址不是当前权重入口，也不能假定存在软链接。通过 `encoding_helpers.weight_paths()` 调用 `project_layout.resolve_path()` 将models映射到runtime/models，或使用上表真实目录；不要改历史receipt来迁就路径。

必要DataMake模块均在 `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/scripts/common/`：

- `native_reference_inference.py`：`load_inference_models`、`encode_negative_context`、`build_generation_state`、`generate_latent`。
- `reference_positions.py`、`native_reference_strategy.py`：V7时空位置与参考顺序。
- `native_cached_dataset.py`、`compact_text_cache.py`：单样本collate与文本缓存恢复。
- `encoding_helpers.py`、`project_layout.py`：权重与迁移路径解析。
- `native_image_cache_contract.py`、`encode_native_dataset.py`：图片预处理、VAE/Gemma编码基础实现。

**当前V7必要模块未纳入Git。** 只拉取官方LTX-2仓库不能复现这里的多参考编辑；需要保留当前DataMake模块、原生源码、环境、配置和权重。不要修改师兄现有环境或训练目录。

### 怎样使用 ref

输入sample只有三个顶层对象：`source`、`references`、`conditions`。

| 对象 | 编码与形状（collate前） |
| --- | --- |
| `source` | VAE编码的BF16 `latents [128,F,H/32,W/32]`，以及 `num_frames/height/width/fps` 元数据；121像素帧→16 latent帧，49帧→7，fps=12 |
| `references` | 有序列表，每项 `type/view/index/latent`；index从0连续递增，latent是与source相同字段的对象；motion走视频VAE、同时间轴/画幅，图片只编码1帧 |
| `conditions` | 完整英文prompt经Gemma和feature extractor得到的 `video_prompt_embeds` 与 `prompt_attention_mask`；是connector之前的特征，不能用任意文本embedding代替 |

V7图片保留原尺寸，仅用边缘像素补齐32倍数，不统一缩成SOURCE尺寸；图片latent空间大小可不同。动作不是pose，而是完整motion视频latent。图像 `type` 支持identity/hair/garment/scene/outfit/camera，视频为motion；前后视图必须按原front/back配对顺序，不能排序或随意换slot；official garment有单图例外。SOURCE不放进可选references列表。

最终位置与token布局为 **有序参考 → SOURCE → 待生成区域**，各段沿原生时间坐标顺序拼接。待生成区域按SOURCE形状创建噪声，不来自GT；`generate_latent`没有target video参数。SOURCE与参考latent在去噪中保持固定，最后只提取生成区。基座也使用零效果的新PEFT adapter以匹配计算路径；这是一套同协议基座，不代表官方最佳多参考编辑流程。

更换输入时需要重新准备：

- SOURCE/motion：按12fps时间轴和同一画幅读取，归一化RGB至[-1,1]，转 `[1,3,T,H,W]` BF16，经 `ltx_trainer.model_loader.load_video_vae_encoder` 编码，保存上述latent与元数据。不要读GT；新视频不能冒用旧latent。
- 图片：`native_image_cache_contract.prepare_image(image)` 做原尺寸边缘补齐，再用相同VAE编码单帧；更新type/view/index。
- prompt：`load_text_encoder` → `encoder.encode([prompt])[0]` → `processor.feature_extractor(hidden, mask, 'left')`。输入conditions用去掉batch维的features/mask；缓存时使用 `pack_text_features`，读取用 `unpack_text_features`，同时重新生成匹配的text contract。不要把旧prompt的缓存继续用于新指令。

源码细节见 `encode_native_dataset.py` 的 `encode_media(job,args,identity,vae)`、`encode_text(job,identity,encoder,processor,verify_connector=False)` 和 `native_image_cache_contract.prepare_image`。前者是基础编码实现，图片预处理需采用V7的原尺寸策略；不能把其通用contain_image分支直接当V7图片策略。新输入准备在自己的目录完成，不运行历史全量缓存入口，也不篡改冻结索引。

### 怎样运行

保存以下代码为自己的 `ltx_one.py`，先复用已验证的identity缓存。读取器只从索引恢复source、references和text，不加载GT。代码中的OUT可以替换；`MODEL=ltx_base` 为基座，下一节说明A849的唯一加载差异。

```python
import gc, json, os, sys
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION') == 'testqueue'
assert os.environ.get('SLURM_JOB_ID')
N = Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5')
T = Path('/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112')
sys.path.insert(0, str(N / 'scripts/common'))
from project_layout import add_script_paths
add_script_paths()
sys.path.insert(0, str(T / 'scripts'))
from native_cache import load_sample
import torch, yaml
from accelerate.utils import send_to_device
from native_cached_dataset import collate_single_sample
from native_reference_inference import load_inference_models, encode_negative_context, generate_latent
from encoding_helpers import weight_paths, sha_file
from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
from ltx_pipelines.utils.media_io import encode_video
assert torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
out = Path(os.environ['OUT'])
out.mkdir(parents=True, exist_ok=False)
name = os.environ.get('MODEL', 'ltx_base')
assert name in ('ltx_base', 'a849')
cfg = yaml.safe_load((N / 'configs/training/mixed720p_sequential_visual_dropout/train.yaml').read_text())
assert cfg['reference_protocol'] == 'reference_positions_v7_refs_source_target'
_, weights = weight_paths(N / 'runtime/models/native_model_revision.json',
                          N / 'runtime/verification/native_weights_verified.json')
ckpt = None
if name == 'a849':
    ckpt = N / 'experiments/training/mixed720p_sequential_visual_dropout/job_197731/checkpoints/lora_weights_step_00849.safetensors'
    receipt = json.loads(ckpt.with_suffix('.contract.json').read_text())
    expected = receipt['files'][ckpt.name]
    assert ckpt.stat().st_size == expected['bytes'] and sha_file(ckpt) == expected['sha256']
    assert receipt['identity']['config']['lora'] == cfg['lora']
    assert receipt['identity']['config']['reference_protocol'] == cfg['reference_protocol']
settings = json.loads((N / 'configs/inference/stronger_guidance/inference.json').read_text())['settings']
device = torch.device('cuda:0')
# 先编码negative并释放Gemma，再加载DiT，避免同时驻留。
negative = encode_negative_context(weights, settings['negative_prompt'], device=device).to(device)
model, processor, scale = load_inference_models(weights, cfg['lora'], checkpoint=ckpt,
    protocol=cfg['reference_protocol'], device=device, ref_cfg=1.)
decoder = VideoDecoder(str(weights['vae']), dtype=torch.bfloat16, device=device)
gc.collect()
torch.cuda.empty_cache()
item = next(json.loads(x) for x in (T / 'metadata/inputs.jsonl').read_text().splitlines()
            if json.loads(x)['id'] == 'identity_01')
sample = load_sample(item)
# 换成自己的输入时替换sample：只提供source/references/conditions，不给target。
batch = send_to_device(collate_single_sample([sample]), device)
assert 'video_latents' not in batch
latent = generate_latent(model, processor, source=batch['source'],
    references=batch['references'], conditions=batch['conditions'], scale_factors=scale,
    protocol=cfg['reference_protocol'], seed=42, steps=40, negative_context=negative,
    cfg_scale=3, stg_scale=1, guidance_rescale=.7, stg_blocks=(28,),
    guidance_batch_size=1, ref_cfg=1.)
with torch.inference_mode():
    frames = decoder(latent, generator=torch.Generator(device=device).manual_seed(42))
    encode_video(frames, fps=12, audio=None, output_path=str(out / 'identity_01.mp4'),
                 video_chunks_number=1, thread_count=2)
print(out / 'identity_01.mp4')
```

```bash
# 1×H200 testqueue节点；将脚本路径替换为自己的位置
N=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
export MODEL=ltx_base
export OUT="/scratch/prj0000000275/$USER/model_inference/${MODEL}_$(date +%Y%m%d_%H%M%S)"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
export LD_LIBRARY_PATH="$N/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/ltx_${USER}_${SLURM_JOB_ID}"
"$N/runtime/envs/trainer/bin/python" "$HOME/ltx_one.py"
```

多条使用时模型只加载一次，循环sample与解码保存即可。样本帧数来自source latent；不能仅改保存fps来得到另一时长。

### 已有脚本如何复用

`benchmark_full10s_20261112/scripts/native_worker.py --model ltx_base|a849 --stage smoke|second|all --shard 0..3 [--shards N] [--control]` 是常驻分片worker，默认shards=2；阶段含义与VACE一致。它绑定历史清单、缓存索引、日志、输出，没有通用输出目录参数；不能原样启动新任务。早期百样本版本还绑定job199052。`native_cache.py` 的 `load_sample(item)` 可只读复用已有输入，但给新媒体时要建立自己的sample。

另一个入口 `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/scripts/inference/run_fixed_case_inference.py` 接受 `--bundle`、重复的 `--uid`、`--config`、`--output`、可选 `--checkpoint` 和guidance参数，面向已有bundle中的缓存样本，不是任意视频命令行工具。其默认配置和CFG/STG不等于本节协议，不能省略相关参数照搬。`scripts/common/run_trainer_python.sh` 选择原生环境，但会把缓存指向共享runtime；本文独立示例直接使用原生bin/python并将Triton缓存指向作业临时目录。

## A849

### 用的是什么、文件在哪里

A849是**同一LTX2.5 dev基座＋step849 LoRA**，不是单独的完整基模。Python环境、V7源码、图片/motion/SOURCE编码、prompt特征、参考顺序、推理步数和guidance全部沿用上一节。

| 文件 | 完整位置 |
| --- | --- |
| step849 LoRA | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/experiments/training/mixed720p_sequential_visual_dropout/job_197731/checkpoints/lora_weights_step_00849.safetensors` |
| 身份、配置和哈希receipt | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/experiments/training/mixed720p_sequential_visual_dropout/job_197731/checkpoints/lora_weights_step_00849.contract.json` |
| LoRA配置 | `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/configs/training/mixed720p_sequential_visual_dropout/train.yaml` |

LoRA大小 **1,308,755,416 bytes**，SHA256 **`01ad84f3a3cd2877a55e23c08f86b2f17b6d4f23d3d67c2a806f918006b62eb3`**。rank64、alpha64、dropout0；target modules为 `attn1.to_q/to_k/to_v/to_out.0`、`attn2.to_q/to_k/to_v/to_out.0`、`ff.net.0.proj`、`ff.net.2`。协议必须为 `reference_positions_v7_refs_source_target`；加载器同时检查checkpoint元数据、key集合/shape及加载后tensor相等性。

加载差异只有 `load_inference_models(..., checkpoint=ckpt, ...)`：基座传None，A849传上述safetensors。LoRA保持FP32参数，以未融合PEFT＋BF16 autocast参与计算；不要先merge、量化或把整个模型转BF16来改变adapter精度。推理不需要training_state或optimizer文件。

### ref、运行与脚本复用

ref协议完全沿用上一节。运行同一个 `ltx_one.py`，只改环境变量并生成新的OUT：

```bash
# 沿用上一节已设置的N与环境；1×H200 testqueue
export MODEL=a849
export OUT="/scratch/prj0000000275/$USER/model_inference/${MODEL}_$(date +%Y%m%d_%H%M%S)"
"$N/runtime/envs/trainer/bin/python" "$HOME/ltx_one.py"
```

结果在 `$OUT/identity_01.mp4`。若使用历史 `native_worker.py`，模型选择参数为 `--model a849`，但其固定清单/输出限制仍然适用，不能直接当通用启动器。

## 使用前的只读检查与结果检查

login上可先执行以下检查，只解析路径、读取小型记录，不加载模型：

```bash
python - <<'PY'
import json, sys
from pathlib import Path
E = Path('/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation')
N = Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5')
sys.path.insert(0, str(N / 'scripts/common'))
from project_layout import resolve_path
record = json.loads((N / 'runtime/models/native_model_revision.json').read_text())
for f in record['files']:
    p = resolve_path(record['local_dir']) / f['path']
    assert p.is_file() and p.stat().st_size == f['bytes'], p
    print(p)
for p in [N / 'runtime/envs/trainer/bin/python',
          E / 'ltx25_vace_a100_20261111/runtime/vace/bin/python',
          E / 'minimax_h3_ref2va90_local_20261112/runtime/env/bin/python']:
    assert p.exists(), p
    print(p, '->', p.resolve())
T = E / 'benchmark_full10s_20261112'
entry = json.loads((T / 'metadata/native_cache_index.json').read_text())['identity_01']
for ref in [entry['source'], entry['text'], *entry['references']]:
    assert Path(ref['path']).is_file(), ref['path']
print('identity_01缓存文件存在；不等同于GPU推理已执行')
PY
```

运行后在自己的OUT中检查MP4能完整解码、画幅/帧数正确：LTX/VACE示例121帧、12fps；H3示例243帧、24fps。可用各自环境中的PyAV遍历decode统计帧数，在login完成。文件存在、完整解码和协议核验只说明技术步骤完成，编辑效果仍需观看视频确认。
