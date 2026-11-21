# 300 对冻结测评集 · 跨服务器交接

本包为四模型逐对对照的固定输入。300 对：UE 合成 200（8 类各 25）、CoinVE 75、TripVVT 25。旧 100 对的 ID、完整 prompt、SOURCE、GT 与有序参考身份保留；新增 120/65/15 对只取当前正式 test。抽样 seed=20261113，不读取模型输出挑样，不加入外部 96 条任务。独立 SOURCE 数量和横竖幅/人物统计见 `sampling.json`，pair 数量不等于独立视频数量。

师兄服务器运行 H3、VACE、LTX2.5 基座；A849 step849 在本机使用这份冻结清单重新推理。本包不向任何服务器自动提交任务，也不包含模型权重、环境或推理脚本。旧输出不计入新批次完成数。技术验收不代表人工质量通过，本包没有质量评分。

## 离线 HTML 审核

解压后直接用浏览器打开 `index.html`。审核页按人物展示30对（8个合成人物各3对，另有CoinVE/TripVVT各3对），按任务连续展开；全部300对原始媒体及清单仍完整包含在本交接包中，顶部固定任务导航；可按样本ID、uid或英文指令筛选。每条展示完整prompt、SOURCE、有序图片/动作参考、GT，TripVVT完整官方caption单独展示。同步播放和共同时间滑条关联同一条的SOURCE/动作参考/GT。`review_media/`为共同区间的12fps H.264小尺寸浏览副本和首帧海报，`review_manifest.json`记录其原件映射和完整解码证据；每个区域保留完整原件入口。浏览副本不替代模型输入，不提供评分或人工通过标记。

## 文件与读取接口

- `inputs.jsonl`：一行一对，稳定 `id`/`uid`、类别、完整双语 `prompt`、`source`、按 `index` 升序的 `references`、`canvas_wh`、`fps`、`num_frames`、时间戳、原始解码帧索引、冻结时间。CoinVE 另保存全部 `ordered_instructions`。
- `ground_truth.jsonl`：按同一 `id` 关联 GT、审核 caption 和 GT 采样帧索引，只用于评测。TripVVT caption 与换衣指令分开保存。
- `media/`：完整原始视频/图片，按内容哈希去重。SOURCE/参考与 GT 均是实际文件，不依赖原服务器路径，不把浏览转码当作原件。
- `media_inventory.jsonl`：包内路径、原始规格、来源相对路径、字节数、SHA256、完整解码结果。`origin.dataset_path` 仅溯源，不用于加载。
- `sampling.json`、`validation.json`：抽样及验收；`model_interfaces.json`：准确模型版本、所需文件、代码位置/哈希与实际接口；`result_schema.json`：结果回收字段。

以解压目录为根，读取 `root / item['source']['path']`；有序参考读取 `root / ref['path']`。所有媒体加载路径都是相对路径。推理进程只加载 inputs 及其引用；不要扫描整个 media 目录、读取 GT 表、target caption 或从 GT 制作条件。镜头/场景清单中已冻结的官方参考首帧是合法输入，不再另读 GT 获取任何额外帧。

## 统一时间轴与分辨率

合成使用 t=0 到 t=10 秒（含端点），12fps、121 帧。媒体文件保存完整原件，推理取 `sample_frame_indices` 指定的解码显示帧；SOURCE 与动作参考使用同一组绝对时间戳。真实数据从 SOURCE/GT/动作参考共同有效区间起点开始，上限 10 秒，按 `8*floor(span*12/8)+1` 向下选取合法帧数，逐条以 `temporal_window` 为准。真实数据使用最后一个已解码 PTS 判定有效端点，不用容器 duration 推测可用尾帧。

索引采用与目标时刻最近的原生 PTS，平局取后帧（1e-9秒数值容差，与原生协议一致）；不变速、不补长尾。合成原件部分为30fps、300帧，显示覆盖10秒而最后PTS为9.9667秒；沿用既有原生121帧协议，仅t=10端点映射至最后一帧，最多一个原生帧间隔，并在每行 synthetic_endpoint_policy 标明。12fps 编码播放时长是 N/12，评测内容区间是 (N−1)/12，二者须区别。短片不再因不足10秒而全部暂缓，按各自行的实际合法区间执行；SOURCE、GT、动作参考不可各自按百分比重采样。

LTX/A849/VACE 画布为 1280×736 或 736×1280，等比例 contain、黑边居中；按清单横竖幅选择，不拉伸。LTX 图片参考保留原始尺寸，按 V7 最少 edge padding 到 32 倍数，不拼成九宫格。H3 按原生短边768，16:9为1344×768、9:16为768×1344。H3 用同一绝对时间轴将12fps条件映射到24fps（既有10秒适配为 `min(120,(j+1)//2)`），只在共同有效区间之后保持端点以满足原生对齐；原生帧数必须实际回报，10秒已测为243帧。对较短真实视频不得硬套243帧或延长到10秒。审核只比较冻结区间，H3对齐尾部单独标注，不参与评分。

## 固定模型下载与代码

下载命令在对方服务器执行。全部固定 revision，勿用 main 替代。逐文件清单见 `model_interfaces.json`。

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 0919fb684102ef115126612f3fbb84d1027e7317 --include 'Ref2VA/*' model_index.json --local-dir models/H3
hf download Wan-AI/Wan2.1-VACE-14B-diffusers --revision db79b90c60bbb45ceec9e41b9d5a4df934538ac4 --local-dir models/VACE
hf download Lightricks/LTX-2.5 --revision 49b093a39c19169a8a81721dcc541e07e945232f diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors vae/ltx-2.5-video-vae-conv-bf16.safetensors --local-dir models/LTX-2.5
```

H3 需要 Ref2VA 的 transformer 全13个分片及 index、text_encoder 全14个分片及 index、processor/tokenizer、video_vae（包括 source/model.safetensors）、audio_vae 及对应配置/模块。VACE 下载整份 diffusers snapshot，包括 transformer、text_encoder、tokenizer、vae、scheduler、model_index。LTX 本次只做视频，使用上述三个 dev 组件，无 upscaler 或训练 LoRA。

官方入口：[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[Wan VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)、[LTX2.5](https://huggingface.co/Lightricks/LTX-2.5)。以下参数来自本地已验证的测评接口，不将模型卡通用示例误当成此任务接口。

### MiniMax H3

本地 H3-Base Ref2VA，无 Context-IR，不调用付费API。上游 SGLang commit `fa179312e94c764f8637724ea2a364d302888906`（https://github.com/sgl-project/sglang）。本地 portability patch 和 requirements 的位置、SHA256 在 model_interfaces 中；它们不随本包分发，复用时须取得对应文件，不能把当前上游 HEAD 当成已验证环境。

启动接口为 `sglang serve --model-path models/H3/Ref2VA --served-model-name MiniMaxAI/MiniMax-H3 --num-gpus 4 --tp-size 2 --ulysses-degree 2 --encoder-parallel auto --performance-mode speed --enable-torch-compile false --warmup-mode off --batching-max-size 1`。实际调用 `POST /v1/videos`，以返回 id 轮询 `GET /v1/videos/{id}`，下载 `/content`。不是 diffusers 文生视频示例。

请求：`model=MiniMaxAI/MiniMax-H3, task=ref2va, num_inference_steps=50, seed=42, flow_shift=12, audio_flow_shift=3, quality=lossless, reference_resize_mode=max, num_outputs_per_prompt=1`；target 提供 short_edge=768、清单横竖比、实际冻结区间 duration_seconds。50 指实际原生更新次数，需记录服务端执行证据。DiT 使用原生 BF16/FP32，VisualVAE 按原生 FP16 autocast 解码；禁用量化、compile、加速 LoRA 和跳步缓存。

conditions 第一项是完整冻结区间 SOURCE video，随后按清单顺序加入 image/motion video，均 `role=reference`，视频 start_time_seconds=0。URI 在对方服务器由包内文件构造，不直接传本机绝对路径。prompt 使用 subject_definitions、summary、retention_analysis、detailed_description、overall_soundscape、non_diegetic_music 结构；定义 Video 1 为 SOURCE、Subject 1 为目标人物，逐个编号 Picture/Video 与参考对应。detailed_description 原样插入完整 `prompt.en`；保留 CoinVE 子指令顺序，描述完整冻结时间轴和保持项。声音不参与本次视觉测评。CoinVE references 为空时不添加 Picture 或虚构条件。动作参考为合法有序视频，禁止用 GT 代替。

### Wan2.1-VACE-14B

实际使用 diffusers **0.35.2** `WanVACEPipeline`（torch 2.10.0+cu126、transformers 4.57.1），pipeline 源文件 SHA256 及完整调用签名见 model_interfaces。代码版本以官方发布 tag v0.35.2 和记录的实际源文件为准，不使用当前 HEAD。`AutoencoderKLWan` FP32，transformer BF16；`UniPCMultistepScheduler.from_config(..., flow_shift=5)`。

调用参数为 `prompt=adapted_prompt, video=prepared_source_or_pose, mask=white_editable_mask, reference_images=ordered_images_or_None, height=H, width=W, num_frames=N, num_inference_steps=40, guidance_scale=5, conditioning_scale=1, generator=seed42`。固定 negative_prompt 为 `blurry, excessive noise, compression artifacts, flickering, distorted proportions, deformed facial features, extra limbs, disfigured hands, unnatural transitions`，max_sequence_length=512、output_type=pil。原英文编辑指令完整保留，追加所需重绘/参考语义并另存 adapted_prompt；不得删除复合编辑。reference_images 按清单 index 顺序传入，不手动反转（pipeline 内部 latent prepend 的逆序与参数顺序分别记录）。无参考 CoinVE 使用 None。

mask 白255表示可编辑，黑0表示保持。重绘区域在 RGB 输入上填128，再传官方 pipeline；只传原RGB＋mask与本协议不同。局部 mask 从 SOURCE 分割、跟踪或合法 SOURCE ROI 生成；动作 pose 从合法动作参考按同一时间轴生成，SOURCE 首/中/末帧提供外观；镜头任务沿用原 SOURCE camera control，不做局部置灰。所有条件记录来源和 SHA256，禁止读 GT 制作 mask/pose、选择区域或修正结果。CoinVE 多编辑需覆盖全部要求，不能只执行第一条。困难样本可记录失败原因，禁止按输出好坏替换。

已验证的辅助模型 revision：Grounding DINO `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`、VACE-Annotators pose `63512cbbe4e3a98e1636356ef2230da84649f617`、segformer_b2_clothes `584abc1e1d260e23c0fc627c5217a09b2b461046`；仓库名、本地 SAM2 checkpoint/pose 代码位置见 model_interfaces。需要条件生成时下载对应 snapshot；SAM2 是额外本地依赖，本包不携带。

### LTX2.5 基座＋DataMake V7 与 A849

两者共享官方 dev 权重、SOURCE/有序参考/文本、采样时刻、V7布局。基座是 **A849 同协议未微调对照**，不代表官方开箱即用的通用参考编辑能力。基座 `checkpoint=None`；A849只额外加载指定 step849 checkpoint，身份和 SHA256 见 model_interfaces。禁止将其他 LoRA 或 distilled transformer 混入基座。

官方 LTX-2 代码基线 `e98f6d927d83c92ef2be6aec76c8440b19306e39`（https://github.com/Lightricks/LTX-2）。重要限制：必要 DataMake V7 模块当前未纳入 Git。model_interfaces 的 `v7.modules` 列出现存文件位置、SHA256、tracked_in_git 状态；这些模块、配置和本地环境不会随数据包交付。仅 clone 官方仓库不能复现此对照，应从师兄现存工作区取得匹配字节，再按接口适配包内路径，不能声称 Git commit 已包含全部实现。

接口：`native_reference_inference.load_inference_models(weights, lora_config, checkpoint=..., protocol='reference_positions_v7_refs_source_target', device=..., ref_cfg=1)`，`build_generation_state(source, references, ...)` 和 `generate_latent(..., conditions, steps=40, seed=42, cfg_scale=3, stg_scale=1, stg_blocks=(28,), guidance_rescale=.7, ref_cfg=1)`；用官方 `VideoDecoder` BF16 视频解码，12fps 输出。prompt.en 不裁剪/摘要，不用 target caption；负向文本和 lora 配置从记录的固定 inference/training 配置读取。

SOURCE/motion 在相同画布和12fps上编码，N=8n+1对应 n+1 个 latent 时间帧；图片单独编码为一个 latent帧，保留自身空间尺寸并最少 edge pad 到32倍数。references 使用连续 index；motion 无 image slot，图片 slot 等于它在原参考表中的 index。front/back 次序保持，TripVVT official garment 恰好一张；CoinVE允许空表，不造占位图。

V7 的物理 token 顺序与位置时钟均为 **有序 optional references → SOURCE → target**。先使用官方 causal-corrected `[B,3,N,2]` FP32 坐标，局部时间起于0；图片局部区间为[0,1/12]，各段依次累积原生跨度 `((latent_frames−1)*8+1)/12` 秒，不采用旧 V1 图片槽的12秒起始偏移。空间坐标保留原生 extent；2048是RoPE尺度，不是硬裁切上界。target 最后且只有 target 去噪，references/SOURCE是干净条件，输出只抽取 target；不能把 target首帧或GT latent作为条件。V7 rotate、denoise mask、position和token必须一致变换，不能只交换 latent 数组。

## 已验证硬件与实际限制

对方服务器硬件未知，以下仅为本地 H200 记录，不承诺其他显卡容量或吞吐。H3 为4×H200、TP2＋Ulysses2协同一条、单请求并发，10秒首测原生峰值约76.8GiB/卡。LTX/A849/VACE为每卡独立进程、4卡分片。121帧 A849 已测 PyTorch 峰值约48.25GiB；VACE已完成子集峰值最大约69.23GiB（26条历史子集），原始字节数见 model_interfaces 的实测补充（统计口径为PyTorch分配，非整卡占用）。模型加载、条件准备、排队、参考数量和长尾会影响端到端时间，不把历史4秒吞吐套到本批。

## 结果回收与验收

每个模型重新运行全部300对；输出为 `results/<model>/<id>.mp4` 和同名 JSON，汇总 `results.jsonl`。id 必须完全匹配本包。输入版本采用 inputs.jsonl 的 SHA256（见 sampling.json），模型版本同时记录权重revision、代码commit/hash和LoRA身份。

每条记录按 result_schema 保存 seed、全部参数、原始/adapted prompt、有序输入/条件哈希、原生帧数/fps/尺寸、冻结比较区间、H3尾端区间、耗时、峰值显存及口径、GPU/并行方式、成功或失败原因；失败保留一行，未运行不能写complete。先核对模型协议与可完整解码，再由人工判断编辑完成、内容保持、时序、画质；不新增自动质量分。按稳定ID与本机A849逐对合并，禁止按输出文件序号配对。

validation 检查300条及配额、旧100身份、媒体SHA/完整解码、参考完整、有序指令、合法区间、相对路径、推理无GT/caption。冻结后的媒体或清单如需改变，必须新版本另建，不覆盖此包。包上传至 `/apps/bypy/lance_review/`，只上传ZIP，以上传器成功回执为准，不另做远端大小/哈希核验或整包回读。现有H3/VACE任务及续跑监控保留，独立运行。
