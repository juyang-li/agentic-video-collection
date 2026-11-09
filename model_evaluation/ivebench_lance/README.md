# Lance IVEBench 分阶段评测

本轮已经验证的协议、踩坑、结果解释与复用建议见 [`IVEBENCH_EVALUATION_LESSONS.md`](./IVEBENCH_EVALUATION_LESSONS.md)。

当前执行计划分三阶段：IVEBench `Short / subject_editing` 全量 91 条、`Long / subject_editing` 全量 43 条、`Short+Long / attribute_editing` 全量 94 条。每个阶段分别运行官方 Lance 3B Base 与从该基模训练的 4K LoRA step 677，所有生成 seed 固定为 42；不使用师兄的微调权重作为起点。

官方数据固定为 `Coraxor/IVEBench@1e6cb655c0a537f6093edcf18e2607fb03e76734`，官方评测代码固定为 `RyanChenYN/IVEBench@2dcf8900ee65197f43854c76f61c66a32bb8dbfc`。91 条由官方 Short prompt JSON 的 `category == subject_editing` 机械筛选得到：添加 28、删除 33、替换 30，不做抽样。

每阶段执行顺序为：下载并按官方 LFS SHA256 校验输入；运行一对 Base/LoRA smoke；四卡生成 Base；四卡生成 LoRA；完整解码与严格配对校验；本地指标与 Gemini 3.8 Flash 评分；构建滚动审核包并上传百度网盘。正式 API judge 只从 `OPEN_ROUTER_QWEN` 读取密钥，固定 Google AI Studio、low reasoning、`temperature=0`、禁止 fallback，并建立与旧 OmniEdit Vertex 结果不混用的新协议。

评测包含官方实现的 9 个适用本地指标；官方 Qwen2.5-VL-72B 的 Instruction Satisfaction 与 Content Fidelity 改为 Gemini API。Subject/Attribute 子集不存在数量编辑，Quantity Accuracy 结构性记作 N/A，不伪造为 0，并从 compliance 分母排除。除两组绝对 IS/CF 外，每条还做 A/B 与 B/A 顺序对调的双盲比较，并固定抽取 9 条 × 2 模型复评，量化 `temperature=0` 下的残余漂移。

聚合严格使用 IVEBench 官方归一化和权重：VTSS、Instruction Satisfaction、Content Fidelity 的权重分别为 4、3、3，其余适用指标为 1；Quality、Compliance、Fidelity 三个维度最终等权。类别子集分数只用于 Base/LoRA 配对比较，不冒充完整 600 条榜单分数。

API 媒体最多取 20 个严格配对时间点、最长边 426、5 fps；本地指标使用全部生成帧。源视频都按 Lance 实际采样索引对齐，两个模型的输出帧数、尺寸和 12 fps 必须逐条相同。API 首先各跑一条绝对评分和 pairwise 预检，读取 OpenRouter 返回的真实费用与实时余额，留出 25% 安全余量后才展开全量。

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/prepare_short_subject.py --download
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/run_generation.py \
  --jobs artifacts/short_subject/generation_jobs.json
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/orchestrate_short_subject.py
```

`orchestrate_short_subject.py` 可在生成期间提前启动；它会等待 `generation_status.json` 完成，然后无人值守地完成媒体准备、API 与四卡本地评测、聚合、离线三列滚动审核页、真实 Chromium 播放验收、ZIP/哈希、百度网盘上传和远端目录核验。阶段状态见 `artifacts/short_subject/orchestration_status.json`，详细日志见 `artifacts/short_subject/logs/orchestrator_*.log`。

审核页的 H.264 播放验收优先使用项目内 `runtime/google-chrome-stable/opt/google/chrome/chrome`，并通过 `runtime/debs/extracted/usr/lib/x86_64-linux-gnu` 补充运行库。视频编码信息优先由 `ffprobe` 读取；环境没有 `ffprobe` 时，审核包构建器会使用随 `imageio-ffmpeg` 提供的 ffmpeg 输入探测回退。

后续既定顺序是 Long / Subject Editing 43 条，然后 Short+Long / Attribute Editing 94 条；前一阶段审核包上传完成后才启动下一阶段生成。Long 输入先从完整官方视频时间轴均匀取 121 帧，编码为 12 fps 的确定性模型输入；Lance 的 `max_duration=10.1`，因此模型实际源输入和输出均严格为 121 帧。Short 输入仍按原时长和 Lance 的 `4k+1` 约束采样。

后两阶段使用同一入口并显式指定阶段：

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/prepare_next_stage.py long_subject
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/run_generation.py \
  --jobs artifacts/long_subject/generation_jobs.json --master-port 29940
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/orchestrate_short_subject.py --stage long_subject

/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/prepare_next_stage.py attribute_all
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/run_generation.py \
  --jobs artifacts/attribute_all/generation_jobs.json --master-port 30040
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/orchestrate_short_subject.py --stage attribute_all
```

长视频模型输入可在 GPU 生成前用空闲 CPU 并行预填。该命令覆盖两个后续阶段共 77 个 Long 源视频，使用独立临时文件并原子替换，仍执行 121 帧、12 fps、全帧解码和 SHA256 校验：

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python \
  scripts/prefill_long_model_inputs.py --workers 4
```

跨 Slurm 墙钟时间的后继入口为 `scripts/resume_remaining_stages.sbatch`。它不依赖上一作业遗留的 queue 状态，而是重新校验准备清单，并根据 `generation_status.json`、输出视频校验和 API receipt 跳过已完成工作或按样本续跑：

```bash
sbatch --dependency=afterany:<current_job_id> scripts/resume_remaining_stages.sbatch
```

2026-09-07 的实际调度为当前四卡作业 `124267`，以及依赖 `afterany:124267` 的后继四卡作业 `124505`。Short Subject 已完整评分、浏览器验收并上传；其审核包及远端回执位于 `artifacts/short_subject/review/`。

跨阶段协议回归测试：

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python -m unittest -v scripts/test_evaluation_contract.py
```
