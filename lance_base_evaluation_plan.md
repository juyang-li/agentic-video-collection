# Lance 原始模型部署与最小评测计划

> 日期：2026-08-24  
> 状态：部署、15 条生成与技术检查已完成；人工视频评分待审阅  
> 范围：只验证 Lance 原始 `video_edit`，不训练、不做多轮、不搭服务、不跑完整 benchmark

## 1. 当前结论

Lance 不需要从零下载和安装。三个搜索结果分别承担不同角色：

| 路径 | 作用 | 是否是执行入口 |
| --- | --- | --- |
| `/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/` | 实验工作区，包含代码、权重入口、输入、脚本、日志和输出 | **是** |
| `/scratch/prj0000000275/jcy/.cache/huggingface/hub/models--bytedance-research--Lance/` | Hugging Face 缓存记录；目前基本只有 `refs/main`，不是完整权重 | 否 |
| `/scratch/prj0000000275/jcy/.conda/envs/xks_lance/` | Python/CUDA 运行环境 | 否 |

真正的模型组件是：

```text
DataMake/10_research/model_experiments/lance/
├── lance/                         # Lance 代码仓库
│   └── downloads -> ../models/Lance
├── models/Lance/                  # 约 18 GB 的实际权重
│   ├── Lance_3B_Video/
│   ├── Qwen2.5-VL-ViT/
│   └── Wan2.2_VAE.pth
├── inputs/
├── outputs/
└── scripts/run_video_edit.sh
```

现有环境已经验证为 Python 3.11.15、PyTorch 2.8.0+cu126、CUDA 12.6、flash-attn 2.8.3。已有 smoke 输出也能正常解码：45 帧、12 FPS、560×752。这说明“部署链路曾经跑通”。

但 `lance/lance/` 当前有多轮实验产生的本地修改和未跟踪文件。虽然原始单轮路径大体仍在，严格的 base baseline 不应直接依赖这个脏目录。

## 2. 本轮目标

本轮只回答两个问题：

1. 固定的原始 Lance 代码和现有权重能否在 H200 上稳定完成单轮视频编辑？
2. 在一个很小但固定的 add/remove/replace 与人物外观编辑集合上，原始模型的质量如何？

这里的“部署”只表示批量推理可运行，不启动 Gradio，也不配置长期在线服务。

## 3. 第一步：冻结干净的原始模型

不修改或清理 DataMake 中现有代码。本项目下已经建立独立的干净 checkout，并在评测后整理为：

```text
video_collection/model_evaluation/lance_base/
├── README.md             # 显眼的评测报告入口
├── comparison/           # 15 组左右并排 H.264 MP4 与指令
├── configs/
├── scripts/
├── data/
├── runtime/upstream/     # 固定到 Lance commit 9717d9160ed16fedfdad5ce39ccbc6c9d69695e0
├── benchmark_15.csv
├── outputs_raw/
├── scores.csv
├── run.json
└── lance_base_comparison_15_h264.zip
```

`upstream/downloads` 只链接到已有的 `models/Lance/`，继续使用已有 `xks_lance` 环境，不重新下载权重、不新建 Conda 环境。

在 `run.json` 中记录：代码 commit、HF 权重 revision、环境版本、GPU、随机种子、推理参数、输入和输出路径。

### Smoke test

- GPU：1 张 H200；官方要求至少 40 GB 显存。
- 任务：`video_edit`。
- 输入：现有可解码视频中的 1 条。
- 指令：只做一个明显、容易判断的原子编辑。
- 快速参数：12 steps、seed 42、CFG 4.0、`video_480p`、prompt enhancement 关闭。
- 验收：无 OOM/依赖/解码错误，生成 MP4 能完整解码，指令和运行参数被保存。

Smoke 不通过就停止，不开始 benchmark。若干净代码无法运行而当前修改版可以运行，应先查明必要 patch；在原因确认前不能把修改版结果称为原始 Lance。

## 4. 第二步：最小 benchmark

使用 [OmniEdit-Bench](https://huggingface.co/datasets/OmniEdit-Bench/OmniEdit-Bench) 的 Spatial 部分，但第一轮只冻结 **15 条内部开发切片**，不跑完整 240 条，也不把结果称为官方总分。

从以下五类各分层随机抽 3 条，随机种子固定为 42：

- object adding；
- object removal；
- object replacement；
- appearance change；
- clothes change。

抽样只根据官方 metadata 和 source video 完成，在生成结果前冻结 `video_id`，不得根据模型输出好坏换样本。这样既覆盖当前最关心的 add/remove/replace，也保留人物外观和服饰编辑。

正式 baseline 参数固定为：30 steps、seed 42、CFG 4.0、`video_480p`、50 frames、prompt enhancement 关闭。先跑 2 条作为输入规格门禁，通过后再跑剩余 13 条。原始视频不覆盖；若必须转码，只创建可追溯的派生输入，并在 manifest 中记录转换参数。

## 5. 用什么 evaluate

第一轮采用“自动技术检查 + 人工视频评分”，不部署额外本地大模型，也不调用 GPT/Gemini API。

### 自动技术检查

逐条检查 source/output 是否可完整解码，并记录帧数、FPS、宽高、时长和空文件。技术失败直接标记，不悄悄更换样本。

### 人工视频评分

每个案例同时观看 source MP4、instruction 和 output MP4，四项各打 0–5 分：

| 指标 | 权重 | 判断内容 |
| --- | ---: | --- |
| 编辑准确性 | 35% | 指令是否真正完成，位置、对象和属性是否正确 |
| 非编辑内容保持 | 25% | 主体身份、动作、背景和构图是否被无关改写 |
| 时序一致性 | 25% | 是否闪烁、漂移、突然消失或跨帧形变 |
| 视觉质量 | 15% | 清晰度、真实感、边缘、纹理和局部伪影 |

```text
overall = 0.35 × accuracy
        + 0.25 × preservation
        + 0.25 × temporal_consistency
        + 0.15 × visual_quality
```

保持、时序和视觉质量合计占 65%，避免只看“改没改到”而忽略视频本身是否可用。除均分外，还报告每类 3 条的明细和失败原因；15 条样本太少，不做显著性结论。

不使用 PSNR、SSIM 或 LPIPS 作为主指标：该 benchmark 没有唯一目标视频，而 source 与成功编辑结果本来就应该不同。也不先用 LLM/VLM 裁判：15 条人工观看更直接，尤其能看到闪烁和动作漂移。以后扩到完整 Spatial 240 时，再考虑接官方 Gemini evaluator。

## 6. 方便观察的交付形式

保留模型原始输出，同时生成可直接观察和下载的 comparison。每条案例包含一个左右并排 MP4 和同名指令：

```text
01-Sp_Object_adding_01.mp4   # 左 source，右 Lance output
01-Sp_Object_adding_01.txt   # video_id、类别和指令
```

comparison MP4 统一转为 H.264 + `yuv420p` + fast-start，保证 VS Code 和本地播放器容易播放；不生成 GIF。源视频按完整相对进度对齐输出时长。`scores.csv` 初始为空，由人工审核填写，脚本不得覆盖已有分数。

## 7. 完成条件

本轮完成只需要以下产物：

- 干净 commit 的单样本 smoke 成功；
- 冻结的 15 条 benchmark manifest；
- 15 个原始模型输出和技术检查结果；
- 一个方便下载查看的 MP4 review ZIP；
- 一页简短报告，汇总四项分数、各任务表现和典型失败。

完成以上内容后再决定是否开始 CoinVE 数据整理和微调。本轮不顺带实现数据筛选、训练格式、API 评审或完整 benchmark。
