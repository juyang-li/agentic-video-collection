# Lance 原始模型最小评测报告

> 日期：2026-08-24  
> 模型：`Lance_3B_Video`  
> 代码：官方 commit `9717d9160ed16fedfdad5ce39ccbc6c9d69695e0`  
> 状态：部署、生成、技术检查和观察包已完成；完整视频人工评分待填写

## 先看这里

- 直接观察：[`comparison/`](./comparison/) 中有 15 个“左源视频 / 右 Lance 输出”的 MP4。
- 下载观察：[`lance_base_comparison_15_h264.zip`](./lance_base_comparison_15_h264.zip)，约 21 MiB。
- 填写评分：[`scores.csv`](./scores.csv)。
- 查看冻结样本：[`benchmark_15.csv`](./benchmark_15.csv)。

comparison MP4 统一为 H.264、`yuv420p`、fast-start、1920×540、12 FPS。每侧保持原始长宽比并补黑边；源视频按完整相对进度对齐输出时长，方便观察对应动作，而不是只截取源视频开头。它只是人工观察副本，原始 source 和 output 均保留不变。

## 执行结果

- 使用独立干净 checkout，未修改 DataMake 中已有的多轮实验代码。
- 复用现有约 18 GB 权重和 `xks_lance` 环境，没有重复下载模型。
- 在 SLURM job `115576` 的单张 H200 上完成 smoke 和 benchmark。
- Smoke：12 steps，1/1 生成成功并完整解码。
- Benchmark：从 OmniEdit-Bench Spatial 以 seed 42 分层随机冻结 15 条；30 steps、seed 42、CFG 4.0。
- 生成循环耗时：2 条 gate 5 分 13 秒，剩余 13 条 32 分 37 秒，共 37 分 50 秒。
- 15/15 输出完整解码通过；14 条为 69 帧，1 条因 source 较短为 45 帧，均为 12 FPS。
- 15/15 comparison 完整解码通过；ZIP 共 35 个成员，完整性检查通过。

第一次 smoke 配置使用了非数字样本 ID，触发原始 loader 报错；改成数字 ID 后正常运行。另发现上游 `inference_lance.sh` 在 Python 子进程失败后仍可能打印 `Done`。本项目 wrapper 已增加“必须出现预期 MP4 且完整解码”的外部硬检查，没有修改上游模型代码。

## 固定评测集

| 类别 | 数量 |
| --- | ---: |
| object adding | 3 |
| object removal | 3 |
| object replacement | 3 |
| appearance change | 3 |
| clothes change | 3 |

完整 ID、指令、官方路径、revision 和 source SHA256 见 [`benchmark_15.csv`](./benchmark_15.csv)。15 条在模型生成前已经冻结，没有按结果换样本。

## 初步质量观察

下表来自稀疏帧检查，只用于快速发现模式，不能替代完整 MP4 的时序评分。

| 类别 | 初步结果 | 主要现象 |
| --- | --- | --- |
| adding | 3 条明显完成 | 人物、蜡烛、狗均被加入，原场景整体保持较好 |
| removal | 3 条均未明显完成 | 透明罩、人物、玻璃杯仍可见，是最明确的失败类 |
| replacement | 2 条明显完成，1 条部分完成 | 天鹅→灰鸭、台灯→发光球较好；猕猴桃→橙子有语义变化，但形状和位置跨帧不稳 |
| appearance | 3 条均有明显问题 | “变老/换脸”容易变成新增另一个人，或只产生很弱的外观变化 |
| clothes | 2 条明显完成，1 条部分完成 | 白色礼服和蓝色牛仔夹克较清楚；黄色连衣裙仍残留原红色袖子 |

当前初步判断是：原始 Lance 在大尺度 add、replace 和 clothes 上较强，remove 和人物年龄/换脸类细节编辑较弱。必须在观看完整 MP4、确认闪烁、动作漂移和身份保持后，才能把它写成正式定量结论。

## 人工评分

逐条打开 [`comparison/`](./comparison/) 中同名 MP4 和 TXT，在 [`scores.csv`](./scores.csv) 填写四项 0–5 分：

```text
overall = 0.35 * accuracy
        + 0.25 * preservation
        + 0.25 * temporal_consistency
        + 0.15 * visual_quality
```

评分表已有 15 行技术通过记录，人工分数目前保持为空；生成脚本不会覆盖已有评分。填写后再汇总总体和分任务 baseline。

下载包信息：

```text
文件：lance_base_comparison_15_h264.zip
大小：21,829,072 bytes（约 21 MiB）
成员：35
SHA256：3c73e6f051bb7e0bcf0ae1f3a83a60079cfdb3ca3205d255788799d6f8761603
```

## 目录与复现

```text
lance_base/
├── README.md                       # 本报告
├── comparison/                     # 15 组左右对比 MP4、指令和清单
├── lance_base_comparison_15_h264.zip
├── benchmark_15.csv                # 冻结样本
├── technical_results.csv           # 自动技术检查
├── scores.csv                      # 人工评分入口
├── configs/                        # smoke / benchmark 推理配置
├── scripts/                        # 准备、运行、校验和打包脚本
├── data/                           # 官方 metadata、source 和 smoke 输入
├── outputs_raw/                    # 模型原始输出
├── logs/                           # 原始运行日志
└── runtime/upstream/               # 固定 commit 的 Lance 代码
```

正式评测入口：

```bash
bash scripts/run_benchmark.sh configs/benchmark_all.json benchmark_all
```

生成左右 comparison：

```bash
/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python scripts/build_comparison.py --rebuild
```

关键记录：[`run.json`](./run.json)、[`technical_results.csv`](./technical_results.csv)、[`comparison/inventory.csv`](./comparison/inventory.csv)。
