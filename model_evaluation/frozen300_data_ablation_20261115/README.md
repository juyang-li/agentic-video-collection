# Frozen300：纯真实／纯合成训练数据消融

比较师兄提供的 `real232`、`synthetic600` 两组各300条输出；只评分，不重跑推理。主表加入已经完成且协议一致的无LoRA蒸馏8步基线；混合A849 dev40步仅作单独参考。232、600是checkpoint步数，两组训练步数也不同，不能把差异单独归因为数据来源。

SSIM、PSNR、LPIPS只与冻结GT比较，沿用[正式GT协议](../SSIM_PSNR_LPIPS_README.md)的既有数值核与环境。VBench使用固定commit和权重，计算主体一致性、背景一致性、美学、成像、原始闪烁、运动平滑度六项。**Temporal Flickering全量原始分数，不做静态筛选，不启动RAFT**；不汇总为官方Total Score。详见[VBench口径](../VBENCH_README.md)。本次直接输入已验证的12fps原生输出，不使用未做GPU验收的通用示例脚本。

输入目录（只读）：

- `/scratch/prj0000000275/jcy/xks/fastref/ltx_distilled_real232_frozen300_20261115/`
- `/scratch/prj0000000275/jcy/xks/fastref/ltx_distilled_synthetic600_frozen300_20261115/`

流程与资源：

1. `scripts/prepare.py`在login核对300个冻结输入、600个输出及历史基线，完整解码并记录身份/权重/源码哈希。
2. `scripts/launch.sh`启动login控制器；GPU仅用现有testqueue job205167的4张H200，不申请新作业。按本任务已确认的例外，SSIM/PSNR在节点8进程×4线程、login 2进程×2线程并行；原始闪烁CPU 2进程。不会占用其他作业。
3. 六个固定id、两模型的108项小测全部通过，再执行全量5400个新增视频指标键；LPIPS使用独立的原正式环境，VBench使用原正式VBench环境。
4. 控制器汇总、独立验收，写入完成或失败状态，最终释放整个205167。失败保留已完成结果和逐项错误，不填0。不要在已释放后重新运行入口。

本地结果：[消融报告](report/REPORT.md)、[总体/数据集/任务分组](report/summary.csv)、[逐视频](report/per_video_metrics.csv)、[逐帧GT三指标](report/frames.jsonl)、[差值](report/deltas.csv)、[失败记录](report/failures.json)。[状态](metadata/run_status.json)、[最终验收](metadata/final_validation.json)、[释放记录](metadata/release.json)分别保留。

[实测用时](report/timing_report.md)包含首次加载与启动开销，按600个输出计算九指标端到端吞吐；并行阶段时间不相加，GPU任务墙钟包含加载/解码，不冒称纯计算或单卡速度。详细时间与版本见`metadata/timing.json`、`metadata/implementation.json`、`metadata/tasks/`。运行日志和生成结果保留本地，不纳入Git；没有自动人工通过、打包或上传。

只读查询（不提交/续跑）：

```bash
python scripts/status.py
```
