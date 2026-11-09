# Lance 通用评测工具

本目录保存 checkpoint 无关的 Lance 视频编辑评测工具。具体实验只需要准备 benchmark JSON、模型路径和 comparison panel JSON，不应复制或修改 base 评测目录。

## 单个 checkpoint 推理与校验

```bash
bash model_evaluation/lance_tools/run_video_edit.sh \
  EVAL_ROOT RUNTIME_ROOT MODEL_PATH BENCHMARK_CONFIG RUN_NAME
```

脚本先完整解码已有数字编号 MP4，仅对缺失项运行推理，最后把逐文件尺寸、FPS、帧数、SHA256 和完整解码结果写到 `EVAL_ROOT/technical/RUN_NAME.csv`。推理参数固定为本项目 base benchmark 使用的 30 steps、CFG 4.0 和 480p 配置；framework/global seed、validation data seed 与 diffusion noise seed 均固定为 42。

## 官方 Spatial 视频评审

`judge_spatial_openrouter.py --route vertex --tasks TASKS.json --output SCORES_DIR` 是完整 Spatial 的正式 v2 评审入口，基模与 winner 必须共用该配置。固定 Gemini 3.8 Flash、low reasoning、`google-vertex/global` 路由，完整 source/generated MP4 和官方 prompt；保存逐请求费用并按视频与协议哈希续跑。密钥来自 `OPEN_ROUTER_QWEN`，不写入文件。详见 [`lance_base/README.md`](../lance_base/README.md)。

需要显式零温度时使用 `--temperature-zero`。该选项把 `temperature=0.0` 写入请求与协议哈希，并生成独立的 `*_temp0_v1` 协议；输出必须放入新目录，禁止与未传 temperature 的 v1/v2 结果混合。正式比较仍须让 Base 与 LoRA 使用同一任务清单、路由、prompt 和 temperature 协议。

`--route studio`（兼容旧调用的默认值）只用于复现旧 v1；该路由的 20 MB 限制导致基模 3 条失败。不同路由必须使用独立目录，代码检查 protocol，禁止混分或 fallback。

`--limit 1` 仅做接口预检，不代表完整 benchmark；有效结果可复用。单任务默认最多 4 次请求（跨续跑累计），永久 API 错误不盲目重试；扩大尝试次数需显式传 `--max-attempts`，不得换模型或降低评分规则。

恢复约定（2026-09-06）：8 次后仍为 502/504 的任务，冷却至少 30 分钟并确认没有同阶段评分进程后，可以用 `--max-attempts 9` 等方式每次只增加一次请求，累计硬上限为 12。默认 4 次不变，成功缓存不重评，费用未知的失败请求继续单列。此调整只改变本地重试上限，不改变任何发送给 judge 的参数、视频、prompt 或协议哈希。

## N 联 comparison

```bash
python model_evaluation/lance_tools/build_comparison.py COMPARISON_CONFIG.json
```

JSON 指定 benchmark manifest、任意数量 panel、输出目录、评分表和 ZIP。已有且规格正确的 MP4 会被复用；ZIP 通过临时文件重建并在校验后原子替换。需要强制重编码 MP4 时增加 `--rebuild`。

## 接触表与 loss

```bash
python model_evaluation/lance_tools/build_contact_sheets.py \
  INVENTORY.csv COMPARISON_DIR CONTACT_SHEET_DIR

python model_evaluation/lance_tools/analyze_training_metrics.py \
  --run NAME=RUN_DIR --run NAME_2=RUN_DIR_2 \
  --json SUMMARY.json --svg CURVE.svg
```

接触表只用于快速定位现象，不能替代全视频人工评分。loss 工具默认保留淡色 raw curve，并叠加 rolling mean；跨数据集比较聚合 loss 时还必须查看输出 JSON 中的 `dataset_loss`。
