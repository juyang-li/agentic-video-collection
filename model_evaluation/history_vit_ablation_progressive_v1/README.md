# 已废弃：UE-chain History-ViT ablation v1

> **已废弃（2026-10-17）**：该实验误用了师兄的 UE/DataMake validation 链，并非用户在任务中心构建的 IVEBench real-person multi-turn benchmark，因此结果不能用于本次 History-ViT 消融的审阅、比较或引用。

错误实验产生的 normal/zero/Gaussian 推理视频、指标、日志、本地审核包，以及百度网盘中的审核 ZIP 和 SHA-256 校验文件均已删除。这里只保留 `benchmark.json` 与 `scripts/`，用于追溯错误来源和复用实现代码；不得直接按此 benchmark 重跑。

正确实验应基于：

- `../real_person_multiturn_benchmark_v2/benchmark.jsonl`
- 整链排除含非空 `history_reference_round` 的回撤/历史引用任务
- 推理前移除全部 `camera` 指令
- Base 与 gtfresh 均重新生成 normal/history 条件，不复用此次错误实验输出

按当前筛选定义，候选数据为 24 条非回撤链、去除 camera 后 65 个 turn，其中 41 个为可做历史链消融的 R2+ 检查点。

