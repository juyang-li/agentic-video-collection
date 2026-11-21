# 下一批 4K：Gemini-only Stage 2

2026-09-06。用户要求从新候补中筛出 4,000 对，CoinVE 暂不加入。

## 状态

已完成 12 条启动验证及追加 500 条小批：合计 **512 条，282 accepted、230 rejected、0 quarantine**，记录费用 **$2.937315**。3 次 HTTP 503 失败后重试成功；这些失败请求未返回 usage.cost，费用上限中额外保守计入 $0.09。小批原始响应、路由、台账及输出覆盖核验全部通过。

用户已更新 API key，并明确“批准上传”。新 `LIJUYANG_ICLR` key 的剩余额度 $60、账户余额 $57.851727869，资金足够；但正式续跑的 308 个请求全部被 OpenRouter 拒绝，返回 HTTP 403。最小纯文本请求也得到同样错误：`The request is prohibited due to a violation of provider Terms Of Service.`，`provider_name=null`。进程已停止，当前状态 **blocked_by_openrouter_403**；需要用户恢复该 key 的 API 调用权限。未修改路由或改用其他 provider 绕过拦截。

新 key 运行前后 usage 均为 $0，因此这 308 次访问失败没有计费；`request_cost_reconciliation.json` 逐请求记录了零费用核对依据。此前 3 次旧 key 的 HTTP 503 仍缺少费用回执，保守预留 $0.09。当前仍只有 **512 条有效模型判断、282 条 accepted**，另 308 条账户访问失败暂列 quarantine，恢复权限后可以续跑它们。

`progress.json` 的 `valid_decided_pairs` 表示有效二分类数量，`attempted_pairs` 包含访问失败；历史兼容字段 `judged_pairs` 同样包含失败尝试，不能当作有效模型判断数。`smoke_verification.json` 保留最初 12 条检查，`initial512_verification.json` 保存 512 条小批核验；最新 `partial_verification.json` 包含访问失败分区的检查。

2026-09-07 13:46（新加坡时间）追加诊断：同一节点 `H64Y574`、同一 `LIJUYANG_ICLR` 调用 `qwen/qwen3-vl-32b-instruct` 成功，provider=Alibaba，HTTP 200，返回 `OK`，费用 $0.00000208。此前 Google AI Studio 和 Google Vertex 两路 Gemini 均在路由 attempt=0 时返回相同 403。由此确认该 key 可以正常完成部分模型调用；问题范围集中在 Google/Gemini 访问，具体账户、地区或 IP 策略仍未确定。此 Qwen 请求只用于诊断，不计入 Stage 2 筛选结果。记录见 `access_qwen_diagnostic_20260907.json` 和 `access_vertex_diagnostic_20260907.json`。

进一步交叉测试确认不是计算节点出口被 Google/Gemini 整体封禁：同一节点上，旧环境变量 `OPEN_ROUTER_QWEN` 对 `google/gemini-3.8-flash` 的 Google AI Studio 与 Google Vertex 路由均返回 HTTP 200；一张程序生成的 64×64 纯蓝测试图也由 Google AI Studio 正常识别为 `Blue`。新 `LIJUYANG_ICLR` 则对 Gemini 3.8 Flash、3.7 Flash、2.5 Flash 均在路由 attempt=0 时返回相同 403。因此限制落在新 key 所属 OpenRouter 账户或其 Google 模型访问资格，而非节点。旧账户当前可用余额约 $0.77，不足以完成生产筛选；改用旧账户上传真实数据仍需用户明确批准并补充约 $50 余额。

## 数量与预算

| 分支 | 本批目标 | 首批 Gemini 通过率 | 预计需审查 |
| --- | ---: | ---: | ---: |
| GOKU remove | 600 | 602 / 1,773 = 34.0% | 1,767 |
| GOKU add | 400 | 420 / 1,251 = 33.6% | 1,191 |
| EgoEdit replace | 1,000 | 1,026 / 1,369 = 74.9% | 1,334 |
| ReCo add | 600 | 622 / 867 = 71.7% | 836 |
| ReCo remove | 600 | 634 / 1,237 = 51.3% | 1,171 |
| ReCo replace | 800 | 837 / 1,061 = 78.9% | 1,014 |
| 合计 | 4,000 | 按分支估算 | 约 7,314 |

此前 1,882 条 Gemini 直筛平均 $0.005722 / 对，因此本次期望费用约 **$41.85**。考虑通过率与失败重试，预计 **$40–50**，本轮累计费用运行上限 **$55**。这不是要求把 18,662 条候补全部调用；各分支满额后停止派发该分支。

本批 512 条的实测通过率更新后，中心估计为审查 **7,841 条、约 $44.98**；对用户更新的总预算范围为 **$45–55**，包含已记录的 $2.94，不是另加 $45–55。此估计仍有小样本误差。

2026-09-06 核对的 [OpenRouter 标准 Gemini 3.8 Flash 价格](https://openrouter.ai/google/gemini-3.8-flash) 为 input $0.75 / 1M、output $3.75 / 1M，与历史实测一致。费用估算只含本次 Stage 2 API 调用，不包含此前已完成的 Stage 1、账户充值手续费或其他任务。首次启动实测 $0.0056835 / 对。

## 固定规则

- 输入为 [Stage 1 的 18,662 条 survivor](../../../1_specialist_filtering/artifacts/stage1_reserve_20k_20260906/stage1_survivors.csv)，包含 accepted 和 uncertain；原清单保持冻结。
- 使用 `google/gemini-3.8-flash`，Google AI Studio 标准端点、minimal、max_tokens 2048、禁止 fallback、data_collection deny。
- 复用首批 4K 的 CPU 抽帧、5 个对齐 full/ROI time cards、provenance 路由与规范化判定代码。prompt SHA256：`32e0d04ebc02bd292725425f144e6eab9d36b7915a39a4da2b5511176394212c`。
- seed 20260906，各分支按 sample ID 的固定哈希顺序审查。accepted 加在途请求不得超过分支配额，因此满额时恰好获得 4,000 accepted，不需额外随机丢弃合格项。
- API、解析错误和模型 uncertain 进入 quarantine。已取得有效判断的样本不会因续跑而重复调用；拒绝结果不重试。每个失败样本至多 3 次 API 请求；账户授权/余额错误停止扩发。
- `request_journal.jsonl` 保存每次 HTTP 尝试与原始响应/usage，包括解析失败的收费响应；`score_attempts.jsonl` 保留每次打分结果，最终 CSV 只使用每个样本最后一次结果。缺少 usage.cost 的请求单列，并在费用上限中按 $0.03 / 次保守计入；不冒充零费用。
- Stage 1 已核验本批与历史 pair 去重、与首批 validation/test 在完整候选及历史媒体连通图上无交集。此结论基于 `(dataset, locator)`，不宣称不同 locator 间的内容级去重。
- 本次只冻结新增合格 4K，不修改首批数据、不重新划分旧 train/validation/test。新增样本尚未分配训练 split。

## 文件与续跑

- `config.json`、`candidate_manifest.csv`、`judge_prompt.txt`：本轮冻结配置、候补顺序与提示词。
- `gemini_scores.csv`：当前每条已完成样本的最终 judge 结果。
- `accepted.csv`、`rejected.csv`、`quarantine.csv`、`not_requested.csv`：本次进程结束时生成的分区；运行期间请看 `progress.json`。
- `final_edit_pool_4k.csv`：仅六个分支全部满额时生成。
- `verification.json`：仅完整 4K 核验通过后生成；启动/小批验证用 `partial_verification.json`。
- `access_diagnostic.json`：新 key 的 HTTP 403 与零计费诊断，不含密钥。

目前不可直接重试。用户确认 API 访问已恢复后，先用原模型/路由做最小调用验证，再以 `--max-new-pairs 6 --retry-account-errors` 检查六分支；通过后移除小批数量限制继续。既有 308 条错误的尝试记录保留，最多 3 次总请求约束仍生效。遇到 HTTP 401/402/403 时现已设置全局停止派发，避免连续访问拒绝再次扩大。

从项目根目录 `video_collection` 执行；程序自动续跑：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python \
  lance_general_edit_finetune/2_vlm_filtering/scripts/run_stage2_reserve_4k.py --run

/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python \
  lance_general_edit_finetune/2_vlm_filtering/scripts/verify_stage2_reserve_4k.py
```

小批运行可添加 `--max-new-pairs 500`，随后验证添加 `--partial`。程序通过文件锁防止同一运行目录被两个进程同时写入。密钥从现有环境读取，不写入配置或日志。
