# 当前数据集与筛选方案

> 更新日期：2026-09-06  
> 本文件是后续对话了解“现有数据能否训练、如何筛出”的首要入口。详细过程和历史实验仍以文末链接为准。

## 1. 当前结论

- 已冻结 **4,000 对 single-step 视频编辑数据**；其中 train 3,601、validation 195、test 204。
- 训练/验证/测试按共享媒体连通组划分，没有同一媒体跨 split 泄漏。
- Stage 2 已固定为 **Gemini-only**：每个 Stage-1 survivor 必须获得 Gemini 有效判断，Qwen 结果不再决定 canonical 数据池。
- OpenVid-1M 的 50 条分层样本已经人工快速审阅，整体质量可用；但首轮 LoRA **不混合 T2V**。OpenVid 暂存为后续出现明显基础画质或动态退化时的 replay 候选。
- 首轮 LoRA 只训练 4K 编辑池中的 **3,601 条 train**；validation/test 不参与训练。
- 下一批候补：2026-09-06 已完成 [20K Stage 1 扩筛](../1_specialist_filtering/artifacts/stage1_reserve_20k_20260906/REPORT.md)，保留 **18,662 对**（16,916 accepted + 1,746 uncertain），淘汰 1,338 对，无异常隔离。CoinVE 未加入、付费 judge 调用为 0；该候补池等待后续 Stage 2，不计入当前 4K 正式训练集。
- 下一批 Stage 2 已按用户要求启动，目标为按原六分支配额新增 **4,000 对**，暂不加入 CoinVE。512 条小批已核验，保留 282 条、记录费用 $2.937315；预计总费用 $45–55、上限 $55。用户已更换余额充足的 API key 并批准上传，但新 key 连最小文本调用也被 OpenRouter 服务条款 HTTP 403 拦截，已停止等待 API 权限恢复；新 key 失败请求未计费。新增 4K 尚未完成，详细状态见 [下一批 Stage 2](../2_vlm_filtering/artifacts/stage2_next4k_20260906/README.md)。首批 4K 与其 split 保持冻结。

## 2. 冻结编辑池

| 数据集分支 | 最终条数 | 主要任务范围 | train | validation | test |
| --- | ---: | --- | ---: | ---: | ---: |
| GOKU remove | 600 | object 418、accessory 160、person 22 | 548 | 28 | 24 |
| GOKU add | 400 | object 289、accessory 86、person 25 | 364 | 16 | 20 |
| EgoEdit replace | 1,000 | object replacement | 890 | 49 | 61 |
| ReCo add | 600 | object 341、person 156、clothes 98、其他 5 | 543 | 28 | 29 |
| ReCo remove | 600 | object 351、person 187、clothes 58、其他 4 | 537 | 30 | 33 |
| ReCo replace | 800 | object 387、person 256、clothes 142、其他 15 | 719 | 44 | 37 |
| **合计** | **4,000** | add 1,000、remove 1,200、replace 1,800 | **3,601** | **195** | **204** |

最终清单：

`artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv`

4,000 对引用约 8,000 个本地角色文件，总逻辑体积约 19.1 GiB。按数据集汇总为 GOKU 1,000 对、EgoEdit 1,000 对、ReCo 2,000 对。

## 3. 数量关系

| 层级 | 数量 | 含义 |
| --- | ---: | --- |
| Metadata 候选 | 约 891,646 | 任务范围入选，不代表都已下载或跑模型 |
| 实际完成 Stage 2 | 7,558 | 原池加 Round-4 补池 survivor |
| Gemini accepted | 4,141 | 可用编辑 pair |
| Gemini rejected | 3,402 | 编辑失败或质量不合格 |
| quarantine | 15 | API/安全/解析异常，不冒充质量拒绝 |
| 冻结正式池 | 4,000 | 按固定任务配额和 seed 选择 |
| accepted reserve | 141 | 合格候补，不丢弃 |

Accepted reserve：GOKU remove 2、GOKU add 20、EgoEdit replace 26、ReCo add 22、ReCo remove 34、ReCo replace 37。

## 4. Gemini-only Stage 2

### 4.1 固定方案

- 模型：`google/gemini-3.8-flash`，通过 OpenRouter 的 Google AI Studio provider 调用，禁止 fallback。
- 推理配置：`reasoning_effort=minimal`、`max_tokens=2048`。
- 图像输入：避开 conditioning 首帧的 5 个对齐时刻；每个时刻提供 source/target 全图 pair 和同坐标 tight ROI pair。
- 主要判定：`instruction_following`、`visible_artifact`、`content_preservation`。
- Provenance 明确时，以实际 generated 一侧作为主要质量对象；GOKU 反向 add 按实际生成方向审查。ReCo 逐 pair 生成侧未公开，因此按训练方向检查指令和内容保持，并检查两侧伪影。
- Stage-1 accepted 和 uncertain 都可进入 Gemini；只有 Gemini 返回有效、可解析的 accepted 才能进入最终 accepted pool。
- API、内容安全或结构化解析持续失败的样本进入 quarantine，不能记作质量 rejected，也不能进入训练池。
- 固定 prompt SHA256：`32e0d04ebc02bd292725425f144e6eab9d36b7915a39a4da2b5511176394212c`。

### 4.2 权威关系

Gemini 是当前 Stage 2 的唯一 canonical judge。Qwen2.5/3-VL、旧 rescue cascade 和 VEFX 只保留为历史 benchmark、错误诊断或研究字段，不得直接覆盖 Gemini 最终决定，也不得在分支缺额时降低门槛。

最终 7,558 条结果为 4,141 accepted、3,402 rejected、15 quarantine；随后使用固定数据选择 seed `20260904` 从 accepted 中冻结严格配额的 4,000 对。Stage-2 生产、全量 Gemini 补录和 Round-4 补池的记录费用合计约 `$72.07`。

## 5. 已排除或暂缓的数据

- InsAVE person remove：扩大审核后整体质量低，整支排除。
- InsAVE general editing replacement：指令遵循不足，不进入基线。
- OpenVE：较长视频少且质量不稳定，当前不用。
- GOKU swap/alter：审核质量不理想，暂缓。
- GOKU/InsAVE 衣物、鞋裤直接 add/remove：常出现不合理补全，排除。
- 手持物体 remove：手部和关节破坏普遍，排除。
- CoinVE multi-step：推迟到 single-step 能力稳定之后。
- TripVVT clothes：依赖服装参考图和 mask，不适配当前纯文本原子编辑输入。
- style transfer、camera、movement、subtitle、audio：不属于当前 single-step 训练范围。

ReCo clothes 保留：人工抽查表明其质量优于 InsAVE/GOKU 对应服装分支。ReCo robot/CG 不按关键词整类淘汰，而由质量门和 Gemini 判断。

## 6. OpenVid 与首轮训练决定

OpenVid-1M 的 50 条审核样本按 human 15、vehicle 10、animal 5、scene 10、object 10 分层。用户快速审阅后认为整体可用，但没有逐条导出的结构化人工标签，因此只能视为 **T2V 候选池通过初步验收**，不能表述为 50/50 严格通过。

因为第一轮的核心问题是验证综合编辑数据能否教会 Lance single-step 编辑，而且 LoRA 冻结基座、部署 T2V 时也可以关闭 adapter，本轮不混入原计划的 10% T2V：

1. 使用冻结 4K 清单中的 3,601 条 train 训练；
2. 195 条 validation 和 204 条 test 保持冻结；
3. OmniEdit 衡量编辑增益，VBench 对比基模、关闭 LoRA 和开启 LoRA 的基础视频质量；
4. 若编辑能力提高且 VBench 相对基模下降不超过约 5%，扩量继续使用纯编辑路径；
5. 只有出现明显画质或动态退化时，下一轮才引入约 5%–10% OpenVid replay。

## 7. 权威入口

- 数据清单与详细生产结果：[`STAGE2_PRODUCTION_REPORT.md`](./STAGE2_PRODUCTION_REPORT.md)
- Gemini 全量补录、配置与成本：[`GEMINI_FULL_REVIEW_STATUS.md`](./GEMINI_FULL_REVIEW_STATUS.md)
- OpenVid 抽样与审核包：[`T2V_REVIEW_REPORT.md`](./T2V_REVIEW_REPORT.md)
- Stage 1 门槛和结果：[`../1_specialist_filtering/STAGE1_PRODUCTION_REPORT.md`](../1_specialist_filtering/STAGE1_PRODUCTION_REPORT.md)
- Stage 2 prompt 与 VLM 规则：[`../2_vlm_filtering/README.md`](../2_vlm_filtering/README.md)
- LoRA 参数、seed 与评测约定：[`README.md`](./README.md)
