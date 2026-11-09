# IVEBench 测评经验总结

本文记录 Lance Base 与本项目 LoRA checkpoint 在 IVEBench 上进行配对测评时已经验证过的做法、踩坑与结果解释原则。目标是让后续 checkpoint 或扩展数据集测评能够复用同一协议，而不是重新拼接一套不可比较的流程。

当前范围：

- Short / Subject Editing：91 条，已完成；
- Long / Subject Editing：43 条，已完成；
- Short+Long / Attribute Editing：94 条，已完成；
- Base 与 LoRA 均使用生成 seed 42；LoRA 为从 Lance 3B Base 训练的 step 677，不使用其他微调模型作为起点。

类别子集结果只适合 Base/LoRA 配对比较，不能冒充 IVEBench 600 条完整榜单分数。

## 1. 先冻结比较协议

开始批量推理前必须同时冻结以下内容：

1. 官方数据版本、评测代码版本和机械筛选规则；
2. Base 权重、LoRA adapter 路径和 checkpoint step；
3. seed、分辨率、输出 FPS、最大帧数、最大时长、采样步数与 CFG；
4. judge 模型、provider、reasoning、temperature、prompt 和媒体压缩规则；
5. 指标实现、归一化、权重和不适用指标的处理方法。

本轮协议固定为：

- 数据：`Coraxor/IVEBench@1e6cb655c0a537f6093edcf18e2607fb03e76734`；
- 官方评测代码：`RyanChenYN/IVEBench@2dcf8900ee65197f43854c76f61c66a32bb8dbfc`；
- 生成：480p、12 fps、30 steps、CFG 4.0、seed 42；
- API judge：`google/gemini-3.8-flash`、Google AI Studio only、low reasoning、`temperature=0`、禁止 fallback；
- API key 只从 `OPEN_ROUTER_QWEN` 读取，不写入配置或审核包。

只要其中一项变化，就必须生成新的 protocol hash 和独立结果目录，不能把旧分数混入新结果。

## 2. 模型看到的源视频才是评测真值

不能直接假设原始 MP4 的时长和帧数就是 Lance 的输入。正确比较对象是：

```text
Lance 实际采样后的 SOURCE  ↔  Base 输出  ↔  LoRA 输出
```

三者必须逐条满足：

- 帧数一致；
- FPS 一致；
- 时间轴对齐；
- Base 与 LoRA 使用完全相同的源输入和生成参数；
- 输出能够完整解码，而不只是容器头可读。

### Short 的 `4k+1` 边界

Lance 的验证集先把 Decord 平均 FPS 四舍五入为整数，再由 `MultiClipsFrameSampler` 计算 12 fps 的 `4k+1` 帧数。不能直接用浮点 FPS 估算。

实际踩坑：源视频 73 帧、23.976 fps。Lance 先取整为 24 fps，最终输出 33 帧；旧校验器使用 23.976 计算，误判应为 37 帧。生成本身正确，错误发生在校验器。

因此，技术校验必须复现运行时代码的精确顺序，而不是使用“数学上看起来等价”的近似公式。对应实现与回归测试见：

- `../lance_tools/verify_outputs.py`；
- `scripts/test_evaluation_contract.py`。

### Long 的 121 帧协议

Long 输入先从官方完整时间轴均匀抽取 121 帧并冻结为 12 fps 视频；Lance 使用 `max_duration=10.1`，从而保持 SOURCE、Base、LoRA 均为 121 帧。不能只取原视频开头，也不能让 source 与 target 时长不同。

121 帧推理远慢于 Short。本轮 H200 实测常见约 6–8 分钟/条/卡，状态检查不能因为数分钟没有新文件就判定卡死。应结合主进程、GPU 利用率、显存和 worker 状态判断。

## 3. 自动评分必须同时控制漂移与位置偏差

`temperature=0` 会降低随机性，但不能保证同一视频每次得到相同分数，也不能消除 A/B 位置偏差。因此本轮同时保留三类证据：

1. **绝对评分**：分别给 Base 与 LoRA 的 Instruction Satisfaction、Content Fidelity 打分；
2. **双向成对盲评**：每条都执行 A/B 和 B/A，对调候选顺序；
3. **稳定性复评**：固定抽取样本，对两模型重复绝对评分，统计 exact match 与平均绝对差。

只有两个顺序结论一致时，pairwise 才记为 Base 胜、LoRA 胜或 tie；否则记为 `order_inconsistent`，不能强行归给任何模型。

绝对总分和 pairwise 可能互相矛盾。出现矛盾时应报告两者，而不是只选择支持预期的一项。较稳妥的判断顺序是：

1. 先看配对差值和 bootstrap 95% CI；
2. 再看双向 pairwise 的胜负、平局和顺序不一致；
3. 最后看提升是否集中在单一子类，并回到审核包肉眼核查。

## 4. 指标聚合与适用范围

本轮使用官方实现的 9 个适用本地指标；官方 Qwen2.5-VL-72B 的 IS/CF 改由固定 Gemini API 协议给出。API 媒体最多取 20 个严格配对时间点，最长边 426、5 fps；本地指标仍使用全部生成帧。

聚合沿用官方权重：

- VTSS：权重 4；
- Instruction Satisfaction：权重 3；
- Content Fidelity：权重 3；
- 其他适用指标：权重 1；
- Quality、Compliance、Fidelity 三个维度最终等权。

Subject/Attribute 子集没有 quantity modification。Quantity Accuracy 应记为 N/A 并从 compliance 分母排除，不能伪造为 0，否则会系统性压低子集分数。

## 5. 当前结果教训

| 阶段 | Base | LoRA step 677 | LoRA−Base | 解释 |
| --- | ---: | ---: | ---: | --- |
| Short Subject，91 条 | 0.6609 | 0.6599 | -0.0010 | 基本持平 |
| Long Subject，43 条 | 0.6456 | 0.6559 | +0.0103 | 小幅正向，但 95% CI 跨 0 |
| 两阶段描述性合并，134 条 | 0.6560 | 0.6586 | +0.0026 | 没有整体明显提升 |
| Short+Long Attribute，94 条 | 0.6162 | 0.6170 | +0.0008 | 基本持平，95% CI 为 [-0.0085, 0.0102] |

合并 pairwise 也不支持“整体显著变强”：

- Instruction：Base 38 胜，LoRA 28 胜，44 平，24 顺序不一致；
- Fidelity：Base 47 胜，LoRA 47 胜，16 平，24 顺序不一致；
- Overall：Base 50 胜，LoRA 45 胜，15 平，24 顺序不一致。

但子类分析揭示了总分掩盖的能力迁移：

- 删除主体有明确提升信号；
- 添加主体结果混合；
- 替换主体偏弱，Long 上尤其明显。

因此当前 checkpoint 更准确的描述是“学到了偏向删除主体的局部能力”，而不是“整体编辑能力明显提升”或“微调完全失败”。后续训练应优先检查训练数据的编辑类型分布、目标变化是否足够清晰，以及添加/替换样本的监督质量，不应仅靠增加 epoch 解决。

Attribute 的总分同样掩盖了方向相反的变化：Quality 几乎持平（+0.0003），Compliance 下降（-0.0259），Fidelity 上升（+0.0281）。绝对评分中 LoRA 的 Content Fidelity 从 3.21 升到 3.37，但 Instruction Satisfaction 从 2.71 降到 2.54；双向 pairwise 也显示 Fidelity 倾向 LoRA（45 vs 38），Instruction 倾向 Base（14 vs 33），Overall 为 LoRA 32、Base 40、tie 3、顺序不一致 19。因此不能根据 +0.0008 的总分声称属性编辑提升。

三个属性子类的描述性结果为：

- 颜色调整 45 条：+0.0047，主要来自 Fidelity 增长，Instruction 仍下降；
- 位置变化 22 条：+0.0017，接近持平；
- 主体缩放 27 条：-0.0063，Quality、Compliance、Fidelity 均下降，且 pairwise Overall 为 Base 15 胜、LoRA 6 胜。

这说明后续扩集不能只追求属性样本总量：应优先补足目标变化明确、遮罩与时序一致的缩放样本，并单独跟踪 Instruction 与 Fidelity，防止模型以更保守的画面换取内容保持分，却没有完成编辑指令。本阶段 API 总费用为 0.5331 美元；18 项稳定性复评的 Instruction exact match 为 94.4%，Fidelity exact match 为 88.9%。

## 6. 可续跑比一次跑完更重要

长视频生成和 API 评分都可能跨作业墙钟时间或遇到暂时错误。流水线应以逐样本、可校验、幂等为原则：

- 已有 MP4 必须先完整解码并核对规格，合格才复用；
- 每个分片保存技术 inventory、SHA256 和生成 provenance；
- API 成功结果按任务与协议哈希缓存，失败重试不得重评成功项；
- 状态文件只作调度线索，最终完成必须由输出数量、完整解码、指标记录数和远端回执共同证明；
- 一个分片失败时保留其他分片成果，只补缺失样本。

本轮校验器边界错误说明：进程 return code 非零不等于模型输出无效。恢复前应先定位失败发生在推理、编码、技术校验、指标、API、打包还是上传中的哪一层。修复后先对已有结果重新校验，再决定是否补算。

## 7. 审核包是最终判断界面

自动分数用于筛选和定位，不能替代视频核查。审核包应做到：

- 完全离线，解压后直接用 Chrome/Edge 打开；
- SOURCE、Base、LoRA 三列时间对齐，可同步播放、暂停和拖动；
- 同页展示绝对分数、完整评语、双向 pairwise 与本地指标；
- 人工字段默认填 1，使用 `1=通过、0=不通过、2=存疑`；
- 人工结果可导出 CSV，不能只留在 localStorage；
- 构建后用真实 Chromium 验证播放，再校验 ZIP CRC 与 SHA256；
- 上传百度网盘后核对远端路径、大小和 sidecar，而不是只看到上传命令返回 0。

审核时优先比较：指令是否真正完成、未编辑内容是否保持、时间一致性、主体形变与逻辑正确性。单帧接触表只适合快速定位，不能替代完整视频。

## 8. 推荐执行顺序

```text
冻结数据与协议
→ 生成配置和 manifest
→ Base/LoRA 各 1 条 matched smoke
→ 四卡 Base 全量生成与逐分片校验
→ 四卡 LoRA 全量生成与逐分片校验
→ 冻结时间对齐的 judge media 和本地指标输入
→ API 预检、真实费用估算与余额检查
→ 本地指标和绝对评分
→ 双向 pairwise 与稳定性复评
→ 官方权重聚合、paired bootstrap、子类分析
→ 离线审核包与真实浏览器验收
→ ZIP/SHA256、百度网盘上传与远端复核
```

稳定运行阶段按项目规则低频检查；启动、异常、阶段切换和完成节点可以额外检查。不要通过频繁轮询、重复启动或临时改变协议来制造“进度”。

## 9. 当前事实来源

- 执行协议与入口：`README.md`；
- Short 结果：`artifacts/short_subject/evaluation_report.json`；
- Long 结果：`artifacts/long_subject/evaluation_report.json`；
- 逐样本结果：各阶段 `sample_results.csv`；
- API 协议与费用：各阶段 `gemini_*_temp0_v1/`；
- 技术校验：各阶段 `technical/`；
- 调度状态：各阶段 `generation_status.json`、`orchestration_status.json`；
- 审核包与上传回执：各阶段 `review/`。

Attribute 审核包已上传至百度网盘：`/apps/bypy/lance_review/ivebench_attribute94_base_vs_lora_step677_seed42_gemini38flash_temp0_v1.zip`；远端大小、ZIP CRC、SHA256 与 sidecar 上传状态均已核验。
