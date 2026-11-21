# Lance：仅真实数据 vs 真实+合成数据评测

## 结论与交付

- 对比对象是仅用 CoinVE-4K 真实数据微调的 `step677`，以及真实+合成数据微调的 `gtfresh-v1 step1150`。
- 单轮评测使用 OmniEdit Spatial 48 条（24 类×2），两个模型共生成 96 个结果。
- 多轮评测使用同一组 3/4/5 轮编辑链；人工审核以 autoregressive 结果为主，teacher-forced 和 4 条真实样本结果仅作为诊断保留。
- 最终审核包含 51 个卡片、102 个模型审核单元和 183 个视频，不含 Gemini 评分。
- 百度网盘路径：`/apps/bypy/lance_review/omni48_multiturn_gtfresh1150_vs_coinve677_human_review_20260927.zip`。
- ZIP SHA-256：`7ca1ae64049c1980ad483d48cc737ece235e7dd8f624f4c74b9b35d7bf69a3b6`。

这两个现有模型的训练配置不只数据来源不同，因此结果用于比较实际效果，不能解释为严格的单变量“合成数据消融”。

## 师兄的微调方式

两个模型从同一份官方 `Lance_3B_Video` 基模开始，基模 SHA-256 都是 `7f0550e1...e6447c25`。训练时冻结基模、ViT、VAE、token embedding 和 LM head，只保存 LoRA adapter；因此 checkpoint 目录名中的数字表示**已完成的 optimizer update 数**，不是数据量、样本编号或模型版本号。

| 项目 | 仅真实数据：CoinVE `step677` | 真实+合成：`gtfresh-v1 step1150` |
| --- | --- | --- |
| 实际 checkpoint | `lance_coinve_4k_r64_seed42_e2_v1/checkpoints/0000677` | `datamake_coinve_mixed_ddp8_lora64_both_no_region_gtfresh_v1/checkpoints/0001150` |
| 数据形态 | 只有 CoinVE 单轮 source–target pair | 同一批 CoinVE pair + UE 合成多轮链的 dialogue prefix |
| LoRA | rank 64、alpha 64、dropout 0.05，只挂 generation experts，252 个目标模块 | rank 64、alpha 64、dropout 0，同时挂 generation 和 understanding 两条分支，508 个目标模块 |
| 训练规模 | 4×H200，gradient accumulation 2，全局每步 8 对视频，学习率 `1e-4` | 8×H200，每卡 batch 1、gradient accumulation 4、全局 batch 32，学习率 `3e-4` |
| 历史输入 | `one_step_x0_history`：复用训练 forward 得到的 detached 预测 clean latent，不进行完整在线扩散 rollout | teacher-forced：历史 VAE/VIT 条件和当前 target 都来自数据集真值，不做在线 autoregressive rollout |
| 监督 | Lance 原生 flow-matching latent MSE，CE 权重 0 | 主要也是 latent MSE；配置保留 CE 权重 0.25，但这批视频记录的运行日志中 CE 实际为 0 |
| region loss | 关闭 | 关闭；run name 中的 `no_region` 就是这个含义 |

纯 CoinVE 训练计划为 2 个 epoch、共 902 个 optimizer steps；`step677` 是预先定义的 1.5 epoch 里程碑，也是该仓库固定用于横向比较的 checkpoint，并不是最终的 2 epoch 权重。混合训练计划为 3000 steps；本次评测使用的是已经完整落盘的 `step1150` 中间 checkpoint。训练日志后来至少继续到 step1186，因此不能把 `step1150` 理解为整次训练自然结束的位置。

混合 run 名称可以这样读：

- `mixed`：CoinVE 真实域 pair 与 UE 合成链混合；
- `ddp8`：8 卡 DDP；
- `lora64`：LoRA rank 64；
- `both`：generation 与 understanding 两条分支都加 LoRA；
- `no_region`：不按编辑区域提高 loss 权重；
- `gtfresh`：从官方基模重新初始化 fresh LoRA，并用 ground-truth history 做 teacher-forced 训练；它不是另一个基模名称。

两个被评测 adapter 的 SHA-256 分别是：CoinVE `8c921d04...e4e0e56`，gtfresh `ca2cb30d...1c522c`。混合 adapter 约为纯 CoinVE adapter 的两倍大，主要原因是它覆盖两条分支，而不是因为训练数据文件被打进了权重。

## 数据集里的指令实际是什么样的

### CoinVE：真实画面域中的单轮复合编辑

这里的“真实数据”指真实画面域的成对编辑视频，不是无监督真实视频，也不是简单重建。训练集使用 3601 个 pair；每个样本只有一次 source→target，但一条 instruction 经常是包含 2–3 个子任务的编号列表。例如：

> 1. Add a pair of black over-ear headphones to the woman's head.  
> 2. Replace the red V-neck top worn by the woman with a white button-down shirt.  
> 3. Replace the solid light blue background with a blurred image of a modern office interior with bookshelves.

另一个样本同时要求删除头发和替换背景：

> 1. Remove the long blonde hair from the young woman in the white sweater.  
> 2. Replace the wooden plank wall background with a solid matte blue studio backdrop.

因此 CoinVE 虽然按 `add`、`remove`、`replace` 三个大分支组织，但实际指令会把以下任务混在一次编辑里：

- 添加眼镜、耳机、围裙等人物配饰或物体；
- 删除头发、桌面物体等指定内容；
- 替换衣服、前景物体、桌面材质或背景场景；
- 偶尔对局部区域做风格化。

它的优势是画面和指令更接近开放世界真实编辑，能教模型处理人物、物体、服装和背景的组合变化；缺点是监督往往不是原子操作。一个结果失败时，很难判断模型是不会某个子任务，还是同时执行多个子任务时发生了冲突。数据中的 branch/operation 标签也只表示样本所属的大类，不能完整概括 instruction 内的全部动作。

纯 CoinVE 训练固定使用英语原始指令，不做 prompt rewrite；source 与 target 使用同一个绝对时间起点裁出 5 秒、12 fps、480p 片段，也不做水平翻转。这些约束是为了保持成对视频的时间对齐。

### UE 合成数据：严格受控的原子多轮编辑

混合训练中的 UE 数据不是 CoinVE 风格的复合列表。每一轮通常只改变一个因素，并明确要求其他因素保持不变。当前实际训练记录只出现四种 edit axis：

- `identity_id`：替换人物身份，描述会细到肤色、脸型、眼睛、鼻子、嘴唇、胡须、发型和痣；
- `outfit_id`：只换服装，例如女式连体裤、夏季休闲装、男式 1920 年代服装或街头服装；
- `action_id`：只换动作，例如挥手、跳舞、欢呼或喘息恢复；
- `camera_id`：只换镜头，例如 wide frontal、close frontal 或 side-view。

典型原子指令如下：

> Replace the person in the video with an adult man with light warm skin, a long rectangular face with a broad jaw and squared chin, light blue-gray eyes, a long prominent nose, medium lips, a clean-shaven face, dense black curls on top with closely faded sides, and a small mole below the lower lip, while keeping the outfit, action, expression, scene, and camera unchanged.

> Change the person's clothing to a women's summer casual outfit, while keeping the person, action, expression, scene, and camera unchanged.

> Make the person wave in greeting, while keeping the identity, outfit, expression, scene, and camera unchanged.

> Switch the camera to a wide frontal shot, while keeping the person, outfit, action, expression, and scene unchanged.

多轮链还包含“恢复到第 2 轮后的动作”或“把镜头恢复到初始状态”这类历史引用。训练记录以 dialogue prefix 展开：训练第 N 轮时能看到此前指令和对应的真值视频历史，并预测第 N 轮真值。5 轮 prefix 会重复 3 次，1–4 轮各出现 1 次，所以该配方有意加强长链和后期轮次。

混合训练集中的 CoinVE 部分仍是上述 3601 个真实 pair，每个只放一次；UE 部分来自约 3600 条训练链，展开并重复后形成 14675 个训练 prefix。这里的数字用于说明训练倾向：模型看到的混合记录明显偏向受控的人物身份、服装、动作、相机编辑，而不是涵盖所有可能的合成编辑类型。

## 如何理解这组对比

- CoinVE 主要提供真实画面分布和开放世界复合编辑；UE 主要提供严格对齐、一次只改一个因素的监督，以及多轮历史保持/恢复能力。二者是互补关系。
- 预期混合模型更可能在人物身份、服装、动作、镜头和多轮保持上受益；不能仅凭这批 UE 数据推断它会改善任意物体添加、背景替换、天气、光照或全局风格编辑。
- 混合训练采用 teacher forcing，训练时始终看到正确的上轮视频；真实推理则把模型自己的上轮结果继续送回去。训练没有直接覆盖这种误差累积，所以多轮评测必须看 autoregressive 结果，teacher-forced 只能用于定位单步能力。
- 这不是严格合成数据消融：除了数据，两个模型还同时改变了 LoRA 覆盖分支、dropout、学习率、batch size、历史构造、CE 配置、GPU 数量和训练步数。审核结果可以回答“这两个现成 checkpoint 哪个更好”，不能单独回答“提升是否由合成数据造成”。

## 最重要的执行经验

1. 先做最小 smoke，再启动 4 卡分片正式推理；每个 shard 独立记录配置、日志、技术检查和状态，失败时只重跑缺失项。
2. 加载 adapter 后必须核对张量匹配数和非零 LoRA-B 数，不能仅以“脚本未报错”判断权重生效。
3. 旧 CoinVE checkpoint 使用 legacy schema，新 runner 使用 `D1` 数据键；应创建不修改原 checkpoint 的推理兼容目录，并保留来源路径与哈希。
4. 多轮主结果必须使用 autoregressive 回填，才能观察误差累积；teacher-forced 适合定位单步能力，不应替代多轮结论。
5. 审核包视频统一为 H.264/yuv420p/faststart，逐个完整解码，并用真实离线 Chrome 验证播放、CSV/JSON 导入导出和零网络请求。
6. 网盘交付同时保留 ZIP、SHA-256 sidecar 和分块上传收据；上传完成后重新查询目录，核对远端路径、大小与文件 ID。

## 保留内容

- `configs/`、`logs/`、`technical/`：正式单轮分片的冻结配置、日志和技术检查。
- `metadata/`：冻结任务清单、评测来源凭据和单轮/多轮完成状态。
- `outputs_raw/`：两个模型的 96 个 OmniEdit 正式输出。
- `multiturn/`：两个模型的 autoregressive、teacher-forced 和真实样本诊断结果及 manifest。
- `inference_checkpoints/`：旧 CoinVE checkpoint 的兼容入口；adapter 与原 checkpoint 是硬链接，不是额外权重副本。
- `scripts/`：准备、推理、打包、浏览器验收和最终校验脚本。
- `review/`：已上传的精确 ZIP、SHA-256 sidecar 和上传收据；解压 ZIP 后打开 `index.html` 即可离线审核。

收尾时已删除 Python 缓存、smoke 临时结果/配置/日志、完成后的 `*_pending.json`、多轮输出目录内可从原 checkpoint 重建的 adapter 复制副本，以及可从最终 ZIP 恢复的解压态审核副本。
