# A849 外部视频编辑评测 v1

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。

2026-11-13更新：用户要求扩至96条，并去除身份/发型参考背景。当前正式入口迁至[96条扩展版](../a849_external_vbench_v2_96/README.md)，本目录保留24条历史版本及已交付审核包；这里的身份/发型旧参考输出不计入新版。

后续待办见[TASK_QUEUE.md](TASK_QUEUE.md)：官方Ingredients IC-LoRA补跑当前24条；按用户要求先入队，不立即启动。

冻结24条、三模型共72条输出。用户查看14条预览包后，明确取消smoke关卡并授权直接全量；身份参考改为完整原图、仅头发RGB255，不裁剪、不缩放，其他像素按位保持。当前版本为whitehair_fullframe_v2。旧smoke证据和输出保存在metadata/before_whitehair及archive/smoke_before_whitehair；只复用任务输入完整记录完全相同的11条有效输出，其原始生成契约保留不改，其他61条生成。4条身份任务全部重新生成，不混入旧裁脸参考结果。

不调用Gemini或其他VLM API。所有计算仅在testqueue，优先使用用户指定的199811，过期后使用scripts/run.sbatch申请四卡续跑。按用户要求用休眠等待，避免无必要的频繁查询和token消耗。

当前运行状态以metadata中的逐条记录为准，尚未完成的任务不视为通过。入口：`scripts/run.sh cache|conditions|all`，`scripts/score.sh all`，`scripts/review.py --stage all --pack`。这些入口均要求真实testqueue分配，生成前受gate.py检查；当前full_launch_authorization.json记录用户显式跳过smoke，不伪造smoke评分通过。whitehair_revision.py仅执行一次；早期freeze.py及prepare_references.py保留供追溯，不用于覆盖当前版本。

样本：TripVVT6（上装/下装/dresses各2），OmniEdit自建身份4、单正面发型4、背景4（室内外各2），OmniEdit官方人物编辑6。TripVVT使用已有本地独立测试划分及已有参考图换衣指令；该指令为本地模板，官方caption另存且仅供审核，不能称为官方编辑指令。dresses沿用数据集类别，实际参考可能为连体裤或外套。OmniEdit源任务及完整官方指令保存在selection.jsonl；自建任务保留原任务来源但使用冻结的新指令，无伪造GT。

source按绝对时刻0至4秒（含端点）取49帧、12fps；编码播放长度49/12秒，不变速。统一保持比例补边1280×736或736×1280。实际选择PTS、像素变换与逐帧哈希保存在inputs/prepared；TripVVT沿用已有0至4秒窗口。

基座和A849复用同版本LTX2.5 dev原生权重，v7协议、40步、seed42、CFG3、STG1/block28、rescale0.7、ref_cfg1；仅LoRA有无不同。single_front_adapter.py只放行单张front hair的元数据检查，保留hair类型，不复制照片凑正背双图；参考token顺序、时间位置、模型和采样器不变，实际函数与上下游SHA记录在生成契约。

VACE复用现有固定版本Diffusers环境和Wan2.1-VACE-14B权重，49帧、12fps、40步、seed42、CFG5、flow_shift5、conditioning_scale1。使用相同source和参考图，完整原指令加由source语义编写的目标描述。mask完全从source构建，白色可编辑；可编辑RGB填128，其他像素按位保持。没有读取GT来生成输入或mask。

metadata/training_overlap.json记录A849完整可追溯训练谱系：849契约及保留的750/500/250 run.json，三个训练来源UE/CoinVE/TripVVT及开发集共6份清单。核对清单与各训练阶段记录SHA、source/target/reference记录、候选source和处理前后参考图片实际SHA。未检测到精确文件重合；不代表已排除重编码、近重复或基座预训练重合。

评分使用固定Git revision的官方[VBench custom_input](https://github.com/Vchitect/VBench#new-evaluate-your-own-videos)：subject_consistency、background_consistency、motion_smoothness、dynamic_degree、aesthetic_quality、imaging_quality。逐视频记录、按任务组的A849−base配对差值和VACE单列保留。缺失评分不填0，低质量有效输出保留。动态幅度独立解释，不合成官方总分；24条单种子只能观察趋势，不能证明统计意义上的质量不变，也不是官方榜单成绩。

编码与环境限制：LTX两组都使用官方H.264/YUV420 BT.709、CRF19/veryfast；VACE沿用现有PyAV RGB转YUV420、CRF20/fast。跨VACE画质分数可能包含轻微编码差异。VBench指标源码未修改，使用已实跑检查的PyTorch2.10/CUDA12.6环境，未证明与官方旧依赖栈数值逐位相同；版本、模型SHA及安装兼容处理保存在metadata。

审核包为本地离线连续滚动页，完整prompt和参考图逐条关联、固定导航、无评分控件。VACE prepared input/overlay仅在前两条样本附带展示。最终输出时间序列由Codex直接查看，观察记录与自动评分分开保存。
