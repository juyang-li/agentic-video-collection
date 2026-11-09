# 近期人物动作参照数据调研（2026-10-30）

本轮按数据实际公开可取得时间考察 2025-04-30 至 2026-10-30；不沿用 HMDB51，也不把旧数据集的近期提交时间算作新数据集发布。优先查看动作原片、主体完整性和实际任务协议。公开 benchmark 数量较小，下面同时报告关联项目热度和数据集下载量，不将二者冒称独立用户数。

## 已取得并打包的内容

| 数据／项目 | 时间证据 | GitHub star | HF 月下载／likes | 本包内容 | 实际协议 |
|---|---|---:|---:|---|---|
| X-Dance / SteadyDancer | 2025-11-24 官方发布数据 | 652 | 191 / 20 | 全部12条驱动视频：8复杂动作＋4日常动作，附人物图与官方pose | 不同来源人物图片＋驱动视频；无跨身份编辑后GT |
| RealisDance-Val | HF公开库建立2025-05-21；论文2025-04-21，代码/权重2025-05-19 | 616 | 903 / 1 | 从100条中目视选24条人物运动，官方同编号ref＋gt | 同来源图片＋原视频的自重演；不能作为旧动作→新动作GT |
| MoveBench / Wan-Move | HF库2025-12-08，代码公告2025-12-10发布数据 | 662 | 321 / 3 | 人物运动子集11条，官方首帧＋原视频 | 点轨迹运动控制素材，原始832×480；不是跨人物动作替换数据 |
| Wan-Animate / Wan2.2 | 该任务于2025-09-19发布 | 18,266（整个Wan2.2仓库） | 不适用 | animate与replace各1条官方输入示例 | 参考人物图＋驱动视频；replacement更换角色外观并沿用动作 |
| Wan-Animate-2 | 2026-08-07发布 | 359 | 不适用 | demo1官方输入1条 | 直接视频驱动；本包示例为风格化角色 |

所有数字是2026-10-30官方API快照，原始证据在research_facts.json。RealisDance-Val确实存在“论文早于一年半时间窗、公开数据库在窗内”的区别，已明确披露；不是将2024版RealisDance当作新数据。

X-Dance还有被后续方法采用的证据：[SteadyDancer](https://github.com/MCG-NJU/SteadyDancer)提出它，[SCAIL-2官方项目页](https://teal024.github.io/SCAIL-2/)使用X-Dance做舞蹈比较。RealisDance-Val也被SteadyDancer用于评测。这比只看项目star更能说明数据的研究复用情况。

## 获取入口与适用性

- [X-Dance数据](https://huggingface.co/datasets/MCG-NJU/X-Dance)、[SteadyDancer代码](https://github.com/MCG-NJU/SteadyDancer)：本轮最直接的独立动作参照候选。原始舞蹈、转身与手势清楚，复杂遮挡/运动模糊本身也是评测难点。参考图库和驱动视频在发布库中分开提供；本包展示的具体图片搭配是浏览配对，未声称官方提供了这12条固定组合。
- [RealisDance-Val数据](https://huggingface.co/datasets/theFoxofSky/RealisDance-Val)、[RealisDance代码](https://github.com/damo-cv/RealisDance)：动作和场景比只跳舞更丰富。本轮包含手势、踢腿、健身、瑜伽、家务和全身舞动，优先适合做动作素材池。ref与gt的同编号配对沿用官方。
- [MoveBench数据](https://huggingface.co/datasets/Ruihang/MoveBench)、[Wan-Move代码](https://github.com/ali-vilab/Wan-Move)：最近的Wan运动控制方向，1018条5秒原片、54类内容。选取人物动作部分，用于补充行走、跑步、健身与拉伸。原始480p，清晰度弱于前两组的多数视频，单列补充区，不做超分。
- [Wan-Animate代码](https://github.com/Wan-Video/Wan2.2)、[Wan-Animate-2代码](https://github.com/Wan-Video/Wan-Animate-2)：确认用户记忆中的角色动画／替换任务。仓库示例可以直接取得；没有证据证明完整训练数据随权重一并开放，所以只标“官方示例”。

## 重要但暂未纳入的候选

- **SCAIL-2 / MotionPair-60K（2026-06）**：约1,289 stars，任务上非常值得关注。它提供多种端到端角色动画配对，训练数据包含Wan-Animate/MoCha等生成来源。官方说明可提供较大子集，但需要[填写申请表](https://docs.google.com/forms/d/e/1FAIpQLSfZjC0fZmiYFYHg90_79Yl45ipQLfR8ZhOAahOs19nO8nMvxA/viewform)；不是能匿名直接下载的HF公开集。本轮不代填个人信息或提交申请。[官方代码与数据说明](https://github.com/zai-org/SCAIL-2/tree/wan-scail2)
- **MoCha（2026-01论文）**：约991 stars，是相关的角色替换模型；官方仓库没有直接给出可下载的完整配对数据入口，不能用模型热度代替数据已发布。[官方仓库](https://github.com/Orange-3DV-Team/MoCha)、[数据发布讨论](https://github.com/Orange-3DV-Team/MoCha/issues/26)
- **RefVideo-6M（2026-08）**：HF约24,472月下载、22likes，覆盖参考编辑，但需要人工批准访问。公开gallery里控制任务主要是边缘、深度、scribble等，未据此确认有独立RGB动作替换子集，不能贸然作为本轮动作数据推荐。[数据入口](https://huggingface.co/datasets/RefVideo6M/RefVideo6M)、[官方任务gallery](https://refvideo-6m.github.io/)
- **HuMoSet（2025-12）**：关联项目约1,294 stars，SCAIL-2也将其用作高质量source来源。数据主要是人物参考图、视频、音频与密集描述，公开演示偏讲话人物，不是专门的独立动作驱动配对。本轮聚焦更直接的身体动作候选。[官方说明](https://github.com/Phantom-video/HuMo)
- **UniAnimate-DiT / DynamiCtrl等**：初次发布在2025-04/03或更早，未作为“最近一年半内新数据”的主推荐。DynamiCtrl数据另有24小时删除约束，不适合制作长期离线审核包。

## 与当前微调任务的关系

本轮解决“清楚、明确、可取得的动作参照素材从哪里来”，没有伪造source—target编辑监督。若最终测评接口要求source视频＋独立driving视频，这些动画benchmark还需另行构造输入对，并检查画幅、肢体可见性、时长、动作与场景兼容。自重演GT只证明能重建原动作；跨人物动画示例也不能自动证明保留已有source视频所有非编辑内容。

## 质量与访问说明

不缩小原视频分辨率，不超分、不补帧、不变速；H.264视频仅无损重封装为faststart并去音轨。其他编码若需浏览器兼容，按原分辨率转H.264。各文件的原始规格、处理后规格、来源版本和语义见manifest。

本包仅供研究素材质量浏览。X-Dance标注Apache-2.0、MoveBench标注MIT；RealisDance-Val仓库标注MIT，同时说明原始视频来自互联网、原作者保有权利，仅限学习研究。代码许可证不自动覆盖互联网素材的全部权利。Wan示例采用各官方仓库既有输入，未生成新视频。
