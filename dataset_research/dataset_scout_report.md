# 原子视频编辑数据集候选与下一步

## 当前筛选口径

主线只收以下形式：

```text
source video + 一条原子指令（add / remove / replace） -> target video
```

优先要求官方按任务分开发布，单条样本只有一个可观察操作，并且能够从一个明确总体中随机抽查。人物属性或人物相关道具可以在这一总体上继续筛选，但不能先挑好看的样例再声称随机。

- **CoinVE-200K**：保留为高质量 multi-task 对照，不与 single-task 主数据混合统计。
- **服装重建、VVT、relighting、只有 raw video 的人物数据**：移出本轮主线；已做的 ViViD、Eevee、TripVVT 观察材料只作为旁支证据保留。
- `style`、全局打光和背景风格不是当前下一步。

## 已实际随机观察的主线数据

| 数据集 / task | 随机口径 | 结果 | 判断 |
| --- | --- | --- | --- |
| ReCo `add/remove/replace` | 官方明确每任务随机展示 50 条且无人工挑选；本地包实测每类 51 条，我们浏览每类自然排序前 10 条 | 任务形态最标准；已有质量问题记录，但旧 audit 方向标注需谨慎解释 | 核心参照 |
| OpenVE `local_change` | 固定 2 GiB 归档前缀内 39 条可下载总体，两批无放回抽 30 条 | instruction 20/30；人物相关 19/30；人物相关且指令通过 10/19；30 条压缩后仅 19 个无向视频 pair | 形式正确，先去重再过滤 |
| GOKU `remove` | 32 卷随机命中 `part32`，从卷内 5,438 个完整 case 抽 15 条 | instruction 13/15；严格六项 9/15；目标人物相关 7/15 | 可保留的 single-task 候选 |
| GOKU `swap_alter` | 32 卷随机命中 `part05`，卷内 8,260 条抽 15 条 | instruction 6/15；人物属性 0/15 | 任务形式正确但该分支质量低，不扩量 |
| Señorita `object_swap` | 一个随机卷的完整 cluster 中抽 15 条 | instruction 7/15；严格整体 6/15；人物相关 2/15 | 低于当前门槛 |
| CoinVE-200K | 完整 metadata 随机 15 条 | instruction 无失败；每条同时含 2–5 个编辑 | 仅 multi-task 辅助/对照 |
| InsAVE forward `remove` | 两个官方分片随机命中 `part_01`，从 9,994 个完整 pair 随机 15 条 | 人物 15/15；instruction 15/15；严格七项 13/15 | 可保留的整个人物删除辅助集，但音轨同步变化 |
| InsAVE reverse `add` | 排除语音标记与显式音频词后 97 条中随机 15 条 | 人物 15/15；instruction 12 通过、2 失败、1 存疑；严格七项 8/15 | source 删除伪影较多，只能质检后用 |
| EgoEditData `replace` | 完整发布编辑表 93,644 条中均匀随机 15 条 | 15/15 为单物体 replacement；人物属性 0/15；预览发现 1 个明显错配、1 个后段轨迹偏离 | 可作 object-replace/交互遮挡辅助，不补 add/remove |

因此，**single-task 和 multi-task 必须作为不同 supervision type 管理**。CoinVE 很好，但不能拿它代替 ReCo/OpenVE 这种原子边。

## 现在值得看的候选

| 优先级 | 数据集 | 与目标的关系 | 可获取与抽查判断 |
| --- | --- | --- | --- |
| 1 | GOKU `add` | 官方独立 single-task 分支，可直接与已抽的 `remove` 比较 | 当前账号已有 gate 权限；32 卷约 1.00 TB，固定 seed 命中 `part32` 约 29.98 GB，可按相同方式抽 15 条 |
| 2 | JAVEdit-100k | 约 103K，分 Subject Editing、Subject Removal、Subject Addition 等独立 pipeline，1280×720、121 帧 | 可下载约 108 GB，但它是联合音视频编辑，指令可能同时改视觉和音频；压缩大包也不利于浅层随机抽取，低于纯视觉原子集 |
| 已抽查 | EgoEditData | 第一视角对象 replacement，覆盖手遮挡、交互和大幅相机运动 | 已授权并完成全表随机 15 组；8 个 ZIP 约 245 GB 可 Range 精确取文件，但发布 annotation 没有独立纯 add/remove 子总体 |
| 已尝试 | OpenVE `local_add` / `local_remove` | 与现有 `local_change` 同一标准三元组，官方任务分开 | 固定 4 GiB 前缀仅形成 4 个 add、3 个 remove 完整 pair，达不到随机 15；停止继续盲拉，等待可随机访问索引或更宽松额度 |
| 已抽查 | InsAVE-80K `add_and_remove` | 21,377 对；正向基本都是 remove，反向基本都是 add | 已各做 15 组 H.264 随机观察；remove 明显优于反向 add，但 30/30 音轨均变化，不能冒充纯视觉 supervision |
| metadata 淘汰 | InsViE-1M | 公开 1.02M triplet、约 765 GB，但没有官方原子任务分栏 | 首词 add/remove/replace 各随机 15 条：add 仅 2/15 为具体实体；remove 有复合编辑；replace 大量静态来源和重复模板，因此不拉视频 |
| 仅索引 | OpenVE-HQ-1M | 从 OpenVE 过滤出的 969,578 条 metadata，含 `local_add` 175,814、`local_change` 171,561、`local_remove` 316,870 | 不提供新视频，不算独立数据集；适合做 OpenVE 的候选索引，但数据卡没有说明过滤方法 |
| 等待 | FFP-300K | 论文称 300K 个 720p、81 帧 local/global pair | 目前只确认论文，未找到作者正式数据仓库，不能抽查 |

OpenVE-HQ、ThinkV2V-150K 等基于 OpenVE 的 metadata 二次发布不能当作“新增独立视频数据”；FiVE-Bench、OmniEdit-Bench 只有 source 和编辑指令，没有训练用 target，也不进入监督数据主榜。Aurora 发布的 ROSE insertion/removal 分别依赖 reference image 和 removal mask，也不是本轮要求的纯 `source + text -> target`。

## 下一步顺序

1. 下一轮抽已有权限的 GOKU `add` 15 条，与 GOKU `remove` 同口径比较；这是当前最直接的 single-task 补缺。
2. EgoEdit 不继续盲目扩量：它已补到 atomic object-replace 证据，但不含独立纯 add/remove，且不属于人物属性。
3. JAVEdit 只有在 metadata 中确认能隔离纯视觉单任务后再拉，不直接下载约 108 GB 全量。
4. OpenVE `local_add/remove` 暂不继续扩大 gzip 前缀，除非获得归档成员索引或允许更高带宽。

最终整合至少保留 `atomic_task`、`supervision_type` 和 `audio_scope`：ReCo、OpenVE、GOKU 单任务写 `atomic_edit_pair`；CoinVE 写 `multi_edit_pair`；InsAVE 写 `atomic_entity_edit_with_audio_change`。训练或汇总时不得把三者混成一个通过率。

## 官方来源

- ReCo：[GitHub](https://github.com/HiDream-ai/ReCo)、[dataset](https://huggingface.co/HiDream-ai/ReCo)
- OpenVE：[paper](https://arxiv.org/abs/2512.07826)、[dataset](https://huggingface.co/datasets/Lewandofski/OpenVE-3M)
- OpenVE-HQ-1M：[dataset](https://huggingface.co/datasets/donghao-zhou/OpenVE-HQ-1M)
- GOKU：[paper](https://arxiv.org/abs/2606.30599)、[dataset](https://huggingface.co/datasets/bigfacing/GOKU-2M)
- EgoEdit：[GitHub](https://github.com/snap-research/EgoEdit)、[已授权的数据镜像](https://huggingface.co/datasets/liguang0115/EgoEdit)
- JAVEdit：[paper](https://arxiv.org/abs/2606.03168)、[dataset](https://huggingface.co/datasets/Coraxor/JAVEdit-100k)
- InsAVE：[GitHub](https://github.com/suimuc/InstructAV2AV)、[dataset](https://huggingface.co/datasets/suimu/InsAVE-80K)
- InsViE-1M：[GitHub](https://github.com/langmanbusi/InsViE)、[dataset](https://huggingface.co/datasets/wyh6666/InsViE)
- FFP-300K：[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FFP-300K_Scaling_First-Frame_Propagation_for_Generalizable_Video_Editing_CVPR_2026_paper.html)
