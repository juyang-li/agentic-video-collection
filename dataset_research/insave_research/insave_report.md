# InsAVE-80K 随机审核结论

## 结论

InsAVE 的正向 remove 值得保留为“整个人物删除”辅助数据；反向 add 只能在严格质检后保留。InsAVE 当前没有可直接整栏使用的 single-step 视觉 replacement 分支：`clone_id` 与 `clone_id_voice` 不是视觉 replacement，`clone_voice` 是 replacement 与 speech 的复合任务，`general_editing` 的简单 Replace 文本子总体在随机视频审核中严格可用 0/10。

它也不是 OpenVE 的无条件替代品：这是联合音视频编辑数据，即使文字只写一次 add/remove，音轨也会同步变化。

| 方向 | 随机总体 | instruction | 严格七项全通过 | 主要判断 |
| --- | ---: | ---: | ---: | --- |
| forward remove | `part_01` 的 9,994 个完整 pair | 15/15 | 13/15 | 画面总体强；1 个明显修补伪影、1 个大块失焦存疑 |
| reverse add（无音频词） | 97 个候选 | 12/15 通过、1/15 存疑、2/15 失败 | 8/15 | 多个 source 是删除后留下的幽灵、黑洞或孤立物体，反向训练质量明显低于 remove |
| `general_editing` forward Replace | `part_04` 的 1,295 个文本候选 | 0/10 通过、2/10 存疑、8/10 失败 | 0/10 | no-op、替错对象、endpoint 方向错误或只完成少量描述属性 |

前两批随机结果 30/30 都编辑整个人物，和当前 add/remove 任务方向一致；但它们不是发型、配饰等细粒度人物属性修改。第三批专门排查 replacement，但 3 对人物相关样本都是 person-to-animal，其余 7 对是物体 replacement，没有覆盖 appearance、clothes、accessory 或 background replacement。

## 数据组织与抽样

官方 `add_and_remove` 有 21,377 对、两个 tar 分片，约 32.25 GB。完整 metadata 中 21,374 条正向 instruction 以 remove 开头，对应反向 add；11,755 条反向 instruction 含 `<S>...<E>` 语音内容。

本次先用 seed `20260826` 在两个分片间随机，命中较小的 `part_01.tar`；扫描得到 9,994 个完整 source/target case。remove 从该总体用相同 seed 无放回抽 15。add 则反转 source/target，并排除 `<S>/<E>` 以及 `audio/music/sound/speech/voice/saying/speaking/talking` 等显式音频词，在剩余 97 条中用 seed `20260827` 无放回抽 15。没有按人物、场景或画质挑样。

Replacement 排查先读取 InsAVE 各官方分类的完整 metadata，再固定约 4 GB 的 `train_data/general_editing/part_04.tar`，避免为 10 对抽查下载约 20 GB 分片。该分片含 1,840 个完整 pair；筛选 forward instruction 无显式音频要求、以 `Replace` 开头且没有明确第二操作，得到 1,295 对，以 seed `20260826` 无放回随机抽 10。分片选择只考虑获取成本、不参考视频质量，因此该结果只代表固定 `part_04` 子总体，不能直接估计整个 `general_editing` 的通过率。

Replacement 抽查固定官方 revision `8ba8ccaad1d8f97b08e218dae1f2b439bd3d2289`；`train_data/general_editing.csv` 的 SHA256 为 `0beccbf5598379ab0da4ac47b18782152013aee7d05e22e5675ec7b6a3fdec95`。

## 审核结果

| 字段 | remove 通过 / 失败 / 存疑 | add 通过 / 失败 / 存疑 |
| --- | ---: | ---: |
| person relevance | 15 / 0 / 0 | 15 / 0 / 0 |
| instruction correctness | 15 / 0 / 0 | 12 / 2 / 1 |
| logical correctness | 15 / 0 / 0 | 14 / 1 / 0 |
| content preservation | 15 / 0 / 0 | 15 / 0 / 0 |
| temporal consistency | 15 / 0 / 0 | 13 / 2 / 0 |
| visual quality | 13 / 1 / 1 | 10 / 5 / 0 |
| technical quality | 15 / 0 / 0 | 15 / 0 / 0 |

30 对均为 1280×704，source/target 的分辨率、帧数和 fps 在每对内一致，均可解码。add 的主要失败不是人物没出现，而是其 source 本身来自失败或不干净的删除结果：可见人物残影、黑色修补孔洞、漂浮红鼻子等；另有两条位置/视角描述与 target 不符。

## Replacement 分类栏与视频审核

InsAVE 没有独立的 `replacement` 训练栏。各分类的实际用途如下：

| 官方分类栏 | 行数 | instruction 实际含义 | 当前判断 |
| --- | ---: | --- | --- |
| `clone_id` | 5,102 | 保持人物外观，改变 timbre 和 spoken words | 不是视觉 replacement |
| `clone_id_voice` | 5,625 | 保持人物 identity，只改变 spoken words | 不是视觉 replacement |
| `clone_voice` | 5,254 | 替换人物或服装，同时保持 timbre、改变 spoken words | replacement + speech 复合任务，不进当前 single-step 视觉池 |
| `general_editing` | 49,716 | 混合 replace/change/transform/add、speech 和复合编辑 | 有 Replace 文本候选，但简单过滤后的真实视频质量不合格 |
| `eval_data/eval` | 1,000 | 包含人物、物体和背景 replacement 等评测任务 | 只作评测，不进入训练候选 |

`general_editing` 全部 metadata 中有 19,086 条 forward instruction 不含显式音频要求，15,996 条以 `Replace` 开头；用简单文本规则排除明确第二操作后约有 13,742 条。这个数量只是 metadata heuristic，不代表可用视频数量。

固定 `part_04` 的随机 10 对逐项结果如下：

| metadata index | 指令目标 | 实际 target | instruction |
| ---: | --- | --- | ---: |
| 48013 | car → boxy cargo truck | 外形略变但仍像乘用车，夜景难确认 | 2 |
| 48089 | cratered moon → smooth pearl | 仍是有陨石坑的月亮，仅更亮 | 0 |
| 48325 | young woman → black cat | 同一名女性，近似 no-op | 0 |
| 48461 | gift box → leather suitcase | 紫色礼物盒，未变成手提箱 | 0 |
| 48764 | ladder → wooden stool | 变成蹲着的女性 | 0 |
| 49245 | man and woman → ravens | 男女仍在，近似 no-op | 0 |
| 49360 | blue pickup → olive station wagon | target 为亮蓝色皮卡，endpoint 方向相反 | 0 |
| 49412 | sailboat → black gondola | 船体变暗、变低，但仍有桅杆且无贡多拉特征 | 2 |
| 49634 | spacecraft → airship | 仍是航天器/返回舱 | 0 |
| 49694 | pedestrians → cats | 行人仍在；整面墙被错误重绘 | 0 |

| 审核字段 | 通过 | 失败 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 3 | 7 | 0 |
| instruction correctness | 0 | 8 | 2 |
| logical correctness | 10 | 0 | 0 |
| content preservation | 9 | 1 | 0 |
| temporal consistency | 10 | 0 | 0 |
| visual quality | 9 | 1 | 0 |
| technical quality | 10 | 0 | 0 |

10 对共 20 个原始 MP4 均能逐帧读到容器报告的完整帧数；每对 source/target 的 1280×704 分辨率、帧数和 fps 完全一致。问题主要来自 annotation 与实际 endpoint 不一致，而不是编码损坏。`person relevance` 是覆盖属性，不参与质量严格通过计算。

## 音轨边界

两批各 15 对的 source/target 都含音轨且长度一致，但 30/30 的 16 kHz mono PCM 哈希不同：

- remove：source RMS 中位数 `1051.04`，target `370.00`，15/15 target 更安静。
- add：source RMS 中位数 `469.28`，target `896.87`，15/15 target 更响。

所以，即便 add 已过滤所有显式音频词，它仍包含未写进 instruction 的音频变化。若项目只训练视觉帧，可以显式丢弃音轨；若要保留音频，就不能把这些样本称为纯视觉 single-task。

## 决策

- forward remove：保留为原子“人物实体删除”辅助监督，扩量前仍需自动/人工质量过滤。
- reverse add：只在排除 source 删除伪影后使用，不能原样全收。
- reverse instruction 含 `<S>/<E>` 的样本：归入 audio-visual multi-modal 分支，不混入纯视觉原子榜。
- replacement：InsAVE 没有可直接使用的官方分支；不把 `general_editing` 的简单 Replace 文本子总体加入 baseline。若继续利用它，必须另做 endpoint 语义、方向和视频质量过滤。
- 当前简单 baseline 的 replacement 优先审核独立 single-step 来源，例如 EgoEditData object replacement；appearance、clothes、accessory、background 继续从其他数据集补齐。

官方来源：[InsAVE-80K dataset](https://huggingface.co/datasets/suimu/InsAVE-80K)、[InstructAV2AV code](https://github.com/suimuc/InstructAV2AV)。仓库声明 MIT，但官方同时要求用户自行核验底层媒体权利与使用限制。
