# InsAVE-80K 随机 pair 阶段结论

## 结论

InsAVE 的正向 remove 值得保留为“整个人物删除”辅助数据；反向 add 只能在严格质检后保留。它不是 OpenVE 的无条件替代品：这是联合音视频编辑数据，即使文字只写一次 add/remove，音轨也会同步变化。

| 方向 | 随机总体 | instruction | 严格七项全通过 | 主要判断 |
| --- | ---: | ---: | ---: | --- |
| forward remove | `part_01` 的 9,994 个完整 pair | 15/15 | 13/15 | 画面总体强；1 个明显修补伪影、1 个大块失焦存疑 |
| reverse add（无音频词） | 97 个候选 | 12/15 通过、1/15 存疑、2/15 失败 | 8/15 | 多个 source 是删除后留下的幽灵、黑洞或孤立物体，反向训练质量明显低于 remove |

这两批随机结果 30/30 都编辑整个人物，和当前 add/remove 任务方向一致；但它们不是发型、配饰等细粒度人物属性修改，也没有 replace。

## 数据组织与抽样

官方 `add_and_remove` 有 21,377 对、两个 tar 分片，约 32.25 GB。完整 metadata 中 21,374 条正向 instruction 以 remove 开头，对应反向 add；11,755 条反向 instruction 含 `<S>...<E>` 语音内容。

本次先用 seed `20260826` 在两个分片间随机，命中较小的 `part_01.tar`；扫描得到 9,994 个完整 source/target case。remove 从该总体用相同 seed 无放回抽 15。add 则反转 source/target，并排除 `<S>/<E>` 以及 `audio/music/sound/speech/voice/saying/speaking/talking` 等显式音频词，在剩余 97 条中用 seed `20260827` 无放回抽 15。没有按人物、场景或画质挑样。

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

## 音轨边界

两批各 15 对的 source/target 都含音轨且长度一致，但 30/30 的 16 kHz mono PCM 哈希不同：

- remove：source RMS 中位数 `1051.04`，target `370.00`，15/15 target 更安静。
- add：source RMS 中位数 `469.28`，target `896.87`，15/15 target 更响。

所以，即便 add 已过滤所有显式音频词，它仍包含未写进 instruction 的音频变化。若项目只训练视觉帧，可以显式丢弃音轨；若要保留音频，就不能把这些样本称为纯视觉 single-task。

## 决策

- forward remove：保留为原子“人物实体删除”辅助监督，扩量前仍需自动/人工质量过滤。
- reverse add：只在排除 source 删除伪影后使用，不能原样全收。
- reverse instruction 含 `<S>/<E>` 的样本：归入 audio-visual multi-modal 分支，不混入纯视觉原子榜。
- replace：InsAVE 这一类别不覆盖，仍需从别的数据集补齐。

官方来源：[InsAVE-80K dataset](https://huggingface.co/datasets/suimu/InsAVE-80K)、[InstructAV2AV code](https://github.com/suimuc/InstructAV2AV)。仓库声明 MIT，但官方同时要求用户自行核验底层媒体权利与使用限制。
