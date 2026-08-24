# InsViE-1M 原子任务 metadata 初筛

## 结论

不建议把 InsViE-1M 作为下一套 ReCo/OpenVE 型原子 add/remove/replace 主数据。它虽然公开了 1,019,570 行 source–instruction–target 对应关系，但任务没有官方分栏；按 instruction 首词得到的三个候选总体内部仍混入大量风格、打光、天气、复合编辑和静态图转视频样本。

因此本轮只下载 61 MB metadata 并做随机指令抽查，没有去下载 765 GB 视频分卷。这是基于任务结构的提前淘汰，不是视频质量结论。

## 实测总体

- fixed revision：`12efa8dee73ec310f9ad42aee502ca4fad73bc30`。
- metadata：1,019,570 行，SHA256 `b20cc78dce637eb3a1d5fbd77d6ff6684273b8a740cd6c8d26c57468e41e3ffb`。
- 首词归一化后：add 149,998；remove 1,605；replace/swap 5,368。
- `sample_instruction_probe.py` 对三类分别用 seed `20260828/29/30` 均匀随机 15 条，结果在 `random_instruction_probe.csv`。

## 随机指令观察

- add：随机 15 条只有 2 条是明确实体/配饰添加（蛋糕、圣诞帽）；其余 13 条是 sunset、neon light、winter、grayscale、lighting 等背景/风格/特效。
- remove：随机 15 条里有对象删除，但也有“remove ... and add ... / make sunny”复合编辑；大量候选来自 `InstructPix2Pix` 或 `MagicBrush`，本质更接近静态图序列。
- replace：随机 15 条中存在原子替换，但动态 Pexels 子集高度模板化。2,123 条 Pexels replace 中，726 条是“surfboards → motorcycles + pink ocean”，另有 253 条同类复合指令；随机样本也重复命中这一模式。

候选来源分布进一步说明这一点：

| 首词任务 | Pexels 动态 | InstructPix2Pix | MagicBrush | OpenVid 动态 | 其他静态 |
| --- | ---: | ---: | ---: | ---: | ---: |
| add | 133,132 | 8,576 | 1,337 | 986 | 5,967 |
| remove | 330 | 727 | 541 | 3 | 4 |
| replace/swap | 2,123 | 2,757 | 478 | 1 | 9 |

## 判断

- 如果未来需要大规模混合编辑预训练，InsViE 可以作为宽口径来源。
- 如果目标是本轮的原子 add/remove/replace，它需要先做语义分类、静态来源排除和复合指令过滤，不能从首词总体直接采纳。
- 尤其是 add，随机实体密度太低；replace 虽有潜力，但动态候选模板重复和多操作污染明显，优先级低于 EgoEdit、ReCo 和 OpenVE。

官方来源：[InsViE-1M dataset](https://huggingface.co/datasets/wyh6666/InsViE)、[official code](https://github.com/langmanbusi/InsViE)。数据仓库声明 CC-BY-4.0。
