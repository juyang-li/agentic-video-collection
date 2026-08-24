# ViViD 抽查结论

## 结果

- 官方发布：9,700 对，train 7,759 / test 1,941；三类为 dresses、lower body、upper body。
- 获取：无需授权；单个 32.06 GB ZIP。通过 ZIP central directory + HTTP Range 从完整总体随机抽样，只下载约 20 MiB。
- 随机性：seed `20260823`，完整 9,700 对均匀无放回抽 15；10 train / 5 test；8 upper body / 5 dresses / 2 lower body。
- 人工观察：人物相关 15/15；garment 与 target 匹配 15/15；严格审核 15/15。
- 技术：本轮 15 个 target 全部为 H.264、624×832、25 fps，时长 1.72–8.68 秒；全部可完整解码。

## 适用边界

优点是 target 为真实棚拍视频，服装参考、mask、DensePose、masked video 齐全，人物/背景/动作天然保持。数据集卡标注 Apache-2.0。

但监督本质是 `masked(target) + garment -> target`，不是不同服装之间的 source-target 编辑；且场景高度棚拍化。它只适合支持 garment reference 与 mask conditioning 的服装重建分支，不应混进纯文本原子编辑池，也不能补脸、身份、头发或首饰任务。

相较 TripVVT，ViViD 的随机质量更整齐，但 TripVVT 的 source 确实穿着另一套衣服，更接近真实换装编辑。建议优先级：TripVVT 高于 ViViD；ViViD 可作为稳定的重建辅助监督。

官方来源：[dataset](https://huggingface.co/datasets/alibaba-yuanjing-aigclab/ViViD)、[code](https://github.com/alibaba-yuanjing-aigclab/ViViD)、[paper](https://arxiv.org/abs/2405.11794)。
