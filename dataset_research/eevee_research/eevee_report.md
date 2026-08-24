# Eevee 抽查结论

## 结果

- 论文宣称 9,364 对、8,364 train / 1,000 test；发布包实际只有 9,363 个样本目录，CSV 实际为 8,363 train / 1,000 test。差异来自 dresses train：论文 1,564，仓库 1,563。
- 获取：无需授权；5 个分卷共 230,853,254,915 bytes（约 215.00 GiB）。通过 ZIP central directory + HTTP Range 随机抽样，只下载约 122 MiB。
- 随机性：seed `20260823`，完整 9,363 个目录均匀无放回抽 15；14 train / 1 test；10 upper body / 3 dresses / 2 lower body。
- 人工观察：人物相关 15/15；matched garment 与 full-shot/close-up 视频相符 15/15；任务类别严格通过 14/15。
- 发现的坏例：`39538-eevee-0012.mp4` 位于 `upper_body/06652`，但 garment、caption 和两段视频都明确是连衣裙。
- 技术：本轮 full-shot/close-up 原视频均为 816×1088、25 fps、MPEG-4 Part 2，可完整解码；full-shot 3.00–8.64 秒。观察文件已统一转成 H.264。

## 适用边界

Eevee 的真实 target、1088×816 分辨率、服装细节图、文本描述，以及同一服装的 full-shot/close-up 两段视频，质量和细节明显优于 ViViD。

但它仍是 `masked(target) + matched garment -> target` 的重建监督，而不是不同服装的 source-target 编辑。full-shot 与 close-up 是不同拍摄段落，不能逐帧对齐。官方 test CSV 将 Person 与另一条 Garment 交叉配对，只能做无 ground truth 的生成评估，不能当作监督 pair。

最大阻塞不是下载，而是授权：Hugging Face model card 没有 license 字段，GitHub 也没有 LICENSE；论文说明原始图像/视频来自电商网站。在作者书面澄清数据许可前，建议只保留这 15 条研究抽查，不进入训练或再下载全量。

如果许可补齐，可把它列为高分辨率 garment-reference 重建 Tier A/B；仍不得混入纯文本编辑池。训练前先清理 category contamination，并只使用 matched train rows。

官方来源：[dataset](https://huggingface.co/JianhaoZeng/Eevee)、[code](https://github.com/AMAP-ML/Eevee)、[paper](https://arxiv.org/abs/2511.18957)。
