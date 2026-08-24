# TripVVT 获取与抽样记录

## 官方数据

- 仓库：`TripVVT/TripVVT-10K`
- 固定 revision：`596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3`
- 许可：CC BY-NC 4.0
- train：9,931 行，SHA256 `1f1d1368f182836d697c0a0daa99169b45b61c2bf1a4ce7cef57b629d8538b83`
- test：100 行，SHA256 `590d5b1ed925641f616238cfab9ad69318c1032f29d0c26af34916a2921db62b`
- README：SHA256 `5fe8da8cc033dffc00e6c56ccc872205991fd9fb2bc4a4f25cd9df6e69784ab9`
- 类别：dresses 3,379、lower_body 3,331、upper_body 3,321
- 官方推荐完整获取：18 个 tar shard，约 72.6 GiB；仓库同时保留解压后的独立文件，因此仓库 API 总量会近似翻倍。

官方来源：[dataset](https://huggingface.co/datasets/TripVVT/TripVVT-10K)、[paper](https://arxiv.org/abs/2604.27958)、[project](https://shaodingbao.github.io/TripVVT/)。

## Pair 语义

论文采用 reverse design：`input/` 是从真实视频换装后合成的 source，`groundtruth/` 是未经修改的真实视频 target，`input_cloth_ch/` 是从真实 target 服装重建的商品图。这样 target 监督是真实画面，但 source 仍是生成视频。

metadata 的 `text` 是全局场景和 target outfit 描述，不是可直接执行的编辑 instruction。本抽查另外生成了明确标注 reference conditioning 的规范化 instruction；训练时不能丢掉 garment reference，只用这段泛化文本。

## 随机样本

- seed：`20260822`
- population：10,031 个 triplet
- selected indices：`8640, 1754, 2095, 9355, 2361, 2326, 31, 4410, 5577, 3759, 1178, 8113, 8303, 9955, 331`
- 15 个 source、15 个 target、15 张 garment 共 159,333,408 bytes；
- 全部 source/target：720×1280、121 帧、24 fps、5.042 秒，规格精确匹配并可解码。
- 本地观察包：35,772,063 bytes，SHA256 `17fda80e4dd31677477e5548db6a534ffa14b9e9a2efaf21555ffbe11b3cc317`；解包验证为 33 个文件。

## Split 检查

完整 metadata 只有 5,342 个不同 `groundtruth/output` 视频；多数 output 对应两个不同服装区域的 triplet。官方 test 的 100 个 output 路径全部也出现在 train，只是对应不同的 input/cloth_type。因此官方 split 是 triplet-level holdout，不是原始视频级 holdout；本项目若用于泛化评估，必须按 `output`（最好再按人物身份）重新分组切分。
