# Señorita 获取与抽样记录

## 官方数据

- 仓库：`SENORITADATASET/Senorita`
- 固定 revision：`6e2eac59caf6321353d2712e31815fda2ff573cb`
- 仓库 API 总量：约 1.60 TB
- `object_swap`：177,336 对；官方目录约 76.2 GB，共 187 个独立 `tar.gz`
- 完整 metadata：55,487,962 bytes，SHA256 `c4d46c8fd44ddd7b04f16f1eedc1fd83720a55ee9732b7239e17622b860527d9`
- 抽中的 `144.tar.gz`：458,796,295 bytes，SHA256 `52edec7713378f9b27304547987fb5a7a6c97d41441957c8b87baf908ce5f27d`

官方页面：<https://huggingface.co/datasets/SENORITADATASET/Senorita>；项目页：<https://senorita-2m-dataset.github.io/>。

## Pair 方向

manifest 中使用非 `_org` 文件作为 source、`_org` 文件作为 target。15 条逐一核对后，方向都与 instruction 一致；例如 source aircraft 到 target ship 对应 `Trade aircraft for ship.`。

## 技术观察

- 15/15 source 与 target 都可完整解码且规格精确一致；
- 全部为 592×336、33 帧、8 fps、4.125 秒；
- comparison 已转成 VS Code 和常见播放器更兼容的 H.264/yuv420p MP4；
- 原始 role MP4 只用于生成核验产物，完成后删除，避免重复占空间。
- 本地观察包：13,538,783 bytes，SHA256 `b7821cfbc1c1864f88d67749ca0279b1aeb581128f228be19bf765192e0d9def`；解包验证为 33 个文件。

## 复现

1. `sample_random_pairs.py` 固定 metadata 哈希、revision、seed 和抽样过程；
2. `extract_random_pairs.py` 只从命中的压缩卷解出 30 个 role 文件；
3. `prepare_random_pairs.py` 生成 inventory、comparison 和 contact sheet，默认不覆盖已有 audit。
