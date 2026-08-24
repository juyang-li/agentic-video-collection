# OpenVE `local_add` / `local_remove` 可获取性实测

## 结论

任务形式完全符合当前目标，但官方 split-gzip 的存储顺序不利于浅层随机抽查。本次固定下载 `local_add.tar.gz.00` 的前 4 GiB，只形成 4 个完整 add pair 和 3 个完整 remove pair，达不到每类随机 15 组；因此没有把不足 15 组的结果伪装成正式随机样本，也没有生成零散 ZIP。

## 实测口径

- revision：`d2c134f01c49d22efc2e0d9aefd798772f039a79`。
- `local_add.csv`：400,498 行，SHA256 `8ea018c7b4d8559a503ca341a5d150b77cf407f657149ced4c9b7593fac387eb`。
- `local_remove.csv`：337,463 行，SHA256 `227694ee059bf198ff26b65359c36a03ab31903665b61d9814ba99553e80f5c3`。
- 固定前缀：4,294,967,296 bytes；组合 SHA256 `79274745864db51174b091d7c868340e7351c593df917d6e13743ec442396943`。
- 前缀内被证明完整的 MP4：3,268 个。
- 两端都落入此前缀的 metadata 行：`local_add` 4 条，`local_remove` 3 条。

这里的 3 个 remove 是同一批资产的反向边：两个人物删除和一个玩具熊删除；4 个 add 中有三个人物添加和一个玩具熊添加。这恰好说明任务语义是对的，但数量太小，不能据此估计质量比例。

## 为什么 4 GiB 仍不够

官方把视频放进连续 gzip tar 分卷，文件名近似哈希，source/target 在归档中通常相距较远；只读压缩前缀时，单个 MP4 很多，但一对两端同时出现的概率很低。另外，`local_remove.csv` 的部分行实际引用 `local_add/` 资产，任务分类与物理归档目录并非一一对应。

官方数据卡给出的全量体积约为 `local_add` 760 GB、`local_remove` 453 GB。继续把前缀扩大到 8–10 GiB 仍无法保证 15 对，而且会把额度消耗在不可随机访问的 gzip 扫描上，所以本轮停止扩张，把下载额度用于可形成明确总体的独立数据集。

## 保留的证据

- `local_add/downloadable_population.csv`、`local_remove/downloadable_population.csv`：实际可配对行。
- 两目录的 `prefix_members.txt`：3,268 个完整成员清单。
- `local_add/sample_random_pairs.py`、`local_remove/sample_random_pairs.py`：同一前缀下的复现入口。

官方来源：[OpenVE-3M dataset](https://huggingface.co/datasets/Lewandofski/OpenVE-3M)。数据卡明确将 Local Add、Local Remove 列为独立 spatially-aligned 类别，并声明 CC-BY-NC-4.0。
