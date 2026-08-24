# TripVVT-10K 随机 triplet 结论

## 结论

TripVVT 是目前实际抽查过的外部数据中，最接近“高质量、人物相关、single-task”的候选。随机 15 组全部是人物服装编辑，12/15 是严格可用 triplet；source/target 全部逐帧同规格，观察中身份、动作和背景保持明显好于 OpenVE、Ditto、GOKU 与 Señorita。

但它只适合 **garment-reference-conditioned virtual try-on**。它不是脸、身份、头发或首饰数据，也不是只靠文本 instruction 的通用编辑 pair。只有训练架构能够接收 garment reference（以及按需接收 mask/pose）时，才建议进入主要监督集。

## 方法

- population：官方 train 9,931 + test 100，共 10,031 个 triplet；
- sampling：seed `20260822`，全体无放回均匀随机 15 条，15 个 target 不重复；
- acquisition：只下载命中的 45 个独立文件，共约 159.3 MB；
- review：`SOURCE | GARMENT | TARGET` H.264 comparison 和六帧 contact sheet；
- sample coverage：dresses 7、lower_body 5、upper_body 3。

## 审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 15 | 0 | 0 |
| instruction correctness | 12 | 2 | 1 |
| logical correctness | 15 | 0 | 0 |
| content preservation | 15 | 0 | 0 |
| temporal consistency | 15 | 0 | 0 |
| visual quality | 15 | 0 | 0 |
| technical quality | 15 | 0 | 0 |

严格候选按七项全部通过计算，为 12/15。

## 发现的问题

- 2/15 garment-reference 不匹配：一组参考黑裤但 target 仍是白裤；一组参考吊带连衣裙但 target 是露肩两件套；
- 1/15 source 与 target 都是几乎相同的浅蓝牛仔裤，形成近似 no-op；
- `text` 是全局 target 描述，抽样中会重点描述未编辑区域，不能当作局部换装指令；
- 完整 metadata 的 5,342 个真实 output 被复用于 10,031 个 triplet，而且 test 的 100 个 output 全部与 train 重叠。训练与评估必须按 output/人物重新分组，不能直接沿用官方 split 评估泛化；
- target 是真实视频，但 source 是通过首帧编辑和视频传播生成的换装视频，所以它提供高质量真实 target 监督，不等同于真实 raw-domain 数据。

## 使用建议

若模型支持 garment reference，把审核通过的 TripVVT 作为服装分支 Tier A 数据；若模型是纯文本编辑，则先不混入主池，也不要把 garment image 自动 caption 后当作等价替代。

扩大到全量前先执行四个过滤器：garment-reference/target 一致性、source-target 非 no-op、source 身份与非人物区域保持、时间闪烁。随后按 output/identity 分组重建 split，并把官方 test 与本项目人工观察样本排除出训练。

完整数据约 72.6 GiB、18 个 tar shard，规模可接受；但是否下载全量应等训练接口确认支持 reference image 后再决定。

