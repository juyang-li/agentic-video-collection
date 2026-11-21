# TripVVT-10K 参考图换衣数据审核

## 当前：低变化 10 对阈值校准（2026-11-08）

[低变化审核包](./tripvvt_lowchange10_20261108/tripvvt_lowchange10_20261108.zip)从本地 75 对（新抽查 50＋此前换衣 25）中，按类别对应区域的 RGB 差异中位数升序取 10 对：上衣 3、下装 5、dresses 2。独立 CPU 脚本已复现全部 75 对分数和前 10 名。包内保留完整 source、服装图、真实 target，prompt 位于素材上方，数值默认折叠；审核状态仍为未审核，未冻结自动筛除阈值，未筛除正式训练数据。

可复用程序为 [score_pair_similarity.py](./score_pair_similarity.py)，参数、输入清单、两次实测、完整 75 对分数与复跑说明归档在 [校准目录](./tripvvt_similarity_calibration_20261108/README.md)。4 并发实测 75 对分别耗时 38.64 秒、47.82 秒，对相近规格的 1K 对线性估计为 8.59–10.63 分钟，实际可预留约 10–15 分钟，不含下载，受本地 I/O 与 CPU 负载影响。无需视觉模型或 GPU。

后续正式池目标已更新为训练 10,000＋验证 500＋测试 500，总计 11,000 条，筛除门槛等待本轮人工审核后确定。低变化批次使用独立目录和审核版本，不覆盖此前 50 对或 25 对审核包及标注。

低变化包已上传 `/apps/bypy/lance_review/tripvvt_lowchange10_20261108.zip`（99,994,376 bytes），ZIP 与同名 SHA256 sidecar 的远端大小均已核对。20 次视频播放、10 张参考图、prompt/排名、筛选与标注往返通过离线 Chrome 验证；包内脚本对 10 对审核媒体的复算与原始媒体分数一致（误差 <1e-9）。回执保存在该批次 `baidu_delivery_receipt.json`。

## 当前：全新 50 组三元组抽查（2026-11-08）

[离线审核包](./tripvvt_reference50_20261108/tripvvt_reference50_20261108.zip)包含上衣 17、下装 17、官方 dresses 类 16 组。固定 seed `20261108`，排除历史 15 组及 10 月换衣 25 组覆盖的 40 个真实 target，全包 50 个 target 均不重复。类别配额用于展示，不能外推全库自然占比。

完整解压后打开 `index.html`，每组同时展示合成 source、官方单张服装参考图和原始真实 target；可按类别筛选、同步播放/暂停/跳转，并通过 CSV/JSON 导入导出审核。记录按类别分组、组内 ID 自然排序。新增独立审核项“目标服装是否匹配参考图”和“是否实际发生换衣”，默认项为 `1`、`reviewed=false`，不计作人工通过。

实际规格均为 720×1280、24 fps、121 帧、约 5.042 秒，100 个浏览器视频已完整解码、50 张参考图已验证。来源、官方 caption、本地参考图条件指令、mask/pose 路径和媒体哈希保留在清单；mask/pose 本轮未下载，未运行编辑模型或质量 judge，也未加入正式训练池。详细内容见 [REPORT.md](./tripvvt_reference50_20261108/tripvvt_reference50_20261108/REPORT.md)。

生产入口为 `build_review50.py` 的 select/download/build/package/upload 阶段，浏览器验收为 `verify_review50.py`。本轮素材、冻结清单、日志和交付回执位于 `tripvvt_reference50_20261108/`，原始媒体保留在其外层 `originals/`。远端交付目录为 `/apps/bypy/lance_review/`。

已上传 `tripvvt_reference50_20261108.zip`（570,273,432 bytes）及同名 SHA256 sidecar，两者远端大小均已核对；未整包下载回读。ZIP SHA256：`2e84d959b553972f947aa54415c730027c00831bb2f196feda79374f34eca143`。最终凭据见本轮 `baidu_delivery_receipt.json`。

## 历史：随机 15 组三元组抽查

直接观察：

- `tripvvt_random_pairs_15_h264.zip`：适合下载到本地；
- `examples/comparison/`：15 个 `SOURCE | GARMENT | TARGET` H.264 MP4；
- `random_pairs.csv`：完整随机抽样记录；
- `tripvvt_audit.csv`：人工审核；
- `tripvvt_report.md`：结论。

抽样 population 是官方 train 9,931 加 test 100 的全部 10,031 个 triplet。seed `20260822`，无放回均匀随机 15 条；命中 14 条 train、1 条 test，15 个 target 均不重复。

这里不是纯文本视频编辑。输入由 source video、garment reference 以及可选的 human mask/pose 组成；comparison 把 garment 放在中间，便于判断 target 是否正确对应参考服装。

压缩包只保留观察所需的 comparison、说明、manifest 和 audit；重复的原始 role 视频与 garment PNG 在产物验证后删除。

