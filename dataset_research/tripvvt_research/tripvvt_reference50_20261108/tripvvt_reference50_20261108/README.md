# TripVVT 全新 50 组参考图换衣审核

完整解压 ZIP，用 Chrome / Edge 打开 index.html，无需服务器或联网。

- 上衣 17、下装 17、官方 dresses 类 16；固定 seed 20261108，排除此前 40 个真实 target，全包按 target 去重。不是全库自然分布抽样。
- SOURCE 是合成换装视频，GARMENT 是官方单张服装图，TARGET 是原始真实视频；本次未进行编辑模型推理。
- 同步从头播放、同步暂停/跳转，保留原速、完整分辨率、帧率和时长。浏览器视频 H.264/yuv420p/faststart，无音轨；原始媒体保存在包外 originals/。
- 可按服装类别筛选；所有审核项用 1/0/2，默认 1 仅为填写起点，reviewed=false 不计作人工通过。支持备注、CSV/JSON 导入导出和浏览器保存。
- 参考服装匹配和是否实际换衣为独立审核项，特别注意近似 no-op、款式不符和 dresses 类别不准确。多人物场景还需确认编辑对象。
- 页面指令来自本地类别模板，明确引用服装参考；官方 text 仅是场景/目标描述，完整原文保留在 manifest.json。
- manifest.json 为完整来源与语义，inventory.csv 为媒体/哈希/指令总目录，audit.csv 为未审核默认表，REPORT.md 与 summary.json 为类别及实际规格，media_inventory.json 为转码与完整解码记录。
- 官方 train/test 的真实 target 存在重叠，本包只保留其来源标签，不作为干净的泛化评测 split。mask/pose 路径仅保留在 metadata，未下载。本轮不转换 LTX2.5 训练格式或确定混合比例。
- 来源：TripVVT/TripVVT-10K，revision 596f650e5186a9d6a5fe694e3c2a3eccf2ab7dd3。许可记录见本地调研：CC BY-NC 4.0。
