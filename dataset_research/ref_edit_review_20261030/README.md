# 单轮参考编辑数据审核

交付 25 组 TripVVT 换衣三元组与 25 组独立动作参考输入对。动作无编辑后 GT，尚未运行编辑模型。

- 换衣：上衣 9、下装 8、官方 dresses 类 8。保留 source、单张服装图、真实 target；source 是合成换装视频。
- 动作：挥手 7、拍手 6、全身转身 6、行走 6。真实 source 来自 TripVVT 未修改视频，独立动作参考来自 HMDB51；19 个 source 被复用于 25 对，分组关系显式记录。
- 审核：中文离线网页、同步播放、来源和质量提示、1/0/2 标记、人工审核状态、CSV/JSON 往返导入导出。已嵌入 SIL OFL 的 Noto 中文字体子集。
- 验收：75 个 H.264/yuv420p/faststart 视频完整解码；50 张卡片、100 次视频播放和 25 张参考图加载；无外部请求；CSV/JSON 多行备注往返通过。ZIP 内 109 个文件的 SHA256 均逐个复核。

百度网盘目录：`/apps/bypy/lance_review/`

文件：`ref_edit_clothing25_motion25_review_20261030.zip`（167,519,176 bytes，约 160 MiB）。解压后打开 `index.html`。

ZIP SHA256：`84c3b4f2655bed9880c7deb3818e79c5941218a707fb09243b9a9895c0ec3ec0`。上传和远端核验的最终结果见 `metadata/delivery_receipt.json`，校验 sidecar 与 ZIP 同名。

运行 prepare_assets.py、inspect_assets.py、build_review.py；浏览器验收通过后运行 build_review.py --finalize。详见包内 README 与 manifest。
