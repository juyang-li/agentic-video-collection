# OpenVid T2V 人工审核包

1. 解压归档后，用 Chrome 或 Edge 直接打开 `index.html`。
2. 每条检查 caption 对应、画面质量、时间稳定性和是否适合作为 T2V 训练数据。
3. `1/0/2` 分别表示通过、不通过、存疑；备注可选。
4. 完成后点击页面顶部“导出 CSV”或“导出 JSON”。

本包共有 50 条：human 15、vehicle 10、animal 5、scene 10、object 10。视频是仅供浏览器审核的 H.264 CRF17 代理；manifest 同时保留 OpenVid source key、原始视频 SHA256 和代理 SHA256。
