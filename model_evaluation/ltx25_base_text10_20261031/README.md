# LTX-2.5 无adapter文字视频编辑 · 10例

4外观、4文字换衣、2镜头；没有参考图/动作任务。独立申请GPU，复用已下载BF16 distilled权重和环境，不挂Ingredients或任何LoRA。采用与上一轮相同的完整RGB视频guide，只去掉adapter，保留原始指令、seed42、8步与预处理以便比较。该接法用于测试裸权重响应，不能预先当作官方已验证的通用指令编辑协议。冻结详情见metadata/protocol.json。

已完成10/10；4卡排队后换2卡，独立作业183476占用约3分50秒并已释放。无adapter，日志0 patches。中/末帧逐条观察未见明确按文字完成编辑，不能再把上一轮失败主要归因于Ingredients。当前完整RGB-guide接法有无adapter均不展示有效编辑，不能推导其它协议或SFT上限。

连续下拉审核页：[index.html](review/ltx25_base_text10_scroll_20261031/index.html)。三列为原视频、本次无adapter、上一轮Ingredients。包含完整prompt，无评分栏；ZIP约20MB。技术证据与观察在metadata/。

百度网盘已交付：`/apps/bypy/lance_review/ltx25_base_text10_scroll_20261031.zip`（19,939,912 bytes）。远端大小和fs_id已核验，未整包下载回读；收据见`review/baidu_delivery_receipt.json`。
