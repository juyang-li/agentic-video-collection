# 原bench纯文本编辑复测：先一条，最多四轮

先用appearance_002加黑色墨镜验证，目标描述prompt与空prompt配对对照。前几轮用distilled，本次测试无adapter的dev权重，30步正常CFG，并尝试降低source attention或source latent加噪起点。四轮定义在metadata/experiment_protocol.json。每轮人工查看后才决定下一轮；出现明确目标编辑再以同一配置扩展原10条，不把画面漂移当成功。完成检查后按用户要求自动上传百度。

本轮四次均已实际运行（dev无adapter），共8输出。第三轮有墨镜但源人物场景漂移，其余未见墨镜；未通过原视频编辑检查，按四次边界停止，未扩十条。完整49帧面部检查与各轮三列对照保留。

审核包已自动上传百度：`/apps/bypy/lance_review/ltx25_text_edit_retry_scroll_20261031.zip`，2,724,384 bytes，远端大小与fs_id核验通过。收据见`review/baidu_delivery_receipt.json`。独立作业183514已释放。
