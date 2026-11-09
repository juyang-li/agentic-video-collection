# 官方LTX-2.5日转夜视频编辑复现

两条官方source+prompt（山景、咖啡馆），各运行官方Day-to-Night adapter与不挂adapter的对照，共4输出。是真实视频编辑，不是文字生成或只参考首帧。权重与素材固定revision。采用官方链接中的单阶段distilled工作流：8步CFG1，而非模型卡另一套30步STG建议；不混称。来源、展示视频面板提取与差异见metadata/。

已完成2个官方案例及2个无adapter对照，共4输出。山景出现星空夜景，咖啡馆出现夜间窗外与暖光；裸模型对照仍保持日景。说明正确任务adapter能实现视频编辑，不代表裸模型通用换衣接口。

审核页：[index.html](review/ltx25_official_daynight2_scroll_20261031/index.html)，四列为source、本次adapter、本次bare、官方target。完整prompt、来源及差异保留。ZIP约15MB，已上传百度网盘。

百度交付：`/apps/bypy/lance_review/ltx25_official_daynight2_scroll_20261031.zip`（14,611,212 bytes），远端大小与fs_id已核验，未整包下载回读。收据见`review/baidu_delivery_receipt.json`。
