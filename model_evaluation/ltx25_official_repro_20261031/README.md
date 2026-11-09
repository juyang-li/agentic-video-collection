# LTX-2.5 官方纯文生视频复现

用户最新要求先试官方纯文本。先运行官方模型卡两个原始prompt（金毛犬跑草地、海上日落），使用官方单阶段distilled流程，无任何adapter、source视频或参考图。用于排查基础生成链路，不当作原视频编辑。暂缓Ingredients和Day-to-Night案例。

模型与官方工作流固定revision，seed42、8步、CFG1，768×512、121帧24FPS。Comfy官方节点执行生成，I/O沿用项目编码器；同时采样音频但此包只导出无声视频。

已完成2/2。单卡作业183496于14:49:58获配、14:52:22释放，两段各121帧视频完整解码及浏览器播放通过。金毛犬与海上日落均呈现相应画面，说明这两条基础纯文本生成链路正常；不代表原视频编辑接口验证成功。

审核页：[index.html](review/ltx25_official_t2v2_scroll_20261031/index.html)。ZIP约4.3MB；已上传百度网盘。

百度网盘：`/apps/bypy/lance_review/ltx25_official_t2v2_scroll_20261031.zip`。4,323,591 bytes，远端大小与fs_id已核验，未整包下载回读；收据：`review/baidu_delivery_receipt.json`。
