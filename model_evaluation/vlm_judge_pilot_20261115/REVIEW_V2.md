# 实名审核包V2

入口为review_v2/index.html，交付ZIP为review_v2.zip。10任务和40输出均复用冻结试点，三裁判实名固定顺序，生成模型固定LTX Base、A849、VACE、H3 PDD8。主页面不展开送评帧，每个结果链接到独立evidence.html，只显示所选结果的16张原始timecard。保留完整source入口、TripVVT参考及GT/caption。

全部51个主页面播放器均使用H.264 Main level4.1、8-bit yuv420p、CRF16、faststart浏览副本；保持各原始分辨率、49帧12fps和PTS。不裁剪、不插帧、不去噪。每个副本生成后完整解码验证49帧与时间戳；原始文件及评分输入保持原样。display_manifest.json记录原始/副本SHA和路径。

package_v2.py核对240项基础分数和原始理由、51播放器文件、40输出、10任务及页面本地引用。跨浏览器实播验收按用户后续明确要求跳过，不能宣称Firefox/Chrome实播通过；状态在metadata/review_v2_validation.json。没有新增VLM API调用或GPU推理。

上传目标：/apps/bypy/lance_review/vlm_judge_pilot_20261115_review_v2.zip，上传状态与成功回执保存在metadata/baidu_upload_v2.json。旧版ZIP及其远端交付保留。仅上传ZIP，不增加远端哈希/大小检查。
