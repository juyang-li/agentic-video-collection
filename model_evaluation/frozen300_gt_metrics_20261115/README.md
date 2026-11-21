# frozen300：四模型对GT的全帧指标

仅评价模型输出与GT，SSIM↑、PSNR↑、LPIPS↓；不计算SOURCE比较、不重跑推理、不新增上传。四模型为本地LTX2.5 distilled＋V7（8步）、dev＋A849＋V7（40步），及师兄H3-PDD8、VACE-LightX2V4加速版。后两者不是普通50/40步基线。

冻结HF仓库 `X-GenGroup/frozen300-h3-vace-accel-20261114` @ `0f999ecc4cbdebac8165318b1274686b7912466f`。只下载results视频/JSON、汇总、README和校验/几何代码。视频共12,727,220,429字节。四路下载，分块Range续传，SHA256校验发布，token不写入日志。

本地GT及冻结输入来自相邻 `frozen300_20261113/package`，input SHA256为 `386caf3ca11ff5ad59b59ceac16088e24864c913fb844015cc3a08e8f33614c0`。按稳定ID/UID和输入身份匹配。全部冻结12fps帧：每模型31,212帧，合计124,848对。GT使用冻结sample_frame_indices，输出按实际PTS最近时刻取帧，平局取后帧；H3正常取2i帧，额外原生尾帧不参与。无插帧、时间优化、光流配准或补长。

按原始宽高与协议contain画布推导固定padding，去除协议黑边，不能识别黑色像素再裁剪。有效画面1280×720或720×1280；H3有效画面缩放到同尺寸，GT原件直接缩放一次，Pillow LANCZOS。视频PyAV解码为RGB，native metadata不做自动颜色匹配。

PSNR采用RGB[0,1]逐帧MSE后计算dB。MSE0记录null和psnr_infinite=true，不封顶。SSIM使用scikit-image0.26.0，11×11高斯窗sigma1.5、data_range1、population covariance、RGB通道平均。LPIPS使用官方lpips0.1.4 AlexNet version0.1，FP32、eval、输入[-1,1]，禁止AMP/TF32、缩到256或分块拼接。每帧→视频均值→视频等权均值，不跨视频按帧数加权。

独立runtime环境复用现有Python基础包，显式PYTHONNOUSERSITE=1；torch2.10.0独立安装，不改变共享推理环境。依赖/指标权重哈希在GPU入口记录，协议见metadata/protocol.json。

流程：download.py四路下载并校验→controller.py后台等待下载成功、运行CPU测试→提交run.sbatch，仅testqueue单张H200、16CPU、128GB、8小时。GPU作业内12线程计算CPU指标，LPIPS batch4，OOM仅降2/1，不降分辨率。六条×四模型小测先运行：identity_01、whole_outfit_01、action_01、coinve_06、coinve_11、tripvvt_01；落盘smoke.json耗时估算与首中末GT/输出并排预览，之后继续全量。预览用于几何/时刻审核，不作自动编辑通过判定。

每条分数连同逐帧/取帧/裁剪审计保存在scores/<model>/<id>.json；单条失败立即停止GPU作业，metadata/failure.json保留原因，不重跑生成。再次启动可在输入/输出/协议身份完全一致时跳过已完成指标。下载失败写日志，controller检测进程退出但无完成回执会停止，不自动提交GPU。

状态：metadata/download.json为下载成功回执；submission.json为GPU提交；run_status.json为评价进度；failure.json/controller_failure.json为失败；completion.json为全部1200条完成。提交或下载中均不算评价完成。logs保存bootstrap/download/controller/Slurm日志。

完成后report/REPORT.md提供主表，summary.csv按总体/数据集/任务分组，videos.csv逐视频，frames.jsonl逐帧。只有1200条全部成功才产正式完整汇总；失败先修复后续算，不能用不齐全样本冒称300条排名。三指标反映接近单一GT程度，不直接代表指令成功或时序质量；动作/镜头多解、背景占比、编码损耗影响解释。

CPU已测试固定padding、同图SSIM/无限PSNR、独立PSNR公式、构造24fps视频取偶数帧及尾帧排除、错误画幅拒绝。GPU入口另测LPIPS身份/亮度扰动/对称性/batch一致性，记录实际依赖和权重身份。全部结果只保存在本地，不自动上传。

## 执行调整：CPU先行

按用户新要求，cpu_stage.py在登录节点4线程计算SSIM/PSNR，本地两模型先行，同时继续下载。远端下载完成后CPU阶段接着处理H3/VACE。结果在cpu_scores/，进度在metadata/cpu_progress.json。controller等待CPU全部完成后才申请单卡，GPU入口复用CPU分数，仅进行LPIPS及其必要的数据读取。不会为了SSIM/PSNR占用GPU；不落盘全部规范化无损帧以免耗尽空间。原计划CPU指标在GPU分配内运行的安排已替换。

按用户指定现有分配的CPU可用性核实后，CPU阶段已迁入testqueue现有分配202896的独立step202896.2（16 CPU、128GB、gres=none、12线程），释放登录节点CPU worker。旧44组分数经原合同校验复用；新step已继续产生结果。执行位置/线程数变化及源码变更证据见metadata/cpu_execution_migration.json，评价数值协议及其哈希保持不变。

## 全量评价完成

四模型各300条、合计1200组/124848帧对的SSIM、PSNR、LPIPS已完成，无失败。CPU阶段在login与现有202896分片并行完成；LPIPS作业204990于testqueue单H200正常退出，COMPLETED/0:0，作业耗时01:14:08。六条×四模型小测及独立最终覆盖/逐帧均值/分组均值校验通过，最终SSIM/PSNR与此前CPU表一致。结果见report/REPORT.md、summary.csv、videos.csv、frames.jsonl；审计见metadata/completion_audit.json、report_artifacts.json。未上传新包，未重跑推理。
