# MiniMax H3 本地 Ref2VA：90条非动作对比

运行准备中；仅本地 H3-Base Ref2VA，无官方 Context-IR，无付费API。复用原百样本中的90条非动作、完整原指令和参考身份。H3按原生768短边、24fps、107帧、50次去噪更新、seed42生成；审核共同0–4秒，保留原生完整输出。

SOURCE只使用原协议0–4秒的49个时刻，24fps重复取帧、末端保持至107帧，等比例补边到1344×768或768×1344。中间MP4为libx264rgb CRF0，107帧全部逐像素核对。GT和target caption不进入推理。已有场景/镜头首帧参考按冻结清单照用。

现有testqueue分配199810；隔离SGLang环境，TP2＋Ulysses2四卡协同一条，单请求并发。原生BF16/FP32，lossless质量，禁用torch.compile、近似注意力、跳步缓存、量化和加速LoRA。代码和权重revision、请求、时间映射、安装日志均保留。

最初网络错误为计算节点继承了失效localhost代理；已解除代理并验证直连。安装使用上游SGLANG_BUILD_RUST_EXTS=none，关闭可选Rust扩展；不改变扩散模型路径。三个CoinVE原件因其他目录整理已迁往11K canonical目录，新旧SHA256完全一致，映射见metadata/source_relocations.json。

历史基座/A849媒体正在由已交付结果ZIP恢复，作为本次比较输入；这不是上传后的远端回读校验。本次交付只上传ZIP，以上传工具成功回执为准。尚未完成推理、审核或交付，不代表人工质量通过。

上游兼容修复补丁见metadata/sglang_build_portability.patch：C++模板函数命名查找、原生VisualVAE嵌套权重完整性识别，以及缺失memfd_create时使用原有普通解码回退。模型82个文件的实际哈希已全部核对官方冻结revision，未改权重。

精度说明：DiT按原生BF16/FP32混合路径；VisualVAE以FP32权重驻留，其解码按官方原生FP16 autocast执行，未做额外精度降级。当前CLI不接受quantization=bf16，实际采用quantization=None，保留配置中dit_precision=bf16。
