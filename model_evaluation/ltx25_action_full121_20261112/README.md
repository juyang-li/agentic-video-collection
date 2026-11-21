# Action 完整时长复测

20条输出、10条冻结历史样本。121帧、12fps、40步、seed42，原横竖幅与1280×736/736×1280分辨率。source及motion复用身份、SHA256、时间轴和tensor核对后的训练缓存（16 latent帧）。GT仅供审核。

与旧版协议差异：49→121帧、4→10秒、7→16 latent帧；训练端点规则不变。未并列旧结果，不宣称效果改善或因果。技术检查50段视频完整解码及页面媒体链接通过，不代表人工质量通过。

实际计时与显存（bytes，PyTorch峰值分配）：

```json
{
  "models": {
    "ltx_base": {
      "denoise_seconds": 6574.438161442988,
      "decode_seconds": 31.683451188728213,
      "peak_allocated_bytes": 51805587456
    },
    "a849": {
      "denoise_seconds": 6595.880041694269,
      "decode_seconds": 33.01456691324711,
      "peak_allocated_bytes": 51805587456
    }
  },
  "generation_wall_seconds": 5409.207759618759,
  "package_seconds": 270.52454710006714,
  "elapsed_before_upload_seconds": 5750.066025257111
}
```

原始日志、逐条时间及显存见logs/和outputs/。运行仅接入testqueue现有分配199810；首测两个模型action_01后扩展。结果与模型环境保留，无清理、提交或推送。

网盘路径：/apps/bypy/lance_review/action_full121_base_vs_a849_scroll_20261112.zip。上传状态以 review/ 中上传器回执为准；仅上传ZIP，无远端复核。

上传完成：上传器返回 `complete`。从启动至上传完成约 97.7 分钟；其中生成墙钟 90.2 分钟、打包 4.5 分钟。两模型每卡 PyTorch 峰值分配约 48.25 GiB；运行中 nvidia-smi 观测每卡约 53.1 GiB。完整回执见 `review/action_full121_base_vs_a849_scroll_20261112.baidu_upload.json`。
