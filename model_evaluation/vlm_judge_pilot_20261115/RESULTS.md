# API试评执行结果

三评价器使用相同10任务×4既有输出、独立EA/CP评分，另在8候选上各增加两次独立重复。所有费用均为OpenRouter响应usage.cost之和，包含失败重试中已计费的响应；不称为官方ChatGPT credit。

|评价器|基础完成/80|三次完整案例EA/CP（各8）|全部attempt费用USD|基础平均/p95延迟秒|1500候选基础费用外推USD|
|---|---:|---:|---:|---:|---:|
|deepseek|80|8/8|0.0459|10.57/17.16|1.49|
|qwen|80|8/8|0.5555|12.79/18.34|14.80|
|gpt|80|8/8|5.4819|14.21/22.63|189.97|

1500候选=3000次基础请求。费用与32并发延迟均为当前样本的线性外推，不是报价或完成时限保证；不包含大规模重跑。缓存命中、图像尺寸、reasoning长度和账户并发预算会影响结果。当前OpenRouter账户曾触发402/in_flight_budget_exhausted，GPT恢复时按provider的120秒结算窗口串行补跑，实际墙钟时间不能只按模型延迟计算。

Qwen的部分回答包含模板示例的外层Markdown代码框。原始响应保留；报告使用首个完整、可确定解析的响应，只允许去完整外框，内部仍须严格两行且finish_reason=stop。具体format_compliant和parse_source见metadata/parsed。不得将格式通过等同于视觉判断正确。

匿名离线审核页review/index.html包含完整prompt、source/reference、40输出、三裁判分数理由和实际16帧timecard。inventory.json绑定匿名媒体和SHA；揭盲映射仅保存在包外metadata/reveal.json。本轮尚无独立人工评分，不能认定某裁判更准确。

当前完成状态以metadata/completion_audit.json为准；完整输入协议、12份中文prompt、原始API响应、请求参数及哈希均保留。未成功项不计为零分。
