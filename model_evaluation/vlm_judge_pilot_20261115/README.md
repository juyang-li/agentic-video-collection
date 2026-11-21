# 单轮编辑 VLM 试评

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。

2026-11-15。复用v3_116中10任务×4既有输出；CPU准备、外部API评分，无GPU推理。抽样seed 20261115，分层：hair、hair_color、identity、background、TripVVT、accessory、addition、removal、face、文字clothing各1。最后两项是原计划native两任务的明确配额，保证通用模板覆盖。

中文评价模板继承师兄evaluation_protocol的五类模板；新增instruction_conditioned两份，覆盖原模板不适配的24条任务。原始英文编辑指令不翻译。EA/CP独立调用，1–5整数、score/reason两行；CP不输入reference像素。仅替换面部身份的任务不允许体形/衣物变化。映射、模板原文SHA及本地快照见metadata/prompt_provenance.json；新增模板见prompts/。

16个精确帧索引：0,3,6,10,13,16,19,22,26,29,32,35,38,42,45,48，12fps共同区间。SOURCE使用已冻结prepared PNG；OUTPUT逐帧解码，不插帧。timecard两侧各长边最多1024、保持比例、JPEG95/4:4:4；标签放画面外。reference独立PNG，不裁剪。当前规范化只处理EXIF/RGB/alpha，不额外执行ICC色彩变换。H3使用49帧比较输出。metadata/candidates.json绑定输入、输出和卡片SHA。

评价器：OpenRouter deepseek/deepseek-v4.1-flash-1017（Wafer，thinking关闭）、qwen/qwen3-vl-235b-a22b-instruct（DeepInfra，该端点未列235B Thinking，明确采用Instruct）及openai/gpt-6.1-sol（OpenAI，high reasoning）。关闭provider fallback，保存实际provider/model、usage、延迟、完整原始响应和每次失败。最早四次DeepSeek/Qwen smoke尚未显式锁定provider，实际返回同样为Wafer/DeepInfra；后续均显式固定。按用户最新授权三模型统一使用OpenRouter余额，不将其称为ChatGPT官方credit。

运行：`PYTHONPATH=/tmp/vlm_pilot_deps python scripts/pilot.py prepare`；`python scripts/pilot.py run --judges deepseek qwen --limit 1 --workers 2`通过后去掉limit运行40候选；GPT运行`VLM_GPT_OPENROUTER=1 python scripts/pilot.py run --judges gpt --workers 2`。加`--repeats`执行固定8候选的两次额外重复。GPT在途预算恢复入口为`VLM_GPT_OPENROUTER=1 python scripts/recover_gpt.py`，先等待120秒再串行补跑。依赖Python、Pillow、requests、PyAV。所有入口允许login CPU执行。断点续跑跳过成功项，每次失败完整保存；不能把错误计为低分。运行中不得改动prompt。无网盘上传授权。

试点仅用于人工比较评价器；未经人工验证不能声称正式论文评分可靠。匿名审核包入口review/index.html，结果与重复稳定性见RESULTS.md及metadata/runtime_summary.json，输入和评分完整性验收见metadata/input_check.json、completion_audit.json。后续独立unseen人工验证尚未执行。

执行发现Qwen有时照抄模板示例的Markdown代码框，严格两行parser会拒绝。为避免把格式重试引起的分数变化混入模型选择，`scripts/normalize.py`从保存的attempt中按时间选择首个可确定解析的响应：允许去掉完整的外层text代码框，但内部必须严格为score/reason两行、finish_reason=stop。截断、多分数或额外段落仍拒绝。原始记录完全保留，规范化结果另存metadata/parsed，明确标注format_compliant与parse_source；报告使用该确定性派生结果，并报告严格格式失败率及所有attempt费用。这是解析适配，不是人工改分或让模型重写判断。

用户确认LIJUYANG_ICLR_PERSONAL为其已设置的OpenRouter key后，GPT通过OpenRouter/OpenAI provider完成两次smoke（2026-11-15）；使用`VLM_GPT_OPENROUTER=1`显式选择该路由。用户随后明确授权GPT全量也使用此API，因此本试点三评价器统一使用OpenRouter余额。官方credit绑定由用户稍后处理，不再作为本试点继续执行的前置条件。

GPT首轮基础与重复进程合计12并发触发OpenRouter `402 / in_flight_budget_exhausted`，不是无效密钥，也不能据此判断余额已用尽。首轮成功46项、拒绝66项；所有响应保留。待进程终止后以2并发顺序补跑基础和重复项；成功项不重跑。费用取所有attempt的usage.cost之和。
