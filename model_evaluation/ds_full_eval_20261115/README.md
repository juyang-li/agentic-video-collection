# DeepSeek视频编辑评测：执行与交接README

> 清理交接（2026-11-21）：本目录有成功上传回执的本地ZIP已清理；原输入／输出、评分和回执保留，历史ZIP可按回执从百度网盘恢复。 review/media浏览副本已清理（VLM另含review_hair24/media），本地审核页需重建或恢复后播放；实际送评证据保留。

**交接状态（2026-11-16）：全部4748项评分、汇总、技术验收、40任务审核ZIP和百度上传已完成。用户确认的当前16份Prompt保持不变，未继续针对自建集CP或模型排名修改。** 本文命令均从项目根目录执行。先读根目录AGENTS.md、任务描述.md；审核交付遵循DATASET_AUDIT_GUIDE.md。当前无需重新启动流水线；后文运行命令用于复现或必要续跑。

416任务：冻结300使用六模型（新增Real-only step232、Synthetic-only step600），V4自建116使用四模型，共2264对视频；IF与CP独立调用，共4528次基础评分。20任务重复一次，共220次稳定性评分，总计4748次。18任务小测及100任务检查点均包含在基础评分中；扩展后检查点共1088次。

## 当前协议

- Judge：OpenRouter `deepseek/deepseek-v4.1-flash-1017`，仅 `wafer/us`、`mxfp4`，关闭路由回退与thinking，temperature=0，max_tokens=800。实际返回模型、provider、费用与原始响应逐次保留。
- 输入只包含英文原始指令、SOURCE/OUTPUT同步配对帧，以及IF所需的有序参考。CP不提供参考像素。Judge看不到编辑模型名、GT、caption或人工评价。
- 固定12fps时间轴，K=max(16,ceil(4*(N-1)/12))，端点包含、最近帧向上取整；10秒40点，4秒16点。每侧长边上限1024，保持比例，JPEG95、subsampling=0。静态参考独立PNG；动作参考按同一冻结时间点单独排列。
- CP v2：非目标内容充分保持、只有可忽略的微小差异可以给5。编辑失败/目标方向错误本身不扣CP；低于5须有具体非目标损失。8类CP统一调整，IF未修改。原版和新版提示词及哈希保留于metadata。
- CP v3只修订镜头类：新视角显露的背景、原地标不再入镜本身不算场景替换，须有可对应内容的具体矛盾。100任务诊断后，六模型的36个镜头CP旧结果统一归档并重评；其他提示词与分数保持。最终4748条评分之外，这36条历史评分及所有重试费用也计入总花费。校准集与未参与本次提示词调整的样本分别汇总。
- 100任务检查点：冻结300中72任务×六模型、V4中28任务×四模型；分数据集与维度计算A849相对H3、VACE的配对差值。仅根据可见证据修正评分歧义，统一重评所有模型；实际性能差距保留。若使用检查点修改提示词，这100任务作为校准集单独报告。
- CPU在login执行：4解码进程，每解码器1线程；API初始8并发，当前上限16，限4次派发/秒。错误时并发减半、最低4，恢复32次成功后逐级增加4。32并发曾触发较多上游图片忙/超时，因此上限已调回16。用户已取消人为金额上限和余额保留线；全部尝试费用均记录，实际API余额不足仍会返回错误。

## 运行

依赖Python、httpx、PyAV、Pillow；当前PyAV安装目录为`/tmp/vlm_pilot_deps`。密钥由`LIJUYANG_ICLR_PERSONAL`读取，不写文件或日志。

```bash
PYTHONPATH=/tmp/vlm_pilot_deps python model_evaluation/ds_full_eval_20261115/scripts/engine.py
```

已有成功结果会续用。100任务后自动停下，由代理完成诊断、写入`metadata/checkpoint100_review.json`后续跑，无需用户重复批准。修改提示词必须另存版本并重评受影响维度，不能混用版本。

`extend_models.py`等待现有评测进程释放锁，再保存原清单快照并加入两个消融模型；不修改师兄的输出目录。已有四模型抽帧与成功结果保持不变，缓存仅追加缺失模型。首次扩展需先运行该脚本。新增模型登记及checkpoint证据位于metadata/ablation_models.json。

## 报告限制

冻结300和V4自建116分别汇总。两套数据的LTX/VACE配置不同；V4部分TripVVT经过看输出后选样，不能声称与冻结测试集同等无偏。审核交付为完整结果表和40任务ZIP；主页面展示视频、独立附页展示Judge实际抽帧证据，不额外进行细致浏览器兼容检查。

## 接手快速入口

```bash
cd /scratch/prj0000000275/jcy/ljy/video_collection
export VLM_EVAL_DIR="$PWD/model_evaluation/ds_full_eval_20261115"
export PYTHONPATH="/tmp/vlm_pilot_deps${PYTHONPATH:+:$PYTHONPATH}"

# 先确认现有进程，不重复启动队列
pgrep -af '[p]ython.*ds_full_eval_20261115/scripts/(finish|engine|extend_models)\.py'
tail -n 15 "$VLM_EVAL_DIR/logs/full_engine.py.log"
cat "$VLM_EVAL_DIR/metadata/progress.json"
```

`successful_this_process`只计本次进程新完成项，不是全任务累计。日志短时不更新不代表进程结束；progress和历史failure文件也不能单独证明进程生死。评分入口有文件锁，但整条打包上传流水线仍只启动一份。

确认旧流水线已结束、需要续跑时，默认入口：

```bash
python -u "$VLM_EVAL_DIR/scripts/finish.py" \
  > "$VLM_EVAL_DIR/logs/full_pipeline_resume.log" 2>&1
```

该命令依次执行评分续跑、汇总、必要验收、40任务审核ZIP和已授权百度上传，包含真实API调用。已有合规成功评分会复用。本任务全部是CPU准备和远程API，直接在login运行，不申请GPU、不提交Slurm。

## 模型与数据来源

任务指一份source＋prompt＋references；视频对指该source与某个模型输出。冻结集1800对、自建集464对，合计2264对。基础评分4528项、重复220项，最终有效评分4748项；被替代的旧36项镜头CP和错误重试不混入最终有效表，但费用保留。

| 数据集 | model_id | 含义 |
|---|---|---|
| 冻结300 | ltx_distilled_v7 | LTX2.5蒸馏基座，v7参考协议 |
| 冻结300 | a849_v7 | A849，v7参考协议 |
| 冻结300 | h3_pdd8 | H3 PDD8，复用冻结12fps对齐媒体 |
| 冻结300 | vace_lightx2v4 | VACE LightX2V 4-step |
| 冻结300 | ltx_real232 | 蒸馏基座＋仅真实数据LoRA，checkpoint step232 |
| 冻结300 | ltx_synthetic600 | 蒸馏基座＋仅合成数据LoRA，checkpoint step600 |
| V4自建116 | ltx_base / a849 / h3_pdd8 / vace | 原四模型；LTX/VACE配置与冻结集不同 |

232/600是checkpoint步数，不是样本数；步数不同，不能把全部差异仅归因为数据来源。Judge输入不含这些model_id；仅本地关联和审核展示实名。

具体媒体路径以`metadata/tasks.json`为准。键为`dataset + task_id`，如`frozen300__tripvvt_01`和`v4_116__tripvvt_01`不是同一个样本。主要上游入口：

- 冻结输入：`../frozen300_20261113/package/inputs.jsonl`。
- 冻结四模型输出映射：`../frozen300_vbench_20261115/metadata/scoring_media.jsonl`；同目录`source_media.jsonl`为SOURCE对齐媒体。
- V4输入：`../a849_external_vbench_v4_116/metadata/inputs.jsonl`；SOURCE在其`inputs/prepared/`，输出在`outputs/<model_id>/`。
- Real-only：`/scratch/prj0000000275/jcy/xks/fastref/ltx_distilled_real232_frozen300_20261115/results.jsonl`。
- Synthetic-only：`/scratch/prj0000000275/jcy/xks/fastref/ltx_distilled_synthetic600_frozen300_20261115/results.jsonl`。

师兄两组目录只读复用，不改写、不重新推理、不复制权重。`metadata/ablation_models.json`保存清单哈希、LoRA哈希和step。接入时逐条检查300个ID、uid、source哈希、原始指令、有序参考、fps、帧数和画布；新增输出抽帧时再校验实际文件哈希。

## API认证、请求与费用

本轮是**OpenRouter转发DeepSeek**，不是直连DeepSeek官方接口，也不使用ChatGPT套餐。评分地址为`POST https://openrouter.ai/api/v1/chat/completions`。认证头为`Authorization: Bearer <LIJUYANG_ICLR_PERSONAL>`；密钥只从环境读，不写Git、日志、README或ZIP。

终端安全设置和只检查是否存在：

```bash
read -rs -p 'OpenRouter API key: ' LIJUYANG_ICLR_PERSONAL
export LIJUYANG_ICLR_PERSONAL
python - <<'PY'
import os
print('key configured:', bool(os.environ.get('LIJUYANG_ICLR_PERSONAL')))
PY
```

实际固定请求参数在`scripts/engine.py:PARAMS`和`metadata/protocol.json`：

```json
{
  "model": "deepseek/deepseek-v4.1-flash-1017",
  "temperature": 0,
  "max_tokens": 800,
  "reasoning": {"enabled": false},
  "provider": {
    "only": ["wafer/us"],
    "allow_fallbacks": false,
    "require_parameters": true,
    "quantizations": ["mxfp4"],
    "max_price": {"prompt": 0.04, "completion": 0.6}
  }
}
```

图片通过base64 data URL的`image_url`发送，`detail=high`。`max_price`是每百万token的路由价格过滤条件，不是总预算，也不是实际成交单价。总预算和余额保留线已取消，protocol的budget/minimum_balance为null。实际费用取响应`usage.cost`；不能将本轮Wafer量化版本描述成DeepSeek官方全精度评测。

只读连通性/余额自测：

```bash
python - <<'PY'
import os, requests
r = requests.get('https://openrouter.ai/api/v1/credits',
    headers={'Authorization': 'Bearer '+os.environ['LIJUYANG_ICLR_PERSONAL']}, timeout=30)
print('HTTP:', r.status_code)
r.raise_for_status()
d = r.json()['data']
print('remaining USD:', d['total_credits']-d['total_usage'])
PY
```

单次评分示例直接复用正式消息构造器。此示例有真实费用，但不自动写入正式评分/费用表；正式批量调用使用engine，避免和活跃队列重复测试同一项。

```bash
python - <<'PY'
import os, sys, requests
from pathlib import Path
root = Path(os.environ['VLM_EVAL_DIR'])
sys.path.insert(0, str(root/'scripts'))
import engine as e
case = e.read(next((root/'metadata/prepared').glob('*.json')))[0]
metric = 'content_preservation'  # IF改为edit_achievement
payload = dict(e.PARAMS, messages=e.messages(case, metric))
r = requests.post('https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': 'Bearer '+os.environ['LIJUYANG_ICLR_PERSONAL']},
    json=payload, timeout=180)
r.raise_for_status()
d = r.json()
if d.get('error'):
    raise RuntimeError(d['error'])  # HTTP200也可能携带上游504
print(e.parse(d['choices'][0]['message']['content']))
print('provider:', d.get('provider'), 'cost USD:', d.get('usage', {}).get('cost'))
PY
```

## 环境重建

实测Python3.12.9、httpx0.28.1、PyAV19.0.2、Pillow12.1.1、requests2.32.3。核心依赖固定于requirements.txt。当前额外包在`/tmp/vlm_pilot_deps`，换节点或目录被清理后可重建：

```bash
python -m pip install --target /tmp/vlm_pilot_deps -r "$VLM_EVAL_DIR/requirements.txt"
export PYTHONPATH="/tmp/vlm_pilot_deps${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import av, PIL, httpx, requests; print(av.__version__, PIL.__version__, httpx.__version__, requests.__version__)'
```

PyAV负责解码/编码，不依赖系统ffmpeg命令或CUDA。API沿用环境代理；SOCKS依赖已通过requirements的extras包含。百度上传单独移除大小写HTTP/HTTPS/ALL_PROXY变量。不要通过切换provider解决代理问题，否则会改变实验协议。

## 送评证据与输出格式

时间轴沿用冻结12fps、N帧，不变速、不补尾，不使用GT选帧：

```text
T = (N-1)/12
K = max(16, ceil(4*T))
第i个逻辑帧索引 = floor(i*(N-1)/(K-1) + 0.5)，i=0..K-1
```

10秒121帧取40点，4秒49帧取16点，包含首尾。SOURCE/OUTPUT各侧长边最多1024、保留比例、灰底居中；配对图左SOURCE右OUTPUT，图像外侧标注P编号/索引/时间，JPEG95、subsampling0。静态参考经EXIF/透明底处理后独立PNG；动作参考按输入冻结原生索引映射同一时间轴、独立排列。H3复用现有无损12fps投影，不使用SSIM归一化视频或浏览压缩副本。

消息按如下顺序组织：system任务评分模板 → user英文原始指令 → IF的有序参考（角色/view，动作R1-Txx）→ K张P配对图 → 实际K、评分范围及两行输出要求。CP不提供参考像素。CoinVE保留全部编号子指令。GT、caption、UE属性真值和人工评价均不输入Judge。

8类模板为identity、clothing、action、camera、hairstyle、hair_color、scene、instruction_conditioned，各有IF/CP。CoinVE/native使用通用模板，TripVVT使用服装模板。当前原文在prompts/，完整副本为`ds_full_eval_20261115_prompts_cp_v3.zip`。

```text
score: 5
reason: P00等帧中的具体可见依据。
```

只接受1–5整数与单行理由；完整外层text代码围栏可确定性去除。格式不合法、截断、空响应、API错误均重试，不填0、不当作编辑失败。通用IF输出段残留P00–P15文字，最终实际K、P00至P(K-1)约定及完整实际输入均保留；按用户确认本轮不再修改原文。

## 脚本地图与独立运行

| 脚本 | 用途 | 当前操作 |
|---|---|---|
| engine.py | 抽帧、消息构造、并发API、逐次结果、续跑 | 仅评分入口 |
| finish.py | engine→report→audit→review→上传 | 整轮续跑默认入口 |
| report.py | 覆盖率、逐对表、均分/SD、配对bootstrap、消融/稳定性表 | 不调用评分API |
| audit.py | 4748项覆盖、版本和证据关联验收 | 全部完成后执行 |
| review.py | 40任务页面、浏览副本、证据附录与ZIP | 全部完成后执行 |
| extend_models.py | 保存旧清单并接入两个消融模型 | 已执行，通常不用重跑 |
| revise_cp.py | 启动前CP v2校准 | 历史步骤，不重跑 |
| revise_camera.py | 检查点后镜头CP v3与36项归档 | 已执行，不再改Prompt |
| cp_policy.py | 历史CP v2规则/生成函数 | 实际生效原文以prompts/为准 |

```bash
# 仅评分，已有合规成功项会跳过
python -u "$VLM_EVAL_DIR/scripts/engine.py"
# 检查点汇总
python "$VLM_EVAL_DIR/scripts/report.py" --checkpoint
# 全量汇总；未齐时只报告incomplete和缺项
python "$VLM_EVAL_DIR/scripts/report.py"
# 最终验收、打包
python "$VLM_EVAL_DIR/scripts/audit.py"
python "$VLM_EVAL_DIR/scripts/review.py"
```

`engine.py setup`是早期四模型/上游模板生成入口，不是当前六模型冻结任务的重置入口。不要在现有目录重新初始化。迁移环境时保留metadata、prompts和引用媒体；只复制Git源码不能恢复本轮清单和历史。

评分锁为metadata/run.lock。复用成功项时检查Prompt哈希、参数和卡片清单；不符时报stale score requires versioned migration，不能删断言强行复用。准备缓存为metadata/prepared/<task_key>.json；追加消融时保留原四模型，只生成缺失输出。JSON用唯一临时文件再原子替换。

准备队列容量64，API每项最多5次尝试。429/5xx退避，HTTP200内的error也按实际错误处理；402的in_flight_budget_exhausted全局等待后重试，其他402停止；400/401/403/404需检查配置。单请求正文<45MiB，已派发正文累计受1GiB门限约束，这不等于Python总内存硬上限。失败后先确认旧进程已结束，再从原入口续跑。

## 冻结与论文复现口径

metadata/protocol.json冻结Judge参数、输入清单哈希、16份Prompt哈希和媒体参数；metadata/selections.json冻结抽样成员：

- 18任务技术小测：扩展后184项基础评分，后续复用。
- 100任务检查点：冻结72×六模型、自建28×四模型，共1088项，已完成。
- 20任务重复：冻结15、自建5，共220项额外评分，r1与r0请求签名必须相同。
- 40任务审核：冻结30、自建10，共220视频对、440项IF/CP分数。

原抽样seed20261115，重复选择20261116，100任务选择20261117，诊断随机对照20261118；具体成员以冻结selection为准，不根据当前输出重新挑选。

镜头CP v3只修订camera_content_preservation.md，六模型36项旧镜头CP统一移入results_archive_camera_cp_v2/并重评；其他15份模板不变，无手改分数。历史模板/协议/检查点表在metadata/camera_revision_v2/，放行决策在metadata/checkpoint100_review.json。用户确认的最终哈希在metadata/prompt_lock_user_confirmed.json。metadata/judge_flags.json保留发现的疑似误判，不等同人工真值。

同时报告全量描述性结果，以及校准/未参与此次修订两部分：冻结72/228、自建28/88。两套数据单独汇总，不合并排名；V4部分TripVVT看过输出后选样，须说明选择偏差。检查点触发条件只是诊断入口，不用于选择提高特定模型排名的Prompt。

配对比较用A849减同任务其他模型的差值，按SOURCE哈希聚类bootstrap1000次、固定seed，给95%区间。temperature=0也不保证云端逐次完全一致，重复评分另报完全一致率、差≤1比例和MAE。

## 目录、表格与费用追溯

```text
scripts/、prompts/、requirements.txt   源码、16份原文、依赖
metadata/                             输入/协议/选择/模型登记/准备回执/状态
frames/<task_key>/                    Judge实际接收的JPEG/PNG
results/                              当前结果与每次attempt原始响应
results_archive_camera_cp_v2/          被替代的旧镜头CP与attempt
report/checkpoint100/、report/full/    检查点与全量报告
review/                               离线页面、浏览视频、证据附录
logs/、*.zip                          日志、Prompt包和最终审核包
```

单项名为`<dataset>__<task_id>__<model>__<metric>__r0.json`，重复用r1；尝试另存`...__r0__attempt0.json`等。最终项是最近一次结果，attempt保留历史。记录包含task/case/model/metric、repeat、请求签名、参数、Prompt哈希、卡片/参考哈希、HTTP/实际错误状态、完整API响应、分数理由、耗时及时间戳。响应内有实际Judge型号、provider、token用量和usage.cost；不保存密钥或完整base64请求正文，可由冻结素材、模板和构造器重建请求。

| report/full文件 | 内容 |
|---|---|
| coverage.json | 预期/完成基础评分、缺项、费用、attempt数 |
| pairs.csv | 2264视频对，IF/CP及理由、输入/输出路径 |
| scores.csv | 4528基础评分，一行一个维度 |
| summary.csv | 分数据集/类别/模型/维度的n、均值、SD和分数分布 |
| comparisons.csv | A849与其他模型的配对差和区间 |
| ablation_comparisons.csv | A849对基座、real232、synthetic600 |
| calibration_* | 100校准任务逐项表与汇总 |
| not_used_for_prompt_tuning_* | 316个未参与本次调优的任务表与汇总 |
| stability.csv / stability_summary.json | 220项重复评分及稳定性 |
| diagnostic_cases.* / investigation.json | 诊断候选及触发条件，不是人工标注 |

费用只累加results*目录中的attempt，包含旧档案，避免把最终项与attempt重复计费。progress.cost_usd是本任务响应可归属费用，不是账户余额；余额单独用credits接口查。检查点含镜头重评的费用约$1.10，这是历史节点，不是当前总费用。

## 审核ZIP与百度上传

review.py生成全展开scroll-down页、固定任务/分组导航，无人工评分控件，模型和Judge实名。主页面展示视频；P配对图只在独立evidence.html附录。TripVVT展示SOURCE、服装参考、GT及完整官方caption，后两项仅供人工看，不送Judge。

浏览副本为H.264 Main/yuv420p、12fps、CRF16、faststart，共同时间同步播放，另有完整原SOURCE入口。它们不用于评分。按用户要求跳过细致浏览器兼容测试，记录skipped，不宣称passed。

最终ZIP为`ds_full_eval_20261115_review40.zip`，上传到`/apps/bypy/lance_review/ds_full_eval_20261115_review40.zip`。独立续传命令：

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
python model_evaluation/minimax_h3_ref2va90_local_20261112/scripts/upload_baidu_xpan.py \
  --local "$VLM_EVAL_DIR/ds_full_eval_20261115_review40.zip" \
  --remote /apps/bypy/lance_review/ds_full_eval_20261115_review40.zip \
  --state "$VLM_EVAL_DIR/metadata/baidu_upload.json" \
  --bypy-config dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json
```

该凭据文件不打包/打印/提交；默认~/.bypy不是本轮入口。只上传ZIP，不另传JSONL/哈希/sidecar，不额外核对远端大小/哈希或整包回读。上传器自身分块传输校验照常执行，以上传工具返回成功为依据。

## 完成判据与故障恢复

完整交付要同时满足：

1. metadata/api_completion.json表示评分结束。
2. report/full/coverage.json的successful_scores=expected_scores=4528、missing为空。
3. stability_summary.json的completed=expected=220。
4. metadata/completion_audit.json的status=passed，覆盖4748项当前结果、版本和证据关联。
5. metadata/review_delivery.json确认40任务、220视频对、440评分及ZIP。
6. metadata/baidu_upload.json的status=complete，metadata/delivery_complete.json存在。

仅api_completion、一个快照或已有ZIP均不足以证明完成；技术验收不证明每项VLM语义正确。

| 情况 | 处理 |
|---|---|
| httpcore/socksio缺失 | 按requirements补依赖，不换Judge |
| 429 image_processing_busy | 上游限流；自动退避，不记低分，不开第二队列 |
| HTTP200但内层504/error | engine按实际错误重试 |
| 格式错误/截断 | 重试并保留attempt，不补0分 |
| 认证或真实余额不足 | 检查环境变量/对应账户，修复后原入口续跑；不输出key |
| stale score/哈希不符 | 查清来源与版本，不删断言、不混分 |
| failure/pipeline_failure文件 | 联合实际进程和日志判断，历史文件可能保留 |
| 编码/上传失败 | 保留已评分结果，修对应阶段，不重推理或改分 |

稳定后默认约15分钟检查一次，启动/异常/完成可额外检查。源码、README、Prompt入项目版本管理；媒体、运行清单、响应、日志、ZIP和凭据不入Git。当前gitignore排除这些运行目录，因此交接不能只复制Git源码，还需保留本地metadata/prompts和引用的上游媒体。换机器前明确tasks中绝对路径的迁移映射，不静默改写冻结来源。

## 最终结果快照（2026-11-16）

发型专项审核包可单独生成：`python "$VLM_EVAL_DIR/scripts/review.py" --hair24`。范围为自建V4的全部24个hair任务、四模型96对视频、192项原有IF/CP评分，不重跑API；页面位于review_hair24/，ZIP名为ds_full_eval_20261115_v4_hair24_review.zip，选择与交付记录分别为metadata/hair24_selection.json和metadata/hair24_review_delivery.json。该入口不覆盖原40任务包。专项上传回执为metadata/hair24_baidu_upload.json。

以下为全量描述性均分，1–5分；包括100个校准任务。论文同时提供calibration与not_used_for_prompt_tuning分表，不把全量描述性结果宣称为全部未参与调优的测试结果。

| 数据集 | 模型 | IF | CP |
|---|---|---:|---:|
| 冻结300 | A849 v7 | 4.117 | 4.183 |
| 冻结300 | H3 PDD8 | 3.853 | 3.673 |
| 冻结300 | VACE LightX2V4 | 3.013 | 4.193 |
| 冻结300 | Synthetic-only step600 | 3.810 | 3.777 |
| 冻结300 | Real-only step232 | 2.940 | 2.967 |
| 冻结300 | LTX Distilled v7 | 2.213 | 2.477 |
| V4自建116 | A849 | 4.129 | 4.431 |
| V4自建116 | H3 PDD8 | 3.922 | 4.474 |
| V4自建116 | VACE | 3.224 | 4.474 |
| V4自建116 | LTX Dev Base | 1.431 | 2.940 |

- 基础评分4528项，重复220项，无缺项；当前有效评分4748项。
- 重复评分完全一致率89.55%，差不超过1分99.55%，平均绝对差0.1091（20任务、220项，不是220个独立源视频）。
- 全部attempt（含历史重评与错误重试）的响应累计费用为 **$4.439793182**；共4833个attempt记录，不与最终结果文件重复计费。
- 审核包40任务、220结果视频、440评分，另含SOURCE/参考/GT，共267个浏览视频；ZIP本地大小5,416,655,942字节。
- 百度上传成功回执：`/apps/bypy/lance_review/ds_full_eval_20261115_review40.zip`，fs_id为334645144393535。依据上传工具成功，不额外进行远端大小/哈希核对或整包回读。
- 评分验收见metadata/completion_audit.json；本地页面/证据/ZIP成员检查见metadata/review_artifact_audit.json；上传与最终完成见metadata/baidu_upload.json、metadata/delivery_complete.json。浏览器兼容测试仍为skipped。
