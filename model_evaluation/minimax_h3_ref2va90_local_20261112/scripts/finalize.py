from common import *
import time,psutil,signal
require_job();name='minimax_h3_ref2va90_local_scroll_20261112'
if (ROOT/'metadata/delivery.json').exists() and read(ROOT/'metadata/delivery.json').get('status')=='complete':
 print('DELIVERY_ALREADY_FINALIZED',flush=True);raise SystemExit(0)
receipt=read(ROOT/'review'/f'{name}.baidu_upload.json');assert receipt['status']=='complete' and receipt['remote']==f'/apps/bypy/lance_review/{name}.zip'
validation=read(ROOT/'review'/name/'validation.json');assert validation['status']=='passed' and validation['new_h3_outputs']==90 and validation['videos_full_decoded']==450
assert read(ROOT/'metadata/protocol_gate_all.json')['status']=='passed'
states=[read(p) for p in (ROOT/'outputs').glob('*.json')];assert len(states)==90 and all(s['status']=='complete' for s in states)
report=read(ROOT/'metadata/final_summary.json');assert len(report['per_gpu_allocator_peak_bytes'])==4
now=time.time();execution=read(ROOT/'metadata/execution.json');elapsed=now-execution['resource_check_unix']
record={'status':'complete','remote':receipt['remote'],'local_zip':str(ROOT/'review'/f'{name}.zip'),'completed_unix':now,'elapsed_from_resource_check_seconds':elapsed,'samples':90,'videos_full_decoded':450,'human_quality':'not assessed','upload_policy':'ZIP only; uploader success; no additional remote hash/size/readback verification'}
atomic(ROOT/'metadata/delivery.json',record);execution.update(state='complete',completed_unix=now);atomic(ROOT/'metadata/execution.json',execution)
with (ROOT/'README.md').open('a') as f:f.write('\n上传器已返回complete。自资源清单核验起至交付完成约 %.1f 分钟；生成墙钟 %.1f 分钟，实测四卡合计 %.2f 条/小时。显存分卡原始数据及最终汇总已保留；技术验收不代表人工质量通过。\n' % (elapsed/60,report['generation_wall_seconds']/60,report['four_gpu_wall_clips_per_hour']))
# Release only this experiment's idle serving process; keep allocation and environment.
target=str(ROOT/'runtime/env/bin/sglang');stopped=[]
for p in psutil.process_iter(['cmdline']):
 cmd=p.info['cmdline'] or []
 if target in cmd and 'serve' in cmd:p.send_signal(signal.SIGTERM);stopped.append(p.pid)
record['own_server_stopped_pids']=stopped;atomic(ROOT/'metadata/delivery.json',record);print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
