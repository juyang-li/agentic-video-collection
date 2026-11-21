from common import *
import argparse,time
from package import main
p=argparse.ArgumentParser();p.add_argument('--exit-code',type=int,required=True);a=p.parse_args()
status=read(ROOT/'metadata/run_status.json') if (ROOT/'metadata/run_status.json').exists() else {}
if a.exit_code or status.get('status')!='complete':
    for sid in ORDER:
        path=ROOT/f'outputs/{sid}.json'
        old=read(path) if path.exists() else {}
        if old.get('status')!='complete' and old.get('status')!='failed':
            atomic(path,{**old,'id':sid,'status':'failed' if old else 'not_run','failure_reason':f'GPU step exited {a.exit_code}; see logs/worker.log','completed_unix':time.time()})
    atomic(ROOT/'metadata/run_status.json',{**status,'status':'failed','step_exit_code':a.exit_code,'completed_unix':time.time()})
main()
