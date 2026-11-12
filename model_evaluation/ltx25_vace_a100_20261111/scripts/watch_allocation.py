"""Sleep 30 minutes between scheduler reads; attach only to allocation 199052."""
import os
import subprocess
import time
from common import *

def allocation():
    result=subprocess.run(['scontrol','show','job','199052','--oneliner'],text=True,capture_output=True,check=True).stdout
    fields=dict(p.split('=',1) for p in result.split() if '=' in p)
    assert fields['Partition']=='testqueue','Wrong partition: will not attach'
    assert fields['UserId'].split('(')[0]=='stuchengyou'
    return fields

def main():
    import fcntl
    lock=(ROOT/'metadata/watcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        a=allocation()
        atomic(ROOT/'metadata/allocation_status.json',{'checked_unix':time.time(),'jobid':199052,'state':a['JobState'],'partition':a['Partition'],'node':a.get('NodeList'),'reason':a.get('Reason'),'remaining_limit':a.get('TimeLimit'),'next_check_seconds':1800})
        print(time.strftime('%F %T'),a['JobState'],a.get('NodeList'),flush=True)
        if a['JobState']=='RUNNING':
            cmd=['srun','--jobid=199052','--partition=testqueue','--account=prj0000000275','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','bash',str(ROOT/'scripts/launch_stage.sh'),'prepare']
            atomic(ROOT/'metadata/attached.json',{'command':cmd,'started_unix':time.time(),'stage':'prepare'})
            with (ROOT/'logs/prepare.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            cmd[-1]='smoke'
            with (ROOT/'logs/smoke.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic(ROOT/'metadata/smoke_ready.json',{'status':'awaiting_agent_protocol_review','completed_unix':time.time(),'next_stage':'controls then second after protocol review'})
            return
        if a['JobState'] not in ('PENDING','CONFIGURING'):raise RuntimeError(f'Allocation unavailable: {a["JobState"]}')
        time.sleep(1800)

if __name__=='__main__':main()
