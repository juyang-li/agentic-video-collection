#!/usr/bin/env python3
"""Launch four formal 6-checkpoint ablation workers after normal chains are complete."""

from __future__ import annotations
from datetime import datetime
import json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python")
RUNNER = Path(__file__).with_name("run_small_ablation.py")
JOBS = (
    ("lance_base", ("vit_zero_text_keep", "vit_zero_text_zero")),
    ("lance_base", ("vit_gaussian_text_keep", "vit_gaussian_text_zero")),
    ("gtfresh_step1150", ("vit_zero_text_keep", "vit_zero_text_zero")),
    ("gtfresh_step1150", ("vit_gaussian_text_keep", "vit_gaussian_text_zero")),
)

def write_json(path: Path, value: object) -> None:
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n"); os.replace(tmp,path)

def main() -> int:
    visible=[item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES","0,1,2,3").split(",") if item.strip()]
    if len(visible)<4: raise RuntimeError("four visible GPUs are required")
    clean_env=dict(os.environ)
    for name in ("WAN_DB","WANDB_API_KEY","GEMINI_API_KEY","GOOGLE_API_KEY","LIJUYANG_ICLR","LIJUYANG_ICLR_PERSONAL","OPEN_ROUTER_QWEN"): clean_env.pop(name,None)
    for model in ("lance_base", "gtfresh_step1150"):
        check=[str(PYTHON),str(RUNNER),"--model",model,"--mode","ablation",
               "--conditions","vit_zero_text_keep","--dry-run"]
        subprocess.run(check,env=clean_env,check=True)
    (ROOT/"logs").mkdir(parents=True,exist_ok=True); active=[]; jobs=[]
    for slot,(model,conditions) in enumerate(JOBS):
        log_path=ROOT/"logs"/f"{model}_{'zero' if 'zero' in conditions[0] else 'gaussian'}.log"
        command=[str(PYTHON),str(RUNNER),"--model",model,"--mode","ablation","--conditions",*conditions]
        env=dict(clean_env)
        env.update(CUDA_VISIBLE_DEVICES=visible[slot],PYTHONUNBUFFERED="1")
        log=log_path.open("a",encoding="utf-8"); p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job={"model_id":model,"conditions":list(conditions),"gpu":visible[slot],"pid":p.pid,"log":str(log_path),"status":"running"}
        jobs.append(job); active.append((job,p,log)); print(f"started gpu={visible[slot]} pid={p.pid} {model} {conditions}",flush=True)
    status={"status":"running","started_at":datetime.now().astimezone().isoformat(),"jobs":jobs}; write_json(ROOT/"run_status.json",status)
    failed=False
    for job,p,log in active:
        p.wait(); log.close(); job["returncode"]=p.returncode; job["status"]="complete" if p.returncode==0 else "failed"; failed|=p.returncode!=0
    status.update(status="failed" if failed else "complete",finished_at=datetime.now().astimezone().isoformat()); write_json(ROOT/"run_status.json",status)
    return 1 if failed else 0

if __name__=="__main__": raise SystemExit(main())
