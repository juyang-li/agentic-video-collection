"""Read-only per-worker allocator observations. No CUDA initialization or resets."""
import os,threading,time,json,sys
if os.environ.get('H3_MEMORY_LOG_DIR'):
 def observe():
  path=os.path.join(os.environ['H3_MEMORY_LOG_DIR'],f'allocator_{os.getpid()}.jsonl')
  while True:
   try:
    torch=sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
     for d in range(torch.cuda.device_count()):
      allocated=torch.cuda.memory_allocated(d);reserved=torch.cuda.memory_reserved(d)
      if not (allocated or reserved):continue
      r={'unix':time.time(),'pid':os.getpid(),'cuda_index':d,'uuid':str(getattr(torch.cuda.get_device_properties(d),'uuid','')),'allocated_bytes':allocated,'reserved_bytes':reserved,'peak_allocated_bytes':torch.cuda.max_memory_allocated(d),'peak_reserved_bytes':torch.cuda.max_memory_reserved(d)}
      with open(path,'a') as f:f.write(json.dumps(r)+'\n')
   except Exception:pass
   time.sleep(.5)
 threading.Thread(target=observe,daemon=True,name='h3-readonly-memory-observer').start()
