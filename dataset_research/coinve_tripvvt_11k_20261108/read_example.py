"""Read an unpacked independent ZIP; use identical absolute times on both sides."""
from pathlib import Path
import subprocess
import sys
from dataset_reader import iter_samples
root=Path(sys.argv[1] if len(sys.argv)>1 else '.')
for row in iter_samples(root/'manifest.jsonl'):
    print(row['sample_id'],row['prompt'])
    # Example timestamp inside the common interval; source/target are never time-stretched.
    timestamp=row['common_end_seconds']/2
    for role in ('source','target'):
        path=row[role]
        command=['ffmpeg','-v','error','-i',str(path),'-ss',str(timestamp),'-frames:v','1',
                 '-f','image2pipe','-vcodec','png','-']
        result=subprocess.run(command,capture_output=True,check=True)
        print(role,'timestamp',timestamp,'PNG bytes',len(result.stdout))
    if row.get('reference'):print('garment reference:',row['reference'])
    break
