"""One-time, pre-launch migration; preserves the original frozen prompts."""
import fcntl
from pathlib import Path
from engine import ROOT, read, save, sha
from cp_policy import VERSION, calibrate

def main():
    lock=(ROOT/'metadata/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=ROOT/'metadata/protocol.json';protocol=read(p)
    if protocol.get('cp_version')==VERSION:
        print('CP v2 already frozen');return
    assert not list((ROOT/'results').glob('*.json')), 'Stop: existing scores require explicit versioned migration'
    backup=ROOT/'metadata/cp_v1_before_user_revision'
    backup.mkdir(exist_ok=True)
    save(backup/'protocol.json',protocol)
    for f in sorted((ROOT/'prompts').glob('*.md')):
        assert sha(f)==protocol['prompt_hashes'][f.name]
        (backup/f.name).write_bytes(f.read_bytes())
        if f.name.endswith('_content_preservation.md'):
            f.write_text(calibrate(f.read_text()))
    protocol.update(cp_version=VERSION,prompt_hashes={f.name:sha(f) for f in (ROOT/'prompts').glob('*.md')})
    save(p,protocol)
    save(ROOT/'metadata/cp_revision.json',{'version':VERSION,'reason':'User requested perceptual CP anchors and strict separation from failed instruction following','old_full_evaluation_scores':0,'old_protocol':'metadata/cp_v1_before_user_revision/protocol.json','revised_cp_prompts':8,'if_prompts_changed':False})
    print('CP v2 frozen: 8 prompts; IF unchanged; no full-evaluation scores existed')

if __name__=='__main__':main()
