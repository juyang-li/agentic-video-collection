"""Verify frozen media binding and actual API evidence boundaries on CPU."""
import collections, json
from pilot import ROOT, load, sha, messages, INDICES

def main():
    cs=load(ROOT/'metadata/candidates.json')
    assert len(cs)==40 and len({c['task_id'] for c in cs})==10
    assert set(collections.Counter(c['task_id'] for c in cs).values())=={4}
    hashes=[]
    for c in cs:
        assert sha(c['output'])==c['output_sha256']
        assert sha(c['source'])==c['source_sha256']
        assert [f['frame_index'] for f in c['cards']]==INDICES
        for f in c['cards']+c['references']:assert sha(ROOT/f['path'])==f['sha256']
        for metric in ['edit_achievement','content_preservation']:
            msgs,h=messages(c,metric);hashes.append(h)
            content=msgs[1]['content'];images=[b for b in content if b['type']=='image_url']
            assert len(images)==16+(len(c['references']) if metric=='edit_achievement' else 0)
            serialized=msgs[0]['content']+'\n'+'\n'.join(b['text'] for b in content if b['type']=='text')
            assert c['task_id'] not in serialized and c['model'] not in serialized
            assert 'source_review' not in serialized and 'original_omni_instruction' not in serialized
            assert c['instruction'] in serialized
    report={'candidates':40,'tasks':10,'checked_dimension_inputs':80,'unique_prompt_hashes':sorted(set(hashes)),'media_hashes':'passed','cp_reference_pixels':'absent','identity_and_human_notes_leakage':'absent'}
    from pilot import save
    save(ROOT/'metadata/input_check.json',report);print(report)

if __name__=='__main__':main()
