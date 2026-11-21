"""One evidence-supported, all-model camera CP correction after checkpoint."""
import fcntl,shutil
from engine import ROOT, read, save, sha

RULE='''
### 镜头编辑中的可比较内容与新可见区域
相机转到人物背面或绕拍时，画面可能由建筑转为另一方向的海面、道路或墙面；拉远时也可能显露源视野之外的地面与物体。背景像素、地标的可见性和画面中的前后景关系因此改变，本身不能证明场景被替换。人物由正面变为背面本身也不表示身份、表情或动作被改变。
只比较两段中确实可对应、可观察的主体属性和环境结构。若要因场景替换扣分，须指出不能由指令允许的视角、透视、取景或可见范围变化解释的具体矛盾，例如同一个可对应地标在两段中结构被改写；不能仅因为源建筑不再入镜、新视角出现海面而扣分。
源中看不到的新区域不要求像素匹配，未知不能当作损失，也不能推断原场景里不存在它。若其余可比较内容充分保持，仅有上述允许的可见性变化，CP可给5。明确可见的人物、服装或可对应环境结构的实际损坏仍正常扣分。
这些规则仅用于CP。未按要求改变镜头仍应在IF扣分；不得因为镜头没改而降低CP，也不得因为其他内容被破坏而把镜头成功当作CP通过。
'''
def main():
    lock=(ROOT/'metadata/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(ROOT/'metadata/protocol.json')
    if protocol.get('cp_version')=='cp_v3_camera_visibility_20261115':print('Already revised');return
    coverage=read(ROOT/'report/checkpoint100/coverage.json');assert coverage['successful_scores']==coverage['expected_scores']==1088 and not coverage['missing']
    dest=ROOT/'metadata/camera_revision_v2';dest.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'report/checkpoint100',dest/'checkpoint100_before',dirs_exist_ok=True)
    p=ROOT/'prompts/camera_content_preservation.md';assert sha(p)==protocol['prompt_hashes'][p.name]
    (dest/p.name).write_bytes(p.read_bytes());save(dest/'protocol.json',protocol)
    archived=ROOT/'results_archive_camera_cp_v2';archived.mkdir(exist_ok=True);moved=[]
    for f in sorted((ROOT/'results').glob('*__camera_*__content_preservation__r*.json')):
        assert not (archived/f.name).exists();f.replace(archived/f.name);moved.append(f.name)
    assert sum('__attempt' not in name for name in moved)==36
    p.write_text(p.read_text()+RULE)
    protocol['prompt_hashes'][p.name]=sha(p);protocol['cp_version']='cp_v3_camera_visibility_20261115';save(ROOT/'metadata/protocol.json',protocol)
    save(ROOT/'metadata/camera_revision.json',{'version':protocol['cp_version'],'changed_prompts':[p.name],'reason':'camera_24 rear-view exposes sea, incorrectly penalized as environment replacement despite allowed viewpoint change','all_model_scope':'all six frozen300 models, all25 camera tasks','old_checkpoint_scores_archived':36,'archived_files':moved,'cost_policy':'all archive and current attempt costs included','calibration_tasks':read(ROOT/'metadata/selections.json')['checkpoint100'],'other_prompts_changed':False})
    print('Camera CP revised;36 scores archived for all-model reevaluation',flush=True)
if __name__=='__main__':main()
