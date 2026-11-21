"""Output-conditioned selection: A849 success+preservation hard gate, no H3 floor."""
from common import *

def select(reviews):
    assert len(reviews)==32 and len({r['id']for r in reviews})==32
    candidates=[];decisions=[]
    for r in reviews:
        assert r['reviewed'] is True
        a,h=r['a849'],r['h3_pdd8'];reasons=[]
        for m in [a,h]:
            assert m['edit_success'] in ['successful','partial','failed']
            assert m['preservation'] in ['good','mixed','bad']
        assert r['comparison'] in ['a849_better','comparable','h3_better','mixed']
        if r['comparison']=='comparable' and a['edit_success']=='successful':
            assert h['edit_success']=='successful','A failed H3 edit cannot be a comparable successful pair'
        if a['edit_success']!='successful':reasons.append('A849 did not clearly complete the garment edit')
        if a['preservation']!='good':reasons.append('A849 did not preserve source content well')
        if r['comparison']=='h3_better':reasons.append('H3 stronger; does not meet requested pair comparison')
        if r['comparison']=='mixed':reasons.append('Mixed tradeoff or both incomplete; neither a clear A849 win nor a comparable successful pair')
        tier=0 if r['comparison']=='a849_better'else 1
        if not reasons:candidates.append((tier,r['id']))
        decisions.append(dict(id=r['id'],eligible=not reasons,tier=tier if not reasons else None,reasons=reasons))
    chosen=[sid for _,sid in sorted(candidates)[:8]]
    for r in decisions:
        r['selected']=r['id']in chosen
        if r['eligible']and not r['selected']:r['reasons']=['Eligible; later frozen ID or lower preference tier than chosen8']
    return dict(status='complete'if len(chosen)==8 else'shortfall',selected=chosen,eligible=len(candidates),shortfall=max(0,8-len(chosen)),decisions=decisions,
        standard='A849 successful correct-person garment edit AND good preservation mandatory; H3 may be poor or failed. Prefer A849 wins, then successful comparable cases, then frozen ID.',
        output_conditioned_selection=True,no_automatic_candidate_33=True,no_category_quota=True,no_quality_score_threshold=True)

def main():
    items=rows(ROOT/'metadata/inputs.jsonl');reviews=read(ROOT/'metadata/pair_reviews.json')['cases']
    assert {r['id']for r in reviews}=={r['id']for r in items}
    for review in reviews:
        for model in ['a849','h3_pdd8']:
            path=ROOT/'outputs'/model/f"{review['id']}.mp4";r=read(path.with_suffix('.json'))
            assert r['status']=='complete'and sha(path)==r['output_sha256']==review[model]['output_sha256']
    result=select(reviews);result['input_manifest_sha256']=sha(ROOT/'metadata/inputs.jsonl');result['pair_reviews_sha256']=sha(ROOT/'metadata/pair_reviews.json')
    from collections import Counter
    by_id={r['id']:r for r in items}
    result['selected_categories']=dict(Counter(by_id[sid]['subgroup']for sid in result['selected']))
    result['category_totals']=dict(Counter(r['subgroup']for r in items))
    lines=['# TripVVT32 配对审核与选样','','这是查看输出后的选样，不代表无偏性能比较。A849明确换衣成功且保持良好是硬门槛，H3可以失败。先选A849明显更好，再选双方成功且相近，同档按冻结ID。无最终类别硬配额。',f"\n状态：{result['status']}；入选{len(result['selected'])}/8；缺额{result['shortfall']}。不足时不自动加第33条。",'','|候选ID|类别|A849 edit / preservation|H3 edit / preservation|比较|入选|原因|','|---|---|---|---|---|---|---|']
    decisions={r['id']:r for r in result['decisions']}
    for row in sorted(reviews,key=lambda r:r['id']):
        sid=row['id'];d=decisions[sid]
        lines.append('|'+sid+'|'+by_id[sid]['subgroup']+'|'+row['a849']['edit_success']+' / '+row['a849']['preservation']+'|'+row['h3_pdd8']['edit_success']+' / '+row['h3_pdd8']['preservation']+'|'+row['comparison']+'|'+('是'if d['selected']else'否')+'|'+('; '.join(d['reasons'])or'按预先规定的优先级与冻结ID入选')+'|')
    for row in sorted(reviews,key=lambda r:r['id']):
        lines+=['',f"## {row['id']}",'',row['comparison_notes'],'','A849: '+row['a849']['notes'],'','H3: '+row['h3_pdd8']['notes']]
    (ROOT/'metadata/selection_report.md').write_text('\n'.join(lines)+'\n')
    atomic(ROOT/'metadata/selection_decision.json',result);print(result['status'],result['selected'],flush=True)

if __name__=='__main__':main()
