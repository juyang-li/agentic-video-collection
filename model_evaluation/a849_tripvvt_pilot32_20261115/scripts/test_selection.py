"""Behavioral requirements for the user's output-conditioned quality gate."""
import copy,unittest
from select_results import select

def cases():
    return [dict(id=f'tripvvt_{n:02}',reviewed=True,a849=dict(edit_success='successful',preservation='good'),
        h3_pdd8=dict(edit_success='failed',preservation='bad'),comparison='a849_better')for n in range(1,33)]

class SelectionTests(unittest.TestCase):
    def test_h3_failure_allowed_and_frozen_id_tiebreak(self):
        rows=cases();rows.reverse();r=select(rows)
        self.assertEqual(r['selected'],[f'tripvvt_{n:02}'for n in range(1,9)])
    def test_a849_hard_gate_even_when_h3_fails(self):
        rows=cases();rows[0]['a849']['edit_success']='partial';rows[1]['a849']['preservation']='mixed'
        self.assertEqual(select(rows)['selected'][0],'tripvvt_03')
    def test_prefer_a849_wins_before_successful_ties(self):
        rows=cases()
        for r in rows[:10]:r['comparison']='comparable';r['h3_pdd8']=dict(edit_success='successful',preservation='good')
        self.assertEqual(select(rows)['selected'],[f'tripvvt_{n:02}'for n in range(11,19)])
    def test_shortfall_kept_without_relaxing_gate(self):
        rows=cases()
        for r in rows[7:]:r['a849']['edit_success']='failed'
        result=select(rows);self.assertEqual(result['status'],'shortfall');self.assertEqual(result['shortfall'],1)
        self.assertEqual(len(result['selected']),7);self.assertTrue(result['no_automatic_candidate_33'])
    def test_incomplete_or_duplicate_or_extra_pilot_not_selectable(self):
        for rows in [cases()[:-1],cases()+[copy.deepcopy(cases()[0])]]:
            with self.assertRaises(AssertionError):select(rows)
        rows=cases();rows[0]['reviewed']=False
        with self.assertRaises(AssertionError):select(rows)
    def test_h3_win_excluded(self):
        rows=cases();rows[0]['comparison']='h3_better'
        self.assertNotIn('tripvvt_01',select(rows)['selected'])
    def test_mixed_tradeoff_is_not_a_successful_tie(self):
        rows=cases();rows[0]['comparison']='mixed'
        self.assertNotIn('tripvvt_01',select(rows)['selected'])

if __name__=='__main__':unittest.main()
