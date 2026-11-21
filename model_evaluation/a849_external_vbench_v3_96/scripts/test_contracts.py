import copy,unittest
from contracts import *
class PolicyTests(unittest.TestCase):
 def reviews(self,n):
  return [{'models':{m:{'clear_identity_transfer':i>=n} for m in MODELS}} for i in range(4)]
 def test_identity_threshold(self):
  self.assertFalse(identity_decision(self.reviews(2))['drop_identity'])
  self.assertTrue(identity_decision(self.reviews(3))['drop_identity'])
  r=self.reviews(3);r[0]['models']['vace']['clear_identity_transfer']=True
  self.assertFalse(identity_decision(r)['drop_identity'])
 def test_missing_identity_is_not_failure(self):
  r=self.reviews(4);r[0]['models']['a849']['clear_identity_transfer']=None
  with self.assertRaises(ValueError):identity_decision(r)
 def test_static_requires_source_and_each_model(self):
  flags={(sid,m):True for sid in ['a','b','c'] for m in ('source',*MODELS)}
  flags['b','source']=False;del flags['c','vace']
  self.assertEqual(common_static(['a','b','c'],flags),['a'])
 def test_effective_input_changes_invalidate_reuse(self):
  item={'source':{'sha256':'src'},'references':[{'sha256':'ref','type':'identity','view':'front'}],'prompt':{'en':'edit','zh':'元数据'},'canvas_wh':[1280,736],'fps':12,'num_frames':49,'timestamps_seconds':[i/12 for i in range(49)]}
  sig=generation_input(item,'a849');other=copy.deepcopy(item);other['old_id']='new_metadata'
  self.assertEqual(sig,generation_input(other,'a849'))
  for change in ['prompt','reference','source','timing']:
   other=copy.deepcopy(item)
   if change=='prompt':other['prompt']['en']='different target person'
   elif change=='reference':other['references'][0]['sha256']='different hair'
   elif change=='source':other['source']['sha256']='different clip'
   else:other['timestamps_seconds'][0]=.01
   self.assertNotEqual(sig,generation_input(other,'a849'))
  cond={'mask_sha256':['mask'],'adapted_prompt':'full edit'};sig=generation_input(item,'vace',cond)
  cond['mask_sha256']=['changed mask'];self.assertNotEqual(sig,generation_input(item,'vace',cond))
 def test_six_dimensions_and_missing_scores(self):
  self.assertEqual(len(DIMS),6);self.assertNotIn('dynamic_degree',DIMS);self.assertIn('temporal_flickering',DIMS)
  for x in [None,float('nan'),float('inf'),'0',True]:self.assertFalse(finite_score(x))
  self.assertTrue(finite_score(0.0))
if __name__=='__main__':unittest.main()
