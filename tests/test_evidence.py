from dataclasses import replace
import unittest

from medical.calculation import Action, State
from medical.evidence import eligibility, evidence_step, observe, synthetic_world


class EvidenceTests(unittest.TestCase):
    def test_metadata_boundaries(self):
        w,_ = synthetic_world('stale', 0)
        record = next(r for r in w.records if all(eligibility(w,r).values()))
        for age, allowed in ((-1,False),(0,True),(7,True),(8,False)):
            self.assertEqual(eligibility(w,replace(record,day=w.day-age))['time_window'], allowed)

    def test_binding_requires_read_but_accepts_unqualified_source(self):
        for kind in ('wrong_encounter','ambiguous_unit','stale'):
            w,_ = synthetic_world(kind, 13)
            bad = next(r for r in w.records if not all(eligibility(w,r).values()))
            state = State(w.budget)
            action = Action('bind',bad.field,bad.pointer)
            self.assertFalse(evidence_step(w,state,action)[0].bindings)
            read,_,_ = evidence_step(w,state,Action('query',bad.field))
            self.assertIn(bad.pointer,observe(w,read))
            bound, reward, _ = evidence_step(w,read,action)
            self.assertEqual(dict(bound.bindings)[bad.field],bad.pointer)
            self.assertLess(reward,0)

    def test_matched_decisions_differ_only_in_binding(self):
        import json
        from medical.evidence import command_step
        for kind in ('wrong_encounter','ambiguous_unit','stale'):
            w,_=synthetic_world(kind,31)
            bad=next(r for r in w.records if not all(eligibility(w,r).values()))
            good=next(r for r in w.records if r.field==bad.field and all(eligibility(w,r).values()))
            state,_,_=command_step(w,State(w.budget),f'QUERY {good.field}')
            a,_,_=command_step(w,state,f'BIND {good.field} {good.pointer}')
            b,_,_=command_step(w,state,f'BIND {bad.field} {bad.pointer}')
            left,right=json.loads(observe(w,a)),json.loads(observe(w,b))
            self.assertNotEqual(left.pop('bindings'),right.pop('bindings'))
            self.assertEqual(left,right)
