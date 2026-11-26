from decimal import Decimal
from random import Random
from statistics import pstdev
import unittest

from medical.calculation import Action, State, World, observation, outcome, step
from medical.credit import Step, Trace, assign_credit, choose_anchors, verifier_reward


class CreditTests(unittest.TestCase):
    def test_abstention_and_anchor_selection(self):
        self.assertAlmostEqual(verifier_reward(((2, 1), (1, None))), 2 / 3)
        self.assertEqual(verifier_reward(((2, None),)), 0)
        self.assertEqual(verifier_reward(((2, -1),)), -1)
        self.assertEqual(choose_anchors([(0, 0), (1, 4), (2, 8)], 8, Random(3)), [0, 1])

    def test_hand_calculated_batch(self):
        def turns(*names):
            return tuple(Step(name, 0, 0, 3) for name in names)

        traces = [
            Trace("t", 1, turns("anchor", "collision", "a")),
            Trace("t", -1, turns("different", "collision", "b")),
            Trace("t", 1, turns("third", "lonely", "c")),
            Trace("t", -1, turns("fourth", "isolated", "d")),
            Trace("t", -1, turns("anchor", "unmatched-branch"), (0, 0)),
            Trace("t", 1, turns("anchor", "collision"), (0, 0)),
            Trace("t", -1, (*turns("anchor"), Step("masked", 1, 1, 4, False)), (0, 0)),
        ]
        credit = assign_credit(traces, process_weight=0, cost_weight=0)
        self.assertEqual(credit.tier[0, 0], "fork")
        self.assertAlmostEqual(credit.local[0, 0], 4 / 3)
        self.assertAlmostEqual(credit.local[4, 0], -4 / 3)
        self.assertEqual(credit.tier[5, 1], "collision")
        self.assertAlmostEqual(credit.local[1, 1], -2)
        self.assertEqual(credit.tier[2, 2], "turn")
        self.assertEqual(credit.suffix[3, 1], -1)
        self.assertNotIn((4, 1), credit.local)
        self.assertEqual(credit.token_counts[4, 1], 0)
        self.assertEqual(credit.token_counts[6, 1], 0)
        self.assertEqual(credit.sampled_tokens, 55)
        scale = pstdev(credit.local.values()) + 1e-8
        self.assertAlmostEqual(credit.advantages[5, 0], credit.local[5, 0] / scale)
        baseline = assign_credit(traces, beta=0, process_weight=0, cost_weight=0)
        self.assertAlmostEqual(baseline.advantages[0, 1], 1 / (1 + 1e-8))
        self.assertAlmostEqual(baseline.advantages[1, 1], -1 / (1 + 1e-8))
        self.assertEqual(baseline.advantages[5, 0], 0)

    def test_suffix_process_is_discounted_cost_is_not(self):
        sample = Trace("m", 1, (Step("a", 1, 1, 3), Step("b", -1, 1, 3)))
        scores = assign_credit([sample], process_weight=0.1, cost_weight=0.02, gamma=0.5)
        self.assertAlmostEqual(scores.suffix[0, 0], 1.01)


class CalculationTests(unittest.TestCase):
    def test_evidence_validity_and_separate_outcome(self):
        world = World("6", (("e1", "weight", Decimal(72)), ("e2", "height", Decimal("1.8"))))
        state = State(world.budget)
        self.assertEqual(step(world, state, Action("bind", "weight", "e1"))[1], -1)
        state, first, _ = step(world, state, Action("query", "weight"))
        self.assertEqual(first, 1)
        self.assertAlmostEqual(step(world, state, Action("query", "weight"))[1], 1 / 3)
        self.assertEqual(step(world, state, Action("bind", "height", "e1"))[1], -1)
        other = World("6", (("e1", "weight", Decimal(73)), ("e2", "height", Decimal("1.8"))))
        self.assertNotEqual(observation(world, state), observation(other, state))
        end, shortcut_reward, _ = step(world, state, Action("finish", value="22.22"))
        self.assertEqual(shortcut_reward, 0)
        self.assertEqual(outcome(end, Decimal(22), Decimal(23)), 1)
        self.assertEqual(outcome(end, Decimal(25), Decimal(26)), -1)


if __name__ == "__main__":
    unittest.main()
