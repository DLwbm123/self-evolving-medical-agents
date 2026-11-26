"""Run with the rollout dependencies installed."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("torch", "transformers")),
                     "rollout dependencies are required")
class BudgetAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
        import sample_model_rollouts
        cls.rollouts = sample_model_rollouts

    def test_hidden_policy_input_and_menu_are_invariant(self):
        m = self.rollouts
        cases, _ = m.synthetic_cases(11, False)
        world = cases["MAP-1"][0]
        def rendered(remaining):
            state = m.State(remaining)
            return m.with_menu(m.state_prompt(world, state, True),
                               m.action_menu(world, state, budget_independent=True))
        self.assertEqual(rendered(7), rendered(10))
        self.assertNotIn('"remaining"', rendered(7))
        self.assertNotEqual(m.observation(world, m.State(7)), m.observation(world, m.State(10)))

    def test_same_targets_and_budget_intervention(self):
        m = self.rollouts
        for recovery in (False, True):
            reference = None
            for mode in ("fixed", "random", "hidden"):
                cases, examples = m.synthetic_cases(11, recovery, budget_mode=mode, independent_menu=True)
                self.assertTrue(all(world.budget == 8 for world, _, _ in cases.values()))
                labels = []
                remaining = set()
                for prompt, index in examples:
                    context, menu = prompt.split("\nMenu:\n")
                    state = json.loads(context.split("\nState: ")[1])
                    remaining.add(state.get("remaining"))
                    labels.append(menu.splitlines()[index].split(". ", 1)[1])
                self.assertEqual(len(labels), 192)
                if reference is None:
                    reference = labels
                self.assertEqual(reference, labels)
                if mode == "hidden":
                    self.assertEqual(remaining, {None})
                else:
                    self.assertEqual(max(remaining), 10 if mode == "random" else 8)
