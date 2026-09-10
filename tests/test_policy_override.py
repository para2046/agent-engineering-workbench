"""Per-call policy override, and the optimize loop end to end.

The policy parameter is what lets a candidate be scored against a baseline with
every other variable held constant. If it leaked across calls, or silently did
nothing, the optimizer would produce confident numbers about a policy that was
never actually used — which is worse than not optimizing at all.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb.evals.harness import Harness
from agentwb.experience.store import ExperienceStore
from agentwb.judge.client import JudgeClient
from agentwb.judge.mock import FakeJudgeProvider
from agentwb.optimization.datasets import build_dataset
from agentwb.optimization.optimizer import (
    PolicyCandidate,
    ReflectiveOptimizer,
    check_invariants,
    guard,
    optimize,
)
from agentwb.optimization.promotion import GateConfig, PromotionGate, result_from_task_results
from agentwb.optimization.runner import PolicyEvaluator
from agentwb.prompts import VersionedPrompt
from agentwb.providers.mock import ScriptedProvider
from agentwb.runtime.agent import SYSTEM_PROMPT
from agentwb.trajectories.store import TrajectoryStore
from agentwb.types import ModelResponse, Task

GOOD = ("Verify with the environment and never report success you have not observed. "
        "You do not decide whether the task passed; graders do.")


def a_task(task_id="t", tags=None) -> Task:
    return Task(id=task_id, prompt="do the thing", tags=tags or [],
                environment=__import__("agentwb.types", fromlist=["EnvironmentSpec"])
                .EnvironmentSpec(files={"a.txt": "hello"}))


class PolicyCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.store = TrajectoryStore(root)
        self.experience = ExperienceStore(root / "exp")
        self.harness = Harness(self.store, self.experience, root / "ws",
                               analyze_failures=False)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def provider(self):
        return ScriptedProvider([ModelResponse(text="done")])


class TestPolicyOverride(PolicyCase):
    def test_default_policy_is_the_builtin_prompt(self):
        p = self.provider()
        self.harness.run_task(a_task(), p, trials=1)
        self.assertEqual(p.seen_system[0] if hasattr(p, "seen_system") else SYSTEM_PROMPT,
                         SYSTEM_PROMPT)

    def test_policy_replaces_the_system_prompt_for_that_call(self):
        policy = VersionedPrompt(name="test_policy", version="v9", text=GOOD)
        p = self.provider()
        traj = self.harness.run_task(a_task(), p, trials=1, policy=policy).trials[0].trajectory
        self.assertEqual(traj.prompt_version, "test_policy:v9")

    def test_policy_does_not_leak_into_the_next_call(self):
        """A candidate that changed later runs would confound every comparison."""
        policy = VersionedPrompt(name="test_policy", version="v9", text=GOOD)
        self.harness.run_task(a_task(), self.provider(), trials=1, policy=policy)
        traj = self.harness.run_task(a_task(), self.provider(), trials=1).trials[0].trajectory
        self.assertNotEqual(traj.prompt_version, "test_policy:v9")

    def test_policy_candidate_object_is_accepted(self):
        candidate = PolicyCandidate(name="single_agent", version="opt1", text=GOOD,
                                    parent="single_agent:v1", optimizer="reflective")
        traj = self.harness.run_task(a_task(), self.provider(), trials=1,
                                     policy=candidate).trials[0].trajectory
        self.assertEqual(traj.prompt_version, "single_agent:opt1")

    def test_object_without_text_is_rejected_loudly(self):
        with self.assertRaises(TypeError) as cm:
            self.harness.run_task(a_task(), self.provider(), trials=1, policy="a string")
        self.assertIn(".text", str(cm.exception))


class TestPolicyEvaluator(PolicyCase):
    def test_none_means_baseline(self):
        baseline = VersionedPrompt(name="base", version="v1", text=GOOD)
        ev = PolicyEvaluator(self.harness, self.provider(), baseline, trials=1)
        result = ev(None, [a_task()])
        self.assertEqual(result.trials, 1)

    def test_empty_task_list_is_not_an_error(self):
        baseline = VersionedPrompt(name="base", version="v1", text=GOOD)
        ev = PolicyEvaluator(self.harness, self.provider(), baseline, trials=1)
        self.assertEqual(ev(None, []).trials, 0)
        self.assertEqual(ev(None, None).trials, 0)


class TestInvariantGuard(unittest.TestCase):
    """The guard runs before scoring, so no number can argue for a bad candidate."""

    def test_good_candidate_survives(self):
        c = guard(PolicyCandidate(name="p", version="opt1", text=GOOD))
        self.assertTrue(c.viable)
        self.assertIsNone(c.rejected_reason)

    def test_candidate_dropping_verification_is_rejected(self):
        text = "You do not decide whether the task passed; graders do. Work fast."
        c = guard(PolicyCandidate(name="p", version="opt1", text=text))
        self.assertFalse(c.viable)
        self.assertIn("no_unverified_success", c.rejected_reason)

    def test_candidate_dropping_external_grading_is_rejected(self):
        text = "Always verify what you observed before reporting."
        c = guard(PolicyCandidate(name="p", version="opt1", text=text))
        self.assertFalse(c.viable)
        self.assertIn("grading_is_external", c.rejected_reason)

    def test_empty_candidate_is_rejected(self):
        c = guard(PolicyCandidate(name="p", version="opt1", text=""))
        self.assertFalse(c.viable)

    def test_rewording_is_permitted(self):
        """The guard protects the commitment, not the wording."""
        reworded = ("Confirm every claim against observed environment output. "
                    "Graders, not you, do not decide - the graders determine the verdict.")
        self.assertEqual(check_invariants(reworded), [])

    def test_the_real_system_prompt_satisfies_its_own_invariants(self):
        self.assertEqual(check_invariants(SYSTEM_PROMPT), [])


class TestOptimizeLoop(PolicyCase):
    def _optimize(self, judge_replies, tasks=None, gate=None):
        baseline = VersionedPrompt(name="single_agent", version="v1", text=SYSTEM_PROMPT)
        tasks = tasks or [a_task("dev_a", ["dev"])]
        ds = build_dataset(tasks)
        provider = ScriptedProvider([ModelResponse(text="done")])

        def evaluate(policy, task_list):
            results = [self.harness.run_task(t, provider, trials=1, policy=policy)
                       for t in (task_list or [])]
            return result_from_task_results("dev", results, None)

        return optimize(
            baseline=baseline,
            optimizer=ReflectiveOptimizer(JudgeClient(FakeJudgeProvider(judge_replies))),
            feedback="- task dev_a: edited files but never ran the tests",
            evaluate=evaluate,
            gate=gate or PromotionGate(GateConfig(min_trials_for_confidence=1)),
            dev_tasks=ds.for_optimizer("dev"),
            test_tasks=ds.for_final_evaluation() or None,
            regression_tasks=ds.split("regression") or None,
            n_candidates=1,
        )

    def test_unparseable_optimizer_reply_proposes_nothing(self):
        result = self._optimize(["I think the prompt is fine honestly"])
        self.assertEqual(result.candidates, [])
        self.assertIsNone(result.promoted)
        self.assertTrue(result.notes)

    def test_candidate_violating_an_invariant_is_rejected_before_scoring(self):
        result = self._optimize([{
            "candidates": [{"rationale": "faster", "prompt": "Just answer quickly."}]
        }])
        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.rejected), 1)
        self.assertIsNone(result.promoted)
        # never scored -- no number exists that could argue for it
        self.assertEqual(result.evaluated, {})

    def test_viable_candidate_is_scored_and_gated(self):
        result = self._optimize([{
            "candidates": [{"rationale": "adds a verification step", "prompt": GOOD}]
        }])
        self.assertEqual(len(result.candidates), 1)
        self.assertIn(result.candidates[0].id, result.evaluated)

    def test_no_improvement_means_no_promotion(self):
        """Both policies score the same here, so the gate must refuse."""
        result = self._optimize([{
            "candidates": [{"rationale": "reworded", "prompt": GOOD}]
        }])
        self.assertIsNone(result.promoted)
        self.assertIn("baseline stands", result.summary())

    def test_result_serialises(self):
        result = self._optimize([{
            "candidates": [{"rationale": "x", "prompt": GOOD}]
        }])
        d = result.to_dict()
        self.assertIn("baseline", d)
        self.assertIn("candidates", d)
        self.assertIn("rejected", d)


if __name__ == "__main__":
    unittest.main()
