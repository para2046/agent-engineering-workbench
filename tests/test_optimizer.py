"""Policy optimization tests.

The invariant guard gets the most attention here, because the failure it
prevents is the one an optimizer is actively incentivised to commit: deleting
"verify before claiming success" from a system prompt reliably raises a naive
success metric, and the resulting system looks better while being worse.
"""

from __future__ import annotations

import unittest

from agentwb.judge.client import JudgeClient
from agentwb.judge.mock import FakeJudgeProvider
from agentwb.optimization.optimizer import (
    PROTECTED_INVARIANTS,
    OptimizationResult,
    PolicyCandidate,
    ReflectiveOptimizer,
    check_invariants,
    feedback_from_analyses,
    guard,
    optimize,
)
from agentwb.optimization.promotion import GateConfig, PromotionGate, SplitResult
from agentwb.prompts import VersionedPrompt, register

GOOD_PROMPT = (
    "You are an engineering agent. Verify claims against the environment and never "
    "report success you have not observed. You do not decide whether the task passed; "
    "graders inspect the environment afterwards."
)

# Rephrased but semantically intact -- an optimizer is allowed to do this.
REPHRASED = (
    "Work carefully. Check your work by observing the environment directly rather than "
    "assuming; do not claim a result you did not verify. Note that graders, not you, "
    "determine the outcome -- you do not decide."
)

# The dangerous edit: fluent, plausible, and quietly drops both commitments.
GUTTED = (
    "You are an engineering agent. Work efficiently and confidently. When you have "
    "finished, state clearly that the task is complete so the run can be scored."
)


def baseline_prompt() -> VersionedPrompt:
    return register(VersionedPrompt(name="test_policy", version="v1", text=GOOD_PROMPT))


def judge_returning(*candidates) -> JudgeClient:
    return JudgeClient(FakeJudgeProvider([{"candidates": list(candidates)}]))


def split(passed=30, trials=40, passed_ids=None, failed_ids=None, name="dev") -> SplitResult:
    return SplitResult(split=name, tasks=trials, trials=trials, passed=passed,
                       passed_task_ids=set(passed_ids or []),
                       failed_task_ids=set(failed_ids or []))


class TestInvariantGuard(unittest.TestCase):
    def test_intact_prompt_has_no_missing_invariants(self):
        self.assertEqual(check_invariants(GOOD_PROMPT), [])

    def test_rephrasing_is_allowed(self):
        """An optimizer may rewrite freely as long as the commitment survives."""
        self.assertEqual(check_invariants(REPHRASED), [])

    def test_gutted_prompt_is_caught(self):
        missing = check_invariants(GUTTED)
        self.assertIn("no_unverified_success", missing)
        self.assertIn("grading_is_external", missing)

    def test_guard_marks_candidate_rejected_with_a_reason(self):
        c = guard(PolicyCandidate(name="p", version="v2", text=GUTTED))
        self.assertFalse(c.viable)
        self.assertIn("no_unverified_success", c.rejected_reason)
        self.assertIn("verify", c.rejected_reason)

    def test_guard_passes_a_viable_candidate(self):
        self.assertTrue(guard(PolicyCandidate(name="p", version="v2", text=REPHRASED)).viable)

    def test_empty_prompt_is_rejected(self):
        self.assertFalse(guard(PolicyCandidate(name="p", version="v2", text="")).viable)

    def test_every_protected_invariant_has_keywords(self):
        for name, keywords in PROTECTED_INVARIANTS.items():
            self.assertTrue(keywords, f"{name} has no keywords, so it can never be enforced")


class TestReflectiveOptimizer(unittest.TestCase):
    def test_proposes_candidates_with_lineage(self):
        opt = ReflectiveOptimizer(judge_returning(
            {"rationale": "address unverified claims", "prompt": REPHRASED}))
        [c] = opt.propose(baseline_prompt(), "agents kept skipping tests", n=1)
        self.assertTrue(c.viable)
        self.assertEqual(c.parent, "test_policy:v1")
        self.assertEqual(c.optimizer, "reflective")
        self.assertIn("unverified", c.rationale)

    def test_gutted_candidate_comes_back_rejected(self):
        opt = ReflectiveOptimizer(judge_returning({"rationale": "be concise", "prompt": GUTTED}))
        [c] = opt.propose(baseline_prompt(), "feedback", n=1)
        self.assertFalse(c.viable)

    def test_unparseable_optimizer_reply_proposes_nothing(self):
        """It never falls back to editing the prompt itself."""
        opt = ReflectiveOptimizer(JudgeClient(FakeJudgeProvider(["I'd suggest being clearer"])))
        self.assertEqual(opt.propose(baseline_prompt(), "feedback", n=2), [])

    def test_empty_prompt_text_is_skipped(self):
        opt = ReflectiveOptimizer(judge_returning({"rationale": "x", "prompt": "  "}))
        self.assertEqual(opt.propose(baseline_prompt(), "feedback", n=1), [])

    def test_versions_are_distinct_per_candidate(self):
        opt = ReflectiveOptimizer(judge_returning(
            {"prompt": REPHRASED}, {"prompt": REPHRASED + " Also be brief."}))
        cands = opt.propose(baseline_prompt(), "feedback", n=2)
        self.assertEqual(len({c.version for c in cands}), 2)

    def test_optimizer_prompt_tells_the_model_the_guard_exists(self):
        provider = FakeJudgeProvider([{"candidates": [{"prompt": REPHRASED}]}])
        ReflectiveOptimizer(JudgeClient(provider)).propose(baseline_prompt(), "feedback", 1)
        sent = provider.prompts[0]
        self.assertIn("CURRENT PROMPT", sent)
        self.assertIn("FAILURE EVIDENCE", sent)


class TestFeedbackAssembly(unittest.TestCase):
    def test_analyses_become_readable_feedback(self):
        class A:
            task_id = "fix_bug"
            root_causes = ["edited files but never ran the tests"]
            proposed_fix = "strengthen the verification instruction"

        text = feedback_from_analyses([A()])
        self.assertIn("fix_bug", text)
        self.assertIn("never ran the tests", text)
        self.assertIn("strengthen", text)

    def test_no_analyses_says_so_rather_than_returning_empty(self):
        self.assertIn("no failure analyses", feedback_from_analyses([]))


class TestOptimizeLoop(unittest.TestCase):
    def setUp(self):
        self.gate = PromotionGate(GateConfig(min_trials_for_confidence=1))

    def _evaluate_factory(self, candidate_passed=38, baseline_passed=30):
        def evaluate(policy, tasks):
            if policy is None:
                return split(passed=baseline_passed, passed_ids={"a"})
            return split(passed=candidate_passed, passed_ids={"a", "b"})
        return evaluate

    def test_improving_candidate_is_promoted(self):
        opt = ReflectiveOptimizer(judge_returning({"prompt": REPHRASED}))
        result = optimize(
            baseline=baseline_prompt(), optimizer=opt, feedback="f",
            evaluate=self._evaluate_factory(), gate=self.gate,
            dev_tasks=["t"], test_tasks=["t"], n_candidates=1,
        )
        self.assertIsNotNone(result.promoted)
        self.assertIn("promoted", result.summary())

    def test_non_improving_candidate_is_not_promoted(self):
        opt = ReflectiveOptimizer(judge_returning({"prompt": REPHRASED}))
        result = optimize(
            baseline=baseline_prompt(), optimizer=opt, feedback="f",
            evaluate=self._evaluate_factory(candidate_passed=20), gate=self.gate,
            dev_tasks=["t"], test_tasks=["t"], n_candidates=1,
        )
        self.assertIsNone(result.promoted)
        self.assertIn("baseline stands", result.summary())

    def test_gutted_candidate_is_never_evaluated(self):
        """Rejected before scoring, so no number can later argue for it."""
        calls = []

        def evaluate(policy, tasks):
            calls.append(policy)
            return split(passed=40, passed_ids={"a"})

        opt = ReflectiveOptimizer(judge_returning({"prompt": GUTTED}))
        result = optimize(baseline=baseline_prompt(), optimizer=opt, feedback="f",
                          evaluate=evaluate, gate=self.gate, dev_tasks=["t"], n_candidates=1)
        self.assertEqual(calls, [], "a rejected candidate must never be scored")
        self.assertEqual(result.promoted, None)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("rejected before scoring", " ".join(result.notes))

    def test_no_candidates_is_reported_not_crashed(self):
        opt = ReflectiveOptimizer(JudgeClient(FakeJudgeProvider(["nope"])))
        result = optimize(baseline=baseline_prompt(), optimizer=opt, feedback="f",
                          evaluate=self._evaluate_factory(), gate=self.gate,
                          dev_tasks=["t"], n_candidates=2)
        self.assertIsNone(result.promoted)
        self.assertIn("proposed no candidates", " ".join(result.notes))

    def test_regression_blocks_promotion_through_the_loop(self):
        def evaluate(policy, tasks):
            if policy is None:
                return split(passed=20, passed_ids={"keeper"})
            return split(passed=39, passed_ids={"other"}, failed_ids={"keeper"})

        opt = ReflectiveOptimizer(judge_returning({"prompt": REPHRASED}))
        result = optimize(baseline=baseline_prompt(), optimizer=opt, feedback="f",
                          evaluate=evaluate, gate=self.gate,
                          dev_tasks=["t"], test_tasks=["t"], n_candidates=1)
        self.assertIsNone(result.promoted)

    def test_promoted_candidate_is_registered_with_its_parent(self):
        from agentwb import prompts

        opt = ReflectiveOptimizer(judge_returning({"prompt": REPHRASED}))
        result = optimize(baseline=baseline_prompt(), optimizer=opt, feedback="f",
                          evaluate=self._evaluate_factory(), gate=self.gate,
                          dev_tasks=["t"], test_tasks=["t"], n_candidates=1)
        lineage = prompts.lineage(result.promoted)
        self.assertEqual(lineage[-1], "test_policy:v1")
        self.assertEqual(prompts.get(result.promoted).optimizer, "reflective")

    def test_result_serialises(self):
        r = OptimizationResult(baseline_id="p:v1")
        self.assertIn("baseline", r.to_dict())


class TestOptionalOptimizerAdapters(unittest.TestCase):
    def test_gepa_without_the_library_says_what_to_install(self):
        from agentwb.optimization.gepa_optimizer import GEPAOptimizer
        from agentwb.providers.base import ProviderError

        try:
            import gepa  # noqa: F401
            self.skipTest("gepa is installed")
        except ImportError:
            pass
        with self.assertRaises(ProviderError) as cm:
            GEPAOptimizer()
        self.assertIn("pip install gepa", str(cm.exception))

    def test_dspy_without_the_library_says_what_to_install(self):
        from agentwb.optimization.dspy_optimizer import DSPyOptimizer
        from agentwb.providers.base import ProviderError

        try:
            import dspy  # noqa: F401
            self.skipTest("dspy is installed")
        except ImportError:
            pass
        with self.assertRaises(ProviderError):
            DSPyOptimizer(metric=lambda *a: 1.0)


if __name__ == "__main__":
    unittest.main()
