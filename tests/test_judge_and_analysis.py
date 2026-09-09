"""Tests for model-based grading and failure analysis.

Both features ask a model for an opinion, which makes them the two places most
likely to quietly manufacture confidence. Most of these tests are about the
failure modes rather than the happy path.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb import prompts
from agentwb.analysis import failure_analyzer
from agentwb.evals.graders.base import GradingContext, run_grader
from agentwb.evals.graders.llm_judge import DIMENSIONS
from agentwb.judge.client import JudgeClient, JudgeError, _extract_json
from agentwb.judge.mock import FakeJudgeProvider, verdict
from agentwb.types import (
    Evaluation,
    GraderResult,
    GraderSpec,
    GraderVerdict,
    Step,
    Task,
    TerminationReason,
    Trajectory,
)


def make_ctx(judge=None, final_text="I fixed the bug and the tests pass.",
             steps=None, evaluation=None) -> GradingContext:
    task = Task(id="t", prompt="Fix the bug in calculator.py",
                success_criteria="The suite passes.")
    traj = Trajectory(trajectory_id="r1", task_id="t")
    traj.final_output = {"text": final_text}
    traj.steps = steps or []
    traj.evaluation = evaluation
    return GradingContext(task=task, trajectory=traj, workspace=Path("."), judge=judge)


def judge_with(*replies) -> JudgeClient:
    return JudgeClient(FakeJudgeProvider(list(replies)))


def grade(ctx, **params) -> GraderResult:
    params.setdefault("dimension", "groundedness")
    return run_grader(GraderSpec(type="llm_judge", params=params), ctx)


class TestJsonExtraction(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(_extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(_extract_json('here you go:\n```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_with_surrounding_prose(self):
        self.assertEqual(_extract_json('Sure! {"a": 1} hope that helps'), {"a": 1})

    def test_prose_only_returns_none(self):
        self.assertIsNone(_extract_json("I think it looks pretty good overall."))

    def test_malformed_json_is_not_repaired(self):
        self.assertIsNone(_extract_json('{"a": 1,}garbage{'))

    def test_json_array_is_rejected(self):
        self.assertIsNone(_extract_json('[1, 2, 3]'))


class TestJudgeGrader(unittest.TestCase):
    def test_high_score_passes(self):
        result = grade(make_ctx(judge_with(verdict(0.9))))
        self.assertIs(result.verdict, GraderVerdict.PASS)
        self.assertEqual(result.score, 0.9)

    def test_low_score_fails(self):
        result = grade(make_ctx(judge_with(verdict(0.2, "FAIL"))))
        self.assertIs(result.verdict, GraderVerdict.FAIL)

    def test_threshold_is_configurable_and_decides(self):
        ctx = make_ctx(judge_with(verdict(0.6, "PASS")))
        self.assertIs(grade(ctx, threshold=0.5).verdict, GraderVerdict.PASS)
        ctx = make_ctx(judge_with(verdict(0.6, "PASS")))
        self.assertIs(grade(ctx, threshold=0.8).verdict, GraderVerdict.FAIL)

    def test_threshold_overrides_the_models_own_verdict(self):
        """The configured number governs the pass line, not model mood."""
        result = grade(make_ctx(judge_with(verdict(0.2, "PASS"))), threshold=0.7)
        self.assertIs(result.verdict, GraderVerdict.FAIL)
        meta = result.evidence[0]
        self.assertIn("threshold_override", meta)

    def test_no_judge_configured_returns_unknown(self):
        result = grade(make_ctx(judge=None))
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("no judge configured", result.error)

    def test_unparseable_reply_returns_unknown(self):
        result = grade(make_ctx(judge_with("I'd say it's pretty good, maybe 8/10?")))
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("parseable JSON", result.error)

    def test_score_without_evidence_returns_unknown(self):
        """A verdict with nothing to point at is an opinion, not a measurement."""
        result = grade(make_ctx(judge_with({"score": 0.95, "verdict": "PASS", "evidence": []})))
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("no supporting evidence", result.error)

    def test_judge_may_answer_unknown(self):
        result = grade(make_ctx(judge_with(verdict(0.0, "UNKNOWN"))))
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)

    def test_missing_score_returns_unknown(self):
        result = grade(make_ctx(judge_with({"verdict": "PASS",
                                            "evidence": [{"quote": "x", "why": "y"}]})))
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)

    def test_empty_output_returns_unknown(self):
        result = grade(make_ctx(judge_with(verdict(0.9)), final_text=""))
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("no material", result.error)

    def test_unknown_dimension_returns_unknown(self):
        result = grade(make_ctx(judge_with(verdict(0.9))), dimension="vibes")
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("vibes", result.error)

    def test_custom_rubric_is_accepted(self):
        result = grade(make_ctx(judge_with(verdict(0.9))),
                       dimension=None, rubric="Is it polite?", name="politeness")
        self.assertIs(result.verdict, GraderVerdict.PASS)

    def test_evidence_records_prompt_version_and_model(self):
        result = grade(make_ctx(judge_with(verdict(0.9))))
        meta = result.evidence[0]
        self.assertEqual(meta["prompt_id"], "judge_groundedness:v1")
        self.assertEqual(meta["model"], "fake-judge-v1")

    def test_score_is_clamped_to_unit_range(self):
        result = grade(make_ctx(judge_with(verdict(87))))
        self.assertEqual(result.score, 1.0)


class TestDimensionIsolation(unittest.TestCase):
    def test_every_dimension_has_its_own_prompt_version(self):
        ids = {d.id for d in DIMENSIONS.values()}
        self.assertEqual(len(ids), len(DIMENSIONS))
        for pid in ids:
            self.assertTrue(pid.endswith(":v1"), pid)

    def test_dimension_rubric_reaches_the_judge(self):
        provider = FakeJudgeProvider([verdict(0.9)])
        grade(make_ctx(JudgeClient(provider)), dimension="coverage")
        sent = provider.prompts[0]
        self.assertIn("DIMENSION: coverage", sent)
        self.assertIn(DIMENSIONS["coverage"].text[:40], sent)

    def test_groundedness_can_see_the_trajectory(self):
        provider = FakeJudgeProvider([verdict(0.9)])
        steps = [Step(step=1, action={"type": "tool_call", "tool": "run_tests"},
                      observation={"summary": "3 passed"})]
        grade(make_ctx(JudgeClient(provider), steps=steps), include="both")
        sent = provider.prompts[0]
        self.assertIn("WHAT THE AGENT ACTUALLY OBSERVED", sent)
        self.assertIn("3 passed", sent)


class TestPromptRegistry(unittest.TestCase):
    def test_prompts_are_immutable(self):
        p = prompts.VersionedPrompt(name="t_immutable", version="v1", text="original")
        prompts.register(p)
        prompts.register(p)  # identical re-registration is fine
        with self.assertRaises(prompts.PromptConflict):
            prompts.register(prompts.VersionedPrompt(name="t_immutable", version="v1",
                                                     text="edited"))

    def test_lineage_walks_to_the_root(self):
        prompts.register(prompts.VersionedPrompt(name="t_lin", version="v1", text="a"))
        prompts.register(prompts.VersionedPrompt(name="t_lin", version="v2", text="b",
                                                 parent="t_lin:v1", optimizer="GEPA"))
        self.assertEqual(prompts.lineage("t_lin:v2"), ["t_lin:v2", "t_lin:v1"])


class TestFailureAnalysis(unittest.TestCase):
    @staticmethod
    def _failed_traj(steps=None, reason=TerminationReason.AGENT_FINISHED.value,
                     metrics=None, final="done", grader_verdict=GraderVerdict.FAIL):
        t = Trajectory(trajectory_id="r1", task_id="t")
        t.steps = steps or []
        t.termination_reason = reason
        t.metrics = metrics or {}
        t.final_output = {"text": final}
        t.evaluation = Evaluation(
            passed=False, score=0.0,
            results=[GraderResult(grader="suite", verdict=grader_verdict, required=True)],
        )
        return t

    @staticmethod
    def _task():
        """A code task -- it declares a tests_pass grader, which is what tells
        the analyzer that 'the agent never ran the tests' is a relevant finding."""
        return Task(id="t", prompt="Fix the bug", success_criteria="Tests pass",
                    graders=[GraderSpec(type="tests_pass")])

    def test_passing_run_is_not_analysed(self):
        t = Trajectory(trajectory_id="r1", task_id="t")
        t.evaluation = Evaluation(passed=True, score=1.0)
        a = failure_analyzer.analyze(self._task(), t)
        self.assertIn("nothing to analyse", a.root_causes[0])

    def test_edited_without_testing_is_detected(self):
        steps = [Step(step=1, action={"type": "tool_call", "tool": "edit_file",
                                      "arguments": {"path": "a.py"}},
                      tool_result={"ok": True})]
        a = failure_analyzer.analyze(self._task(), self._failed_traj(steps))
        self.assertTrue(any("never ran the tests" in c for c in a.root_causes))
        self.assertTrue(a.optimizer_candidate)
        self.assertIn("run_tests", a.recommended_regression_test)

    def test_no_file_modified_is_detected(self):
        steps = [Step(step=1, action={"type": "tool_call", "tool": "list_files"},
                      tool_result={"ok": True})]
        a = failure_analyzer.analyze(self._task(), self._failed_traj(steps))
        self.assertTrue(any("never modified any file" in c for c in a.root_causes))

    def test_tool_error_codes_produce_specific_fixes(self):
        steps = [Step(step=1, action={"type": "tool_call", "tool": "edit_file"},
                      tool_result={"ok": False, "error": "NO_MATCH"})]
        a = failure_analyzer.analyze(self._task(), self._failed_traj(steps))
        self.assertTrue(any("NO_MATCH" in c for c in a.root_causes))
        self.assertIn("read_file_region", a.proposed_fix)
        self.assertEqual(a.critical_step, 1)

    def test_environment_error_is_marked_unavoidable(self):
        a = failure_analyzer.analyze(
            self._task(),
            self._failed_traj(reason=TerminationReason.ENVIRONMENT_ERROR.value))
        self.assertFalse(a.avoidable)

    def test_unknown_grader_flags_the_eval_not_the_agent(self):
        t = self._failed_traj(grader_verdict=GraderVerdict.UNKNOWN)
        a = failure_analyzer.analyze(self._task(), t)
        self.assertTrue(any("grading was inconclusive" in c for c in a.root_causes))
        self.assertIn("fix the eval", a.proposed_fix)

    def test_without_a_judge_the_source_is_rules(self):
        a = failure_analyzer.analyze(self._task(), self._failed_traj())
        self.assertEqual(a.source, "rules")
        self.assertLessEqual(a.confidence, 0.5)

    def test_with_a_judge_the_source_is_model_and_rules_are_kept(self):
        judge = judge_with({
            "root_causes": ["the agent misread the assertion"],
            "critical_step": 2, "avoidable": True,
            "proposed_fix": "quote the failing assertion in the observation",
            "recommended_regression_test": "task with a confusing assertion",
            "optimizer_candidate": True,
            "categories": ["TASK_UNDERSTANDING", "not_a_real_category"],
            "confidence": 0.8,
        })
        a = failure_analyzer.analyze(self._task(), self._failed_traj(), judge=judge)
        self.assertEqual(a.source, "model")
        self.assertEqual(a.critical_step, 2)
        self.assertEqual(a.confidence, 0.8)
        self.assertIn("the agent misread the assertion", a.root_causes)
        # deterministic findings survive alongside the model's
        self.assertTrue(any(c.startswith("(deterministic)") for c in a.root_causes))
        # invented categories are dropped
        self.assertIn("TASK_UNDERSTANDING", a.categories)
        self.assertNotIn("not_a_real_category", a.categories)

    def test_unparseable_judge_falls_back_to_rules(self):
        a = failure_analyzer.analyze(self._task(), self._failed_traj(),
                                     judge=judge_with("no idea honestly"))
        self.assertEqual(a.source, "rules")
        self.assertIn("falling back", a.error)

    def test_deterministic_facts_are_sent_to_the_judge(self):
        provider = FakeJudgeProvider([{"root_causes": ["x"], "confidence": 0.5}])
        steps = [Step(step=1, action={"type": "tool_call", "tool": "edit_file"},
                      tool_result={"ok": True})]
        failure_analyzer.analyze(self._task(), self._failed_traj(steps),
                                 judge=JudgeClient(provider))
        sent = provider.prompts[0]
        self.assertIn("DETERMINISTIC FACTS", sent)
        self.assertIn("FAILURE TAXONOMY", sent)
        self.assertIn("ran_tests", sent)

    def test_analysis_round_trips_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = failure_analyzer.analyze(self._task(), self._failed_traj())
            failure_analyzer.save(a, Path(tmp))
            loaded = failure_analyzer.load(Path(tmp))
            self.assertEqual(loaded.to_dict(), a.to_dict())

    def test_research_task_is_not_scolded_for_skipping_tests(self):
        """Regression: the analyzer blamed a prose task for never running tests."""
        research = Task(id="r", prompt="Write findings.md",
                        graders=[GraderSpec(type="file_exists", params={"path": "findings.md"})])
        steps = [Step(step=1, action={"type": "tool_call", "tool": "write_file",
                                      "arguments": {"path": "findings.md"}},
                      tool_result={"ok": True})]
        a = failure_analyzer.analyze(research, self._failed_traj(steps))
        self.assertFalse(any("never ran the tests" in c for c in a.root_causes))

    def test_code_task_is_still_scolded_for_skipping_tests(self):
        code = Task(id="c", prompt="Fix it",
                    graders=[GraderSpec(type="tests_pass")])
        steps = [Step(step=1, action={"type": "tool_call", "tool": "edit_file",
                                      "arguments": {"path": "a.py"}},
                      tool_result={"ok": True})]
        a = failure_analyzer.analyze(code, self._failed_traj(steps))
        self.assertTrue(any("never ran the tests" in c for c in a.root_causes))

    def test_empty_model_reply_does_not_claim_model_source(self):
        """Regression: a judge that returned nothing usable was still credited."""
        a = failure_analyzer.analyze(self._task(), self._failed_traj(),
                                     judge=judge_with({"confidence": 0.9}))
        self.assertEqual(a.source, "rules")
        self.assertIn("no root causes", a.error)

    def test_load_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(failure_analyzer.load(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
