"""Tests for `report`, `retrieve`, and the two multi-agent metrics.

The report's job is to be read by someone who did not watch the run, which
makes it the place where an overstated number does the most damage. Several of
these pin what it must *not* claim.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb.cli.main import _unverified
from agentwb.protocols.schemas import AgentMessage, Evidence, MessageType
from agentwb.runtime.orchestrator import Exchange
from agentwb.types import Evaluation, GraderResult, GraderVerdict, Step, Trajectory


def traj_with(steps=None, evaluation=None, retrieval=None) -> Trajectory:
    t = Trajectory(trajectory_id="r1", task_id="t")
    t.steps = steps or []
    t.evaluation = evaluation
    t.retrieval = retrieval or {}
    return t


def tool_step(n, tool, ok=True, error=None) -> Step:
    return Step(step=n, action={"type": "tool_call", "tool": tool, "arguments": {}},
                tool_result={"ok": ok, "error": error})


class TestUnverifiedSection(unittest.TestCase):
    def test_unknown_graders_are_listed(self):
        ev = Evaluation(passed=False, results=[
            GraderResult(grader="groundedness", verdict=GraderVerdict.UNKNOWN,
                         error="no judge configured")])
        lines = _unverified(traj_with(evaluation=ev), ev)
        self.assertTrue(any("groundedness returned UNKNOWN" in l for l in lines))

    def test_never_running_the_tests_is_called_out(self):
        ev = Evaluation(passed=True)
        lines = _unverified(traj_with(steps=[tool_step(1, "read_file_region")], evaluation=ev), ev)
        self.assertTrue(any("never ran the test suite" in l for l in lines))

    def test_a_failure_is_reported_as_established_not_unverified(self):
        """A failed run saying 'no gaps detected' would read as reassurance."""
        ev = Evaluation(passed=False, results=[
            GraderResult(grader="suite_green", verdict=GraderVerdict.FAIL, required=True)])
        lines = _unverified(traj_with(steps=[tool_step(1, "run_tests")], evaluation=ev), ev)
        self.assertTrue(any("established, not unverified" in l for l in lines))

    def test_an_ungraded_run_establishes_nothing(self):
        lines = _unverified(traj_with(steps=[tool_step(1, "run_tests")]), None)
        self.assertTrue(any("nothing was graded" in l for l in lines))

    def test_the_section_is_never_empty(self):
        """A run claiming to have verified everything is the one to doubt."""
        ev = Evaluation(passed=True)
        lines = _unverified(traj_with(steps=[tool_step(1, "run_tests")], evaluation=ev), ev)
        self.assertTrue(lines)


class TestExchangeMetrics(unittest.TestCase):
    @staticmethod
    def _msg(mid, mtype, claim, sender="researcher", evidence=None) -> AgentMessage:
        return AgentMessage(sender=sender, recipient="engineer", type=mtype,
                            claim=claim, message_id=mid, evidence=evidence or [])

    def test_final_report_without_evidence_is_a_verification_failure(self):
        ex = Exchange(task_id="t")
        ex.messages = [self._msg("m1", MessageType.FINAL_REPORT, "all done", "engineer")]
        self.assertEqual(ex.metrics["verification_failures"], 1)

    def test_final_report_with_evidence_is_not(self):
        ex = Exchange(task_id="t")
        ex.messages = [self._msg("m1", MessageType.FINAL_REPORT, "done", "engineer",
                                 evidence=[Evidence("tests", "12 passed")])]
        self.assertEqual(ex.metrics["verification_failures"], 0)

    def test_restating_a_ruled_against_claim_counts_as_ignored_evidence(self):
        """A settled conflict that one side keeps arguing with."""
        ex = Exchange(task_id="t")
        losing = self._msg("m_lose", MessageType.HYPOTHESIS, "an index was dropped")
        winning = self._msg("m_win", MessageType.HYPOTHESIS, "the pool is exhausted",
                            sender="engineer")
        restated = self._msg("m_again", MessageType.CRITIQUE, "an index was dropped")
        ex.messages = [winning, losing, restated]
        ex.disagreements = [{"a": "m_win", "b": "m_lose",
                             "outcome": {"resolution": "RESOLVED_BY_EXPERIMENT",
                                         "winner": "m_win"}}]
        self.assertEqual(ex.metrics["ignored_evidence"], 1)

    def test_an_unresolved_disagreement_produces_no_ignored_evidence(self):
        """Nothing was ruled against, so nothing can be ignored."""
        ex = Exchange(task_id="t")
        a = self._msg("m1", MessageType.HYPOTHESIS, "an index was dropped")
        b = self._msg("m2", MessageType.HYPOTHESIS, "the pool is exhausted", sender="engineer")
        ex.messages = [a, b, self._msg("m3", MessageType.CRITIQUE, "an index was dropped")]
        ex.disagreements = [{"a": "m1", "b": "m2",
                             "outcome": {"resolution": "HUMAN_REQUIRED"}}]
        self.assertEqual(ex.metrics["ignored_evidence"], 0)

    def test_a_clean_exchange_scores_zero_on_both(self):
        ex = Exchange(task_id="t")
        ex.messages = [self._msg("m1", MessageType.HYPOTHESIS, "the pool is exhausted",
                                 evidence=[Evidence("metrics", "100/100")])]
        self.assertEqual(ex.metrics["ignored_evidence"], 0)
        self.assertEqual(ex.metrics["verification_failures"], 0)

    def test_both_metrics_are_always_present(self):
        m = Exchange(task_id="t").metrics
        self.assertIn("ignored_evidence", m)
        self.assertIn("verification_failures", m)


if __name__ == "__main__":
    unittest.main()
