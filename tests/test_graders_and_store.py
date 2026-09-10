"""Grader semantics, comparison confound detection, and storage round-trips."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb.evals.graders.base import GradingContext, grader, registered_graders, run_grader
from agentwb.experiments.comparison import compare
from agentwb.trajectories.store import TrajectoryStore
from agentwb.types import (
    Evaluation,
    GraderResult,
    GraderSpec,
    GraderVerdict,
    Step,
    Task,
    Trajectory,
)

GREEN = ("import unittest\n"
         "class T(unittest.TestCase):\n"
         "    def test_a(self): self.assertEqual(1, 1)\n")
RED = ("import unittest\n"
       "class T(unittest.TestCase):\n"
       "    def test_a(self): self.assertEqual(1, 2)\n")


def ctx_for(ws: Path, traj: Trajectory | None = None) -> GradingContext:
    task = Task(id="t", prompt="p")
    return GradingContext(task=task, trajectory=traj or Trajectory(trajectory_id="r", task_id="t"),
                          workspace=ws)


class GraderCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def grade(self, gtype, params=None, traj=None) -> GraderResult:
        return run_grader(GraderSpec(type=gtype, params=params or {}), ctx_for(self.ws, traj))


class TestDeterministicGraders(GraderCase):
    def test_tests_pass_on_green_suite(self):
        (self.ws / "test_ok.py").write_text(GREEN, encoding="utf-8")
        self.assertIs(self.grade("tests_pass").verdict, GraderVerdict.PASS)

    def test_tests_fail_on_red_suite(self):
        (self.ws / "test_bad.py").write_text(RED, encoding="utf-8")
        self.assertIs(self.grade("tests_pass").verdict, GraderVerdict.FAIL)

    def test_empty_suite_is_unknown_not_pass(self):
        """A green exit that collected zero tests proves nothing."""
        result = self.grade("tests_pass")
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIsNotNone(result.error)

    def test_file_exists(self):
        self.assertIs(self.grade("file_exists", {"path": "x.py"}).verdict, GraderVerdict.FAIL)
        (self.ws / "x.py").write_text("", encoding="utf-8")
        self.assertIs(self.grade("file_exists", {"path": "x.py"}).verdict, GraderVerdict.PASS)

    def test_file_contains_and_absent_mode(self):
        (self.ws / "a.py").write_text("raise ValueError('x')\n", encoding="utf-8")
        self.assertIs(self.grade("file_contains",
                                 {"path": "a.py", "text": "raise ValueError"}).verdict,
                      GraderVerdict.PASS)
        self.assertIs(self.grade("file_contains",
                                 {"path": "a.py", "text": "return 0", "absent": True}).verdict,
                      GraderVerdict.PASS)
        self.assertIs(self.grade("file_contains",
                                 {"path": "a.py", "text": "raise ValueError", "absent": True}).verdict,
                      GraderVerdict.FAIL)

    def test_file_contains_regex(self):
        (self.ws / "a.py").write_text("value = 42\n", encoding="utf-8")
        self.assertIs(self.grade("file_contains",
                                 {"path": "a.py", "pattern": r"value\s*=\s*\d+"}).verdict,
                      GraderVerdict.PASS)

    def test_evidence_is_always_structured(self):
        (self.ws / "a.py").write_text("x\n", encoding="utf-8")
        result = self.grade("file_contains", {"path": "a.py", "text": "x"})
        self.assertTrue(result.evidence)
        self.assertIn("source", result.evidence[0])


class TestTrajectoryGraders(GraderCase):
    def _traj_with_tool(self, tool: str, arguments: dict | None = None) -> Trajectory:
        t = Trajectory(trajectory_id="r", task_id="t")
        t.steps = [Step(step=1, action={"type": "tool_call", "tool": tool,
                                        "arguments": arguments or {}})]
        return t

    def test_tool_used_required(self):
        self.assertIs(self.grade("tool_used", {"tool": "run_tests"},
                                 self._traj_with_tool("run_tests")).verdict, GraderVerdict.PASS)
        self.assertIs(self.grade("tool_used", {"tool": "run_tests"},
                                 self._traj_with_tool("list_files")).verdict, GraderVerdict.FAIL)

    def test_tool_forbidden(self):
        result = self.grade("tool_used", {"tool": "shell", "forbidden": True},
                            self._traj_with_tool("shell"))
        self.assertIs(result.verdict, GraderVerdict.FAIL)

    def test_no_forbidden_changes_detects_attempt(self):
        traj = self._traj_with_tool("edit_file", {"path": "test_calculator.py", "old": "a", "new": "b"})
        result = self.grade("no_forbidden_changes", {"paths": ["test_calculator.py"]}, traj)
        self.assertIs(result.verdict, GraderVerdict.FAIL)
        self.assertTrue(result.evidence[0]["violations"])

    def test_no_forbidden_changes_allows_untouched(self):
        traj = self._traj_with_tool("edit_file", {"path": "calculator.py"})
        result = self.grade("no_forbidden_changes", {"paths": ["test_calculator.py"]}, traj)
        self.assertIs(result.verdict, GraderVerdict.PASS)


class TestGraderRobustness(GraderCase):
    def test_crashing_grader_becomes_unknown_not_fail(self):
        @grader("_boom")
        def _boom(ctx, params):
            raise RuntimeError("grader is broken")

        result = self.grade("_boom")
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("grader is broken", result.error)

    def test_unknown_grader_type_lists_alternatives(self):
        result = self.grade("nope")
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertIn("available", result.evidence[0])
        self.assertIn("tests_pass", registered_graders())


class TestComparison(unittest.TestCase):
    @staticmethod
    def _traj(tid, task="t", model="m1", prompt="v1", passed=True, steps=5):
        t = Trajectory(trajectory_id=tid, task_id=task, provider="mock",
                       model=model, prompt_version=prompt)
        t.evaluation = Evaluation(passed=passed, score=1.0 if passed else 0.0)
        t.metrics = {"steps": steps, "tool_calls": steps, "tool_failures": 0,
                     "latency_ms": 10, "input_tokens": 100, "output_tokens": 50}
        return t

    def test_clean_single_variable_change_is_attributable(self):
        base = self._traj("r1", prompt="v1", steps=8)
        cand = self._traj("r2", prompt="v2", steps=5)
        result = compare(base, cand)
        self.assertFalse(result.confounded)
        self.assertIn("prompt_version", result.config_differences)
        self.assertTrue(any("steps" in i for i in result.improvements))

    def test_two_variables_changed_is_confounded(self):
        base = self._traj("r1", model="m1", prompt="v1", steps=8)
        cand = self._traj("r2", model="m2", prompt="v2", steps=5)
        result = compare(base, cand)
        self.assertTrue(result.confounded)
        self.assertTrue(any("CONFOUNDED" in w for w in result.warnings))
        # no improvement may be claimed from a confounded comparison
        self.assertEqual(result.improvements, [])

    def test_different_tasks_are_not_comparable(self):
        result = compare(self._traj("r1", task="a"), self._traj("r2", task="b"))
        self.assertTrue(result.confounded)

    def test_outcome_regression_is_reported(self):
        result = compare(self._traj("r1", passed=True), self._traj("r2", passed=False))
        self.assertIn("outcome: PASS -> FAIL", result.regressions)


class TestStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TrajectoryStore(Path(self._tmp.name))

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def _write(self, tid="run_abc123", passed=True):
        t = Trajectory(trajectory_id=tid, task_id="t", provider="mock", model="m")
        t.evaluation = Evaluation(passed=passed, score=1.0 if passed else 0.0,
                                  results=[GraderResult(grader="g", verdict=GraderVerdict.PASS)])
        t.metrics = {"steps": 3}
        t.steps = [Step(step=1, action={"type": "tool_call", "tool": "run_tests"})]
        self.store.write(t)
        return t

    def test_round_trip_preserves_the_document(self):
        original = self._write()
        loaded = self.store.load(original.trajectory_id)
        self.assertEqual(loaded.to_dict(), original.to_dict())

    def test_prefix_resolution(self):
        self._write("run_abc123")
        self.assertEqual(self.store.resolve("run_abc"), "run_abc123")

    def test_ambiguous_prefix_raises(self):
        self._write("run_aaa1")
        self._write("run_aaa2")
        with self.assertRaises(ValueError):
            self.store.resolve("run_aaa")

    def test_missing_prefix_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.resolve("run_nothing")

    def test_index_is_queryable_and_rebuildable(self):
        self._write("run_p1", passed=True)
        self._write("run_f1", passed=False)
        self.assertEqual(len(self.store.query("passed = 0")), 1)
        self.store.db.execute("DELETE FROM runs")
        self.store.db.commit()
        self.assertEqual(self.store.query(), [])
        self.assertEqual(self.store.reindex(), 2)   # rebuilt from JSON on disk
        self.assertEqual(len(self.store.query()), 2)


if __name__ == "__main__":
    unittest.main()


class TestNoForbiddenChangesOutcomeFirst(GraderCase):
    """Regression: an independent agent was failed for READING a protected
    file (`python viz.py sample_trajectory.json out.html`) because the grader
    substring-matched shell commands. Mentioning a path is not modifying it."""

    def _ctx(self, protected_content="original\n", on_disk=None, steps=None):
        from agentwb.types import EnvironmentSpec, Trajectory
        task = Task(id="t", prompt="p",
                    environment=EnvironmentSpec(files={"protected.txt": protected_content}))
        # bytes, not write_text: text mode would translate newlines and
        # corrupt the CRLF fixture this class exists to test
        (self.ws / "protected.txt").write_bytes(
            (protected_content if on_disk is None else on_disk).encode("utf-8"))
        traj = Trajectory(trajectory_id="r", task_id="t")
        traj.steps = steps or []
        return GradingContext(task=task, trajectory=traj, workspace=self.ws)

    def _grade(self, ctx):
        return run_grader(GraderSpec(type="no_forbidden_changes",
                                     params={"paths": ["protected.txt"]}), ctx)

    def test_reading_a_protected_file_in_shell_is_not_a_violation(self):
        steps = [Step(step=1, action={"type": "tool_call", "tool": "shell",
                                      "arguments": {"command": "python viz.py protected.txt out.html"}})]
        self.assertIs(self._grade(self._ctx(steps=steps)).verdict, GraderVerdict.PASS)

    def test_an_actual_content_change_is_caught(self):
        result = self._grade(self._ctx(on_disk="tampered\n"))
        self.assertIs(result.verdict, GraderVerdict.FAIL)
        self.assertEqual(result.evidence[0]["violations"][0]["kind"], "content_changed")

    def test_deleting_the_protected_file_is_caught(self):
        ctx = self._ctx()
        (self.ws / "protected.txt").unlink()
        self.assertIs(self._grade(ctx).verdict, GraderVerdict.FAIL)

    def test_newline_differences_are_not_a_change(self):
        """Windows text-mode materialisation must not read as tampering."""
        ctx = self._ctx(protected_content="a\nb\n", on_disk="a\r\nb\r\n")
        self.assertIs(self._grade(ctx).verdict, GraderVerdict.PASS)

    def test_without_an_original_a_shell_write_attempt_is_still_caught(self):
        from agentwb.types import Trajectory
        task = Task(id="t", prompt="p")     # nothing seeded
        traj = Trajectory(trajectory_id="r", task_id="t")
        traj.steps = [Step(step=1, action={"type": "tool_call", "tool": "shell",
                                           "arguments": {"command": "echo x > secret.txt"}})]
        ctx = GradingContext(task=task, trajectory=traj, workspace=self.ws)
        result = run_grader(GraderSpec(type="no_forbidden_changes",
                                       params={"paths": ["secret.txt"]}), ctx)
        self.assertIs(result.verdict, GraderVerdict.FAIL)
