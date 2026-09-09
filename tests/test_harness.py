"""End-to-end harness tests driven by a scripted provider.

Deterministic: no network, no model. Each test pins one behaviour of the
runtime -- what happens when the agent finishes early, loops, passes bad
arguments, or actually solves the task.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb.evals.harness import Harness, classify, load_task
from agentwb.experience.store import ExperienceStore
from agentwb.providers.mock import ScriptedProvider
from agentwb.trajectories.store import TrajectoryStore
from agentwb.types import (
    FailureCategory,
    GraderVerdict,
    ModelResponse,
    Task,
    TerminationReason,
    ToolCall,
)

BUGGY = "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\n"
TESTS = (
    "import unittest\n"
    "from calculator import divide\n"
    "class T(unittest.TestCase):\n"
    "    def test_raises(self):\n"
    "        with self.assertRaises(ValueError):\n"
    "            divide(1, 0)\n"
)


def make_task(**over) -> Task:
    d = {
        "id": "t_fix",
        "prompt": "Fix calculator.py so the tests pass.",
        "environment": {"files": {"calculator.py": BUGGY, "test_calculator.py": TESTS}},
        "max_steps": 8,
        "graders": [
            {"type": "tests_pass", "name": "suite", "required": True},
            {"type": "tool_used", "name": "verified", "required": True,
             "params": {"tool": "run_tests"}},
        ],
    }
    d.update(over)
    return Task.from_dict(d, source_path=str(Path.cwd() / "inline.json"))


def call(name, **args):
    return ModelResponse(tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args)])


def finish(text="done"):
    return ModelResponse(text=text)


class HarnessCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.store = TrajectoryStore(root)
        self.experience = ExperienceStore(root / "experience")
        self.harness = Harness(self.store, self.experience, root / "workspaces")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def run_script(self, script, task=None):
        task = task or make_task()
        provider = ScriptedProvider(script)
        result = self.harness.run_task(task, provider, trials=1)
        return result.trials[0].trajectory


class TestSuccessPath(HarnessCase):
    def test_agent_that_fixes_and_verifies_passes(self):
        traj = self.run_script([
            call("read_file_region", path="calculator.py", start=1, end=10),
            call("edit_file", path="calculator.py", old="return 0",
                 new='raise ValueError("division by zero")'),
            call("run_tests"),
            finish("Replaced the silent zero with a ValueError; suite is green."),
        ])
        self.assertTrue(traj.evaluation.passed)
        self.assertEqual(traj.termination_reason, TerminationReason.AGENT_FINISHED.value)
        self.assertEqual(traj.failure_categories, [])

    def test_success_is_recorded_as_experience(self):
        self.run_script([
            call("edit_file", path="calculator.py", old="return 0",
                 new='raise ValueError("x")'),
            call("run_tests"),
            finish(),
        ])
        entries = list(self.experience.iter_experiences())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["outcome"], "PASS")
        self.assertTrue(entries[0]["actions"])


class TestGradingIsIndependentOfTheAgent(HarnessCase):
    def test_agent_claiming_success_without_fixing_still_fails(self):
        """The whole point: self-report is not evidence."""
        traj = self.run_script([
            call("run_tests"),
            finish("All tests pass and the bug is fixed."),
        ])
        self.assertFalse(traj.evaluation.passed)
        suite = next(r for r in traj.evaluation.results if r.grader == "suite")
        self.assertIs(suite.verdict, GraderVerdict.FAIL)

    def test_fix_without_running_tests_fails_the_process_grader(self):
        traj = self.run_script([
            call("edit_file", path="calculator.py", old="return 0",
                 new='raise ValueError("x")'),
            finish("fixed"),
        ])
        verified = next(r for r in traj.evaluation.results if r.grader == "verified")
        self.assertIs(verified.verdict, GraderVerdict.FAIL)
        self.assertFalse(traj.evaluation.passed)
        self.assertIn(FailureCategory.INCOMPLETE_VERIFICATION.value, traj.failure_categories)

    def test_task_without_graders_cannot_pass(self):
        task = make_task(graders=[])
        traj = self.run_script([finish("trust me")], task=task)
        self.assertFalse(traj.evaluation.passed)
        self.assertIn("no graders", traj.evaluation.notes)

    def test_unknown_grader_type_yields_unknown_not_pass(self):
        task = make_task(graders=[{"type": "does_not_exist", "required": True}])
        traj = self.run_script([finish()], task=task)
        result = traj.evaluation.results[0]
        self.assertIs(result.verdict, GraderVerdict.UNKNOWN)
        self.assertFalse(traj.evaluation.passed)
        self.assertEqual(traj.evaluation.unknown_count, 1)
        self.assertIn(FailureCategory.GRADER_FAILURE.value, traj.failure_categories)


class TestTermination(HarnessCase):
    def test_max_iterations_stops_the_loop(self):
        task = make_task(max_steps=3)
        traj = self.run_script([call("list_files", path=".")] * 10, task=task)
        self.assertEqual(traj.termination_reason, TerminationReason.MAX_ITERATIONS.value)
        self.assertLessEqual(len(traj.steps), 3)

    def test_identical_repeated_action_stops_as_no_new_evidence(self):
        task = make_task(max_steps=20)
        traj = self.run_script([call("list_files", path=".")] * 12, task=task)
        self.assertEqual(traj.termination_reason, TerminationReason.NO_NEW_EVIDENCE.value)
        self.assertIn(FailureCategory.REPEATED_ACTION.value, traj.failure_categories)

    def test_persistent_tool_errors_block_the_run(self):
        task = make_task(max_steps=20)
        script = [call("read_file_region", path=f"missing_{i}.py") for i in range(8)]
        traj = self.run_script(script, task=task)
        self.assertEqual(traj.termination_reason, TerminationReason.BLOCKED.value)


class TestEnvironmentIsolation(HarnessCase):
    def test_each_trial_gets_a_fresh_workspace(self):
        task = make_task(trials=2)
        provider = ScriptedProvider([
            call("write_file", path="scratch.txt", content="a"),
            finish(),
        ])
        # ScriptedProvider exhausts its script; each trial builds its own workspace
        result = self.harness.run_task(task, provider, trials=2)
        ws = [t.workspace for t in result.trials]
        self.assertNotEqual(ws[0], ws[1])
        for w in ws:
            self.assertTrue((w / "calculator.py").is_file())

    def test_workspace_starts_from_the_task_definition_not_previous_runs(self):
        self.run_script([
            call("edit_file", path="calculator.py", old="return 0", new="return -1"),
            finish(),
        ])
        traj2 = self.run_script([call("read_file_region", path="calculator.py"), finish()])
        text = (Path(traj2.workspace) / "calculator.py").read_text(encoding="utf-8")
        self.assertIn("return 0", text)
        self.assertNotIn("return -1", text)


class TestProviderStateDoesNotLeakBetweenRuns(HarnessCase):
    """Regression: `run tasks/` reused one provider instance across tasks, so a
    stateful fixture carried its progress into the next task and the second run
    terminated after a single step. Providers are now reset per trajectory."""

    def test_scripted_provider_is_rewound_for_each_run(self):
        script = [
            call("edit_file", path="calculator.py", old="return 0",
                 new='raise ValueError("division by zero")'),
            call("run_tests"),
            finish("fixed and verified"),
        ]
        provider = ScriptedProvider(script)
        task = make_task()

        first = self.harness.run_task(task, provider, trials=1).trials[0].trajectory
        second = self.harness.run_task(task, provider, trials=1).trials[0].trajectory

        self.assertEqual(len(first.steps), len(second.steps))
        self.assertTrue(first.evaluation.passed)
        self.assertTrue(second.evaluation.passed)

    def test_rule_provider_replays_its_full_sequence_on_a_second_task(self):
        from agentwb.providers.mock import RuleProvider

        provider = RuleProvider()
        guided = make_task(
            id="guided",
            prompt=("Fix calculator.py: replace `return 0` with "
                    "`raise ValueError(\"division by zero\")`. Then run the tests."),
        )
        first = self.harness.run_task(guided, provider, trials=1).trials[0].trajectory
        second = self.harness.run_task(guided, provider, trials=1).trials[0].trajectory

        self.assertTrue(first.evaluation.passed)
        self.assertTrue(second.evaluation.passed, "second run must not inherit stage state")
        self.assertGreater(len(second.steps), 1)


class TestClassification(unittest.TestCase):
    def test_passing_run_gets_no_labels(self):
        from agentwb.types import Evaluation, Trajectory
        t = Trajectory(trajectory_id="r1", task_id="t")
        t.evaluation = Evaluation(passed=True, score=1.0)
        self.assertEqual(classify(make_task(), t), [])


if __name__ == "__main__":
    unittest.main()
