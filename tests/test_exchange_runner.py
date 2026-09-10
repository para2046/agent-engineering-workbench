"""Running a multi-agent exchange against a real environment.

The point of these tests is the rung the orchestrator cannot reach alone: a
disagreement settled by running an experiment in the workspace. Without an
executor bound to a real environment, resolution can only ever fall through to
HUMAN_REQUIRED, and the protocol's central claim -- that disputes end in
observation rather than argument -- is never exercised.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentwb.aci.observations import RawStore
from agentwb.aci.registry import ToolRegistry
from agentwb.experiments.runner import ExchangeRun, make_experiment_runner, run_exchange
from agentwb.judge.client import JudgeClient
from agentwb.judge.mock import FakeJudgeProvider
from agentwb.runtime.orchestrator import Orchestrator
from agentwb.types import Task


def client(*replies) -> JudgeClient:
    return JudgeClient(FakeJudgeProvider(list(replies)))


def msg(mtype, claim, evidence=None, confidence=0.4) -> dict:
    out = {"type": mtype, "claim": claim, "confidence": confidence}
    if evidence:
        out["evidence"] = evidence
    if mtype in ("HYPOTHESIS", "EXPERIMENT_PROPOSAL"):
        out["verification"] = {"metric": "m", "expected_if_correct": "a",
                               "expected_if_wrong": "b"}
    return out


def make_task(**over) -> Task:
    d = {
        "id": "exchange_task",
        "prompt": "Diagnose why marker.txt is missing.",
        "success_criteria": "findings.md explains it.",
        "environment": {"files": {"notes.md": "the marker was removed in the migration\n"}},
        "graders": [{"type": "file_exists", "params": {"path": "findings.md"}}],
    }
    d.update(over)
    return Task.from_dict(d, source_path=str(Path.cwd() / "inline.json"))


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def registry(self) -> ToolRegistry:
        ws = self.root / "ws"
        ws.mkdir(exist_ok=True)
        (ws / "notes.md").write_text("marker removed in migration\n", encoding="utf-8")
        return ToolRegistry(ws, RawStore(self.root / "raw"), timeout=30)


class TestExperimentExecutor(RunnerCase):
    def test_an_experiment_runs_in_the_workspace(self):
        run = make_experiment_runner(self.registry())
        ok, observation = run("python -c \"print('pool saturated')\"")
        self.assertTrue(ok)
        self.assertIn("pool saturated", observation)

    def test_an_empty_experiment_is_refused(self):
        ok, observation = make_experiment_runner(self.registry())("   ")
        self.assertFalse(ok)
        self.assertIn("no experiment", observation)

    def test_a_failing_command_returns_the_failure_as_an_observation(self):
        """'The experiment could not be run' is a result, not an exception."""
        ok, observation = make_experiment_runner(self.registry())("exit 3")
        self.assertFalse(ok)
        self.assertTrue(observation)

    def test_guardrails_still_apply_to_experiments(self):
        """Two agents wanting to check something is not a reason to widen the sandbox."""
        ok, observation = make_experiment_runner(self.registry())("sudo rm -rf /")
        self.assertFalse(ok)
        self.assertIn("GUARDRAIL", observation)

    def test_experiments_cannot_escape_the_workspace(self):
        reg = self.registry()
        ok, observation = make_experiment_runner(reg)("python -c \"import os; print(os.getcwd())\"")
        self.assertTrue(ok)
        self.assertIn(str(reg.workspace.name), observation)


class TestRunExchange(RunnerCase):
    def _factory(self, researcher_replies, engineer_replies, judge=None, max_rounds=2):
        agents = {"researcher": client(*researcher_replies),
                  "engineer": client(*engineer_replies)}

        def factory(run_experiment):
            return Orchestrator(agents, max_rounds=max_rounds, judge=judge,
                                run_experiment=run_experiment)
        return factory

    def test_the_workspace_is_built_from_the_task(self):
        run = run_exchange(
            task=make_task(),
            orchestrator_factory=self._factory(
                [msg("QUESTION", "what changed?")],
                [msg("BLOCKER", "no access")],
            ),
            workspaces_root=self.root / "exchanges",
        )
        self.assertTrue((run.workspace / "notes.md").is_file())
        self.assertIsInstance(run, ExchangeRun)

    def test_each_run_gets_a_fresh_workspace(self):
        task = make_task()
        a = run_exchange(task, self._factory([msg("QUESTION", "a")], [msg("BLOCKER", "b")]),
                         self.root / "exchanges")
        b = run_exchange(task, self._factory([msg("QUESTION", "c")], [msg("BLOCKER", "d")]),
                         self.root / "exchanges")
        self.assertNotEqual(a.workspace, b.workspace)

    def test_the_environment_is_graded_not_the_transcript(self):
        """Agents claiming success does not create findings.md."""
        from agentwb.evals.harness import Harness
        from agentwb.experience.store import ExperienceStore
        from agentwb.trajectories.store import TrajectoryStore

        store = TrajectoryStore(self.root / "traj")
        harness = Harness(store, ExperienceStore(self.root / "exp"), self.root / "ws2")
        run = run_exchange(
            task=make_task(),
            orchestrator_factory=self._factory(
                [msg("QUESTION", "what changed?")],
                [msg("FINAL_REPORT", "I have written the findings",
                     evidence=[{"source": "me", "observation": "trust me"}])],
            ),
            workspaces_root=self.root / "exchanges",
            grade=harness.grade,
        )
        self.assertIsNotNone(run.evaluation)
        self.assertFalse(run.passed, "no file was written, so it cannot pass")
        store.close()

    def test_a_run_that_actually_writes_the_file_passes(self):
        from agentwb.evals.harness import Harness
        from agentwb.experience.store import ExperienceStore
        from agentwb.trajectories.store import TrajectoryStore

        store = TrajectoryStore(self.root / "traj2")
        harness = Harness(store, ExperienceStore(self.root / "exp2"), self.root / "ws3")

        task = make_task()

        def factory(run_experiment):
            # the engineer's "experiment" writes the artifact the grader wants
            run_experiment("python -c \"open('findings.md','w').write('the marker "
                           "was removed during the migration')\"")
            agents = {"researcher": client(msg("QUESTION", "what changed?")),
                      "engineer": client(msg("FINAL_REPORT", "wrote findings.md",
                                             evidence=[{"source": "shell",
                                                        "observation": "file written"}]))}
            return Orchestrator(agents, max_rounds=1, run_experiment=run_experiment)

        run = run_exchange(task, factory, self.root / "exchanges", grade=harness.grade)
        self.assertTrue((run.workspace / "findings.md").is_file())
        self.assertTrue(run.passed)
        store.close()

    def test_ungraded_when_the_task_defines_no_graders(self):
        run = run_exchange(
            task=make_task(graders=[]),
            orchestrator_factory=self._factory([msg("QUESTION", "a")], [msg("BLOCKER", "b")]),
            workspaces_root=self.root / "exchanges",
        )
        self.assertIsNone(run.evaluation)
        self.assertFalse(run.passed)

    def test_disagreement_is_settled_by_running_the_experiment(self):
        """The rung the orchestrator cannot reach without an environment."""
        judge = client(
            {"discriminative": True,
             "experiment": "python -c \"print('index present')\"",
             "metric": "index listing", "expected_if_a": "index present",
             "expected_if_b": "index missing",
             "reasoning": "the two predict different output"},
            {"supports": "A", "reasoning": "the observation says index present"},
        )
        run = run_exchange(
            task=make_task(),
            orchestrator_factory=self._factory(
                [msg("HYPOTHESIS", "the connection pool is exhausted by per-item lookups",
                     evidence=[{"source": "metrics", "observation": "pool 100/100"}])],
                [msg("HYPOTHESIS", "a database index was dropped during the migration")],
                judge=judge, max_rounds=1,
            ),
            workspaces_root=self.root / "exchanges",
        )
        self.assertTrue(run.exchange.disagreements)
        record = run.exchange.disagreements[0]
        resolution = record.get("resolution") or record.get("outcome", {}).get("resolution")
        self.assertTrue(str(resolution).startswith("RESOLVED"),
                        f"expected a resolution, got {resolution}")

    def test_result_serialises(self):
        run = run_exchange(
            task=make_task(),
            orchestrator_factory=self._factory([msg("QUESTION", "a")], [msg("BLOCKER", "b")]),
            workspaces_root=self.root / "exchanges",
        )
        payload = json.loads(json.dumps(run.to_dict(), default=str))
        self.assertEqual(payload["task_id"], "exchange_task")
        self.assertIn("exchange", payload)


if __name__ == "__main__":
    unittest.main()


class TestRoleFixtureProvider(unittest.TestCase):
    """The offline fixture that makes the multi-agent rung demonstrable."""

    def _provider(self):
        from agentwb.judge.mock import RoleFixtureProvider
        return RoleFixtureProvider()

    def test_it_emits_parseable_protocol_artifacts(self):
        from agentwb.protocols.schemas import AgentMessage, validate

        p = self._provider()
        payload = json.loads(p.generate("You are the researcher agent.", [], []).text)
        payload.update(sender="researcher", recipient="engineer")
        self.assertEqual(validate(AgentMessage.from_dict(payload)), [])

    def test_each_seat_gets_its_own_script(self):
        r = json.loads(self._provider().generate("You are the researcher agent.", [], []).text)
        e = json.loads(self._provider().generate("You are the engineer agent.", [], []).text)
        self.assertEqual(r["type"], "HYPOTHESIS")
        self.assertEqual(e["type"], "IMPLEMENTATION_PLAN")

    def test_an_exhausted_script_blocks_rather_than_repeating(self):
        """Restating an earlier position would register as duplicate work."""
        p = self._provider()
        for _ in range(4):
            last = json.loads(p.generate("You are the researcher agent.", [], []).text)
        self.assertEqual(last["type"], "BLOCKER")

    def test_reset_rewinds_between_runs(self):
        p = self._provider()
        first = json.loads(p.generate("You are the researcher agent.", [], []).text)
        p.generate("You are the researcher agent.", [], [])
        p.reset()
        self.assertEqual(json.loads(
            p.generate("You are the researcher agent.", [], []).text)["type"], first["type"])

    def test_it_is_reachable_as_a_provider_key(self):
        from agentwb.providers.base import available_providers, build_provider
        self.assertIn("mock-role", available_providers())
        self.assertEqual(build_provider("mock-role").name, "mock-role")

    def test_a_full_exchange_runs_clean_but_is_still_graded_on_the_environment(self):
        """The engineer reports 'findings.md written'; no file exists. The
        environment wins -- the same rule as for a single agent."""
        from agentwb.evals.harness import Harness
        from agentwb.experience.store import ExperienceStore
        from agentwb.judge.mock import RoleFixtureProvider
        from agentwb.trajectories.store import TrajectoryStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TrajectoryStore(root / "traj")
            harness = Harness(store, ExperienceStore(root / "exp"), root / "ws")

            def factory(run_experiment):
                agents = {"researcher": JudgeClient(RoleFixtureProvider()),
                          "engineer": JudgeClient(RoleFixtureProvider())}
                return Orchestrator(agents, max_rounds=3, run_experiment=run_experiment)

            run = run_exchange(make_task(), factory, root / "exchanges", grade=harness.grade)
            ex = run.exchange
            metrics = ex.metrics if isinstance(ex.metrics, dict) else ex.metrics.to_dict()

            self.assertEqual(metrics["protocol_violations"], 0)
            self.assertGreaterEqual(metrics["handoffs"], 4)
            self.assertIsNotNone(ex.final_report)
            self.assertFalse(run.passed, "a claimed file that was never written cannot pass")
            store.close()
