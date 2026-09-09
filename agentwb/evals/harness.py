"""Evaluation harness: task -> workspace -> agent -> trajectory -> graders -> experience.

Evaluation is part of the runtime, not an afterthought. Grading always runs
against the environment the agent actually left behind, and the agent never
gets a say in whether it passed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..aci.observations import RawStore
from ..analysis import failure_analyzer
from ..experience.store import ExperienceStore
from ..providers.base import ModelProvider
from ..runtime.agent import AgentRunner
from ..trajectories.recorder import TrajectoryRecorder
from ..trajectories.store import TrajectoryStore
from ..types import (
    Evaluation,
    FailureCategory,
    GraderVerdict,
    Task,
    TerminationReason,
    Trajectory,
    new_run_id,
)
from .graders import deterministic  # noqa: F401  -- registers the built-ins
from .graders.base import GradingContext, run_grader


@dataclass
class TrialResult:
    trajectory: Trajectory
    workspace: Path


@dataclass
class TaskResult:
    task_id: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for t in self.trials
                   if t.trajectory.evaluation and t.trajectory.evaluation.passed)

    @property
    def success_rate(self) -> float:
        return self.pass_count / len(self.trials) if self.trials else 0.0


class Harness:
    def __init__(
        self,
        store: TrajectoryStore,
        experience: ExperienceStore,
        workspaces_root: Path,
        keep_workspaces: bool = True,
        judge=None,
        analyze_failures: bool = True,
    ):
        self.store = store
        self.experience = experience
        self.workspaces_root = Path(workspaces_root)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self.keep_workspaces = keep_workspaces
        self.judge = judge
        self.analyze_failures = analyze_failures

    # -- environment ------------------------------------------------------
    def build_workspace(self, task: Task, run_id: str) -> Path:
        ws = self.workspaces_root / run_id
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)

        base = Path(task.source_path).parent if task.source_path else Path.cwd()
        if task.environment.template_dir:
            src = (base / task.environment.template_dir).resolve()
            if not src.is_dir():
                raise FileNotFoundError(f"template_dir not found: {src}")
            shutil.copytree(src, ws, dirs_exist_ok=True)
        for rel, content in task.environment.files.items():
            target = ws / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return ws

    # -- execution --------------------------------------------------------
    def run_task(
        self,
        task: Task,
        provider: ModelProvider,
        trials: Optional[int] = None,
        tool_timeout: int = 60,
    ) -> TaskResult:
        n = trials if trials is not None else task.trials
        result = TaskResult(task_id=task.id)
        for trial in range(max(1, n)):
            result.trials.append(self._run_one(task, provider, trial, tool_timeout))
        return result

    def _run_one(self, task: Task, provider: ModelProvider, trial: int, tool_timeout: int) -> TrialResult:
        run_id = new_run_id()
        ws = self.build_workspace(task, run_id)
        raw = RawStore(self.store.run_dir(run_id) / "raw")
        recorder = TrajectoryRecorder(self.store)
        runner = AgentRunner(provider, ws, recorder, raw, tool_timeout=tool_timeout)

        traj = runner.run(task, trial=trial, run_id=run_id)
        traj.environment_outcome = self._snapshot(ws)
        traj.evaluation = self.grade(task, traj, ws)
        traj.failure_categories = classify(task, traj)

        # Analyse failures while the evidence is fresh. The analysis is a
        # hypothesis and is stored beside the run, never folded into the verdict.
        if self.analyze_failures and not (traj.evaluation and traj.evaluation.passed):
            analysis = failure_analyzer.analyze(task, traj, judge=self.judge)
            failure_analyzer.save(analysis, self.store.run_dir(traj.trajectory_id))
            traj.metrics["analysis_source"] = analysis.source

        recorder.close(traj)
        self.experience.record(task, traj)
        return TrialResult(trajectory=traj, workspace=ws)

    # -- grading ----------------------------------------------------------
    def grade(self, task: Task, traj: Trajectory, workspace: Path) -> Evaluation:
        ctx = GradingContext(task=task, trajectory=traj, workspace=Path(workspace),
                             judge=self.judge)
        results = [run_grader(spec, ctx) for spec in task.graders]

        required = [r for r in results if r.required]
        unknowns = [r for r in results if r.verdict is GraderVerdict.UNKNOWN]

        if not task.graders:
            return Evaluation(passed=False, score=0.0, results=[], unknown_count=0,
                              notes="task defines no graders -- outcome is unverified, "
                                    "so it cannot be counted as a pass")

        # A required UNKNOWN blocks a pass. Insufficient evidence is not success.
        passed = bool(required) and all(r.verdict is GraderVerdict.PASS for r in required)

        weighted = [(r.weight, r.score) for r in results if r.score is not None]
        total_w = sum(w for w, _ in weighted)
        score = sum(w * s for w, s in weighted) / total_w if total_w else 0.0

        notes = ""
        if unknowns:
            names = ", ".join(r.grader for r in unknowns)
            notes = f"{len(unknowns)} grader(s) returned UNKNOWN: {names}"
        return Evaluation(passed=passed, score=round(score, 4), results=results,
                          unknown_count=len(unknowns), notes=notes)

    @staticmethod
    def _snapshot(ws: Path) -> dict[str, Any]:
        files = []
        for p in sorted(ws.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                files.append({"path": p.relative_to(ws).as_posix(), "bytes": p.stat().st_size})
            if len(files) >= 500:
                break
        return {"workspace": str(ws), "file_count": len(files), "files": files}


# --------------------------------------------------------------------------
# failure classification (deterministic; a hypothesis, not ground truth)
# --------------------------------------------------------------------------

def classify(task: Task, traj: Trajectory) -> list[str]:
    """Attach failure labels from observable trajectory facts only.

    Rule-based on purpose: these labels are cheap, reproducible and auditable.
    The V1 failure-analysis agent proposes richer root causes on top of them --
    and its output is explicitly a hypothesis, never a verdict.
    """
    if traj.evaluation and traj.evaluation.passed:
        return []

    labels: set[str] = set()
    m = traj.metrics or {}
    reason = traj.termination_reason

    if reason == TerminationReason.MAX_ITERATIONS.value:
        labels.add(FailureCategory.PREMATURE_TERMINATION.value)
    if reason == TerminationReason.NO_NEW_EVIDENCE.value:
        labels.add(FailureCategory.REPEATED_ACTION.value)
    if reason == TerminationReason.ENVIRONMENT_ERROR.value:
        labels.add(FailureCategory.ENVIRONMENT_FAILURE.value)
    if reason == TerminationReason.BLOCKED.value:
        labels.add(FailureCategory.BAD_TOOL_ARGUMENTS.value)

    if m.get("repeated_actions", 0) >= 2:
        labels.add(FailureCategory.REPEATED_ACTION.value)
    if m.get("tool_failures", 0) >= 2:
        labels.add(FailureCategory.BAD_TOOL_ARGUMENTS.value)
    if m.get("tool_calls", 0) == 0:
        # answered without touching the environment at all
        labels.add(FailureCategory.INCOMPLETE_VERIFICATION.value)

    ev = traj.evaluation
    if ev:
        if any(r.verdict is GraderVerdict.UNKNOWN and r.error for r in ev.results):
            labels.add(FailureCategory.GRADER_FAILURE.value)
        # agent stopped voluntarily but never ran the verification tool
        verified = any((s.action or {}).get("tool") == "run_tests" for s in traj.steps)
        if reason == TerminationReason.AGENT_FINISHED.value and not verified:
            labels.add(FailureCategory.INCOMPLETE_VERIFICATION.value)
        elif reason == TerminationReason.AGENT_FINISHED.value:
            labels.add(FailureCategory.INCORRECT_VERIFICATION.value)

    return sorted(labels)


# --------------------------------------------------------------------------
# task loading
# --------------------------------------------------------------------------

def load_task(path: Path) -> Task:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                f"{path.name} is YAML but PyYAML is not installed -- "
                "`pip install pyyaml`, or use the .json form of the task"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return Task.from_dict(data, source_path=str(path))


def load_tasks(path: Path) -> list[Task]:
    """Load one task file, or every task in a directory."""
    path = Path(path)
    if path.is_file():
        return [load_task(path)]
    tasks = []
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in {".json", ".yaml", ".yml"}:
            tasks.append(load_task(p))
    return tasks
