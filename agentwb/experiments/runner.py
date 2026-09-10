"""Running a multi-agent exchange against a real environment (spec sections 13-16, 21).

The orchestrator can route messages and detect disagreement on its own, but the
disagreement ladder has a rung it cannot reach without help: *run the
discriminating experiment and let the environment decide*. Without something
executing that experiment, resolution can only ever fall through to
HUMAN_REQUIRED, and the protocol's whole point -- that a dispute is settled by
observation rather than by argument -- never actually fires.

This module supplies that missing rung. It builds the task's workspace, binds a
bounded executor to it, and hands that executor to the orchestrator. The
experiment runs where the work happens, against the same files the agents are
arguing about.

Two things it deliberately does not do:

**It does not grade.** The exchange produces a transcript and an environment;
the task's graders inspect the environment afterwards, exactly as for a single
agent. No agent, and no orchestrator, declares the work successful.

**It does not widen the sandbox.** Experiments run through the same ToolRegistry
guardrails as agent tool calls -- workspace-confined, denylisted commands
refused, secrets stripped. "The agents needed to check something" is not a
reason to hand them a wider shell than the task itself gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..aci.observations import RawStore
from ..aci.registry import ToolRegistry
from ..types import Evaluation, Task, Trajectory, new_run_id, utcnow


@dataclass
class ExchangeRun:
    """One multi-agent run: the exchange, the environment it left, and its grade."""

    run_id: str
    task_id: str
    workspace: Path
    exchange: Any = None
    evaluation: Optional[Evaluation] = None
    started_at: str = field(default_factory=utcnow)
    ended_at: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.evaluation and self.evaluation.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "workspace": str(self.workspace),
            "exchange": self.exchange.to_dict() if self.exchange else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


def make_experiment_runner(
    registry: ToolRegistry,
    timeout_note: str = "",
) -> Callable[[str], tuple[bool, str]]:
    """Bind an experiment executor to a workspace.

    Returns ``run(action) -> (ok, observation)``.

    An experiment is a free-text action proposed by the designer, so it is
    routed through ``shell`` -- but through the *registry's* shell, which keeps
    the guardrails. A refused command comes back as a failed observation rather
    than an exception, because "the experiment could not be run" is itself a
    result the disagreement protocol knows how to handle.
    """

    def run(action: str) -> tuple[bool, str]:
        action = (action or "").strip()
        if not action:
            return False, "no experiment was specified"
        result = registry.call("shell", {"command": action})
        observation = result.summary or ""
        if result.error:
            observation = f"[{result.error}] {observation}".strip()
        if timeout_note and not result.ok:
            observation = f"{observation}\n{timeout_note}"
        return result.ok, observation

    return run


def run_exchange(
    task: Task,
    orchestrator_factory: Callable[[Callable[[str], tuple[bool, str]]], Any],
    workspaces_root: Path,
    store_root: Optional[Path] = None,
    grade: Optional[Callable[[Task, Trajectory, Path], Evaluation]] = None,
    run_id: Optional[str] = None,
    tool_timeout: int = 60,
) -> ExchangeRun:
    """Build the environment, run the exchange in it, then grade what it left.

    ``orchestrator_factory(run_experiment)`` receives the bound executor and
    returns a configured Orchestrator. Passing a factory rather than an
    orchestrator keeps the wiring order honest: the executor cannot exist until
    the workspace does.
    """
    run_id = run_id or new_run_id()
    workspace = _build_workspace(task, workspaces_root / run_id)
    raw = RawStore((store_root or workspaces_root) / run_id / "raw")
    registry = ToolRegistry(workspace, raw, timeout=tool_timeout)

    result = ExchangeRun(run_id=run_id, task_id=task.id, workspace=workspace)
    orchestrator = orchestrator_factory(make_experiment_runner(registry))
    result.exchange = orchestrator.run(task.id, task.prompt, task.success_criteria)

    if grade is not None:
        # Graded on the environment, through a trajectory carrying the
        # exchange's final report as its claim -- the same path a single-agent
        # run takes, so a multi-agent pass means the same thing.
        traj = _as_trajectory(result, task)
        result.evaluation = grade(task, traj, workspace)

    result.ended_at = utcnow()
    return result


def _build_workspace(task: Task, target: Path) -> Path:
    import shutil

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    base = Path(task.source_path).parent if task.source_path else Path.cwd()
    if task.environment.template_dir:
        src = (base / task.environment.template_dir).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"template_dir not found: {src}")
        shutil.copytree(src, target, dirs_exist_ok=True)
    for rel, content in task.environment.files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return target


def _as_trajectory(run: ExchangeRun, task: Task) -> Trajectory:
    """Adapt an exchange into the shape graders already understand.

    Multi-agent messages become steps so that trajectory graders -- tool_used,
    no_forbidden_changes -- see the same structure they see for one agent.
    """
    from ..types import Step

    exchange = run.exchange
    traj = Trajectory(
        trajectory_id=run.run_id,
        task_id=task.id,
        provider="multi-agent",
        model=",".join(sorted(getattr(exchange, "agents", {}) or {})) or "multi-agent",
        prompt_version="multi_agent_role:v1",
        workspace=str(run.workspace),
        started_at=run.started_at,
    )
    messages = list(getattr(exchange, "messages", []) or [])
    for i, m in enumerate(messages, 1):
        traj.steps.append(Step(
            step=i,
            action={"type": "message", "sender": m.sender, "message_type": m.type.value},
            observation={"summary": m.claim[:500]},
            rationale=m.claim[:2000],
            started_at=getattr(m, "at", ""),
        ))

    final = getattr(exchange, "final_report", None)
    traj.final_output = {"text": final.claim if final else ""}
    traj.termination_reason = getattr(exchange, "termination_reason", "")
    traj.failure_categories = list(getattr(exchange, "failure_categories", []) or [])

    metrics = getattr(exchange, "metrics", {}) or {}
    traj.metrics = dict(metrics) if isinstance(metrics, dict) else metrics.to_dict()
    return traj
