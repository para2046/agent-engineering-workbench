"""Core data types for the Agent Engineering Workbench.

Everything here is provider-agnostic. No Claude/OpenAI assumptions leak into
these structures: a provider adapter is responsible for translating to and from
its own wire format.

Design notes
------------
* Trajectories are append-only. A recorder writes steps as they happen; nothing
  mutates a step after it is recorded.
* Grader results always carry structured evidence and may return UNKNOWN when
  the evidence is insufficient (spec sections 7 and 8).
* Only observable reasoning artifacts are stored -- a short, explicitly-emitted
  ``rationale`` field. Hidden chain-of-thought is never captured.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------

def new_run_id() -> str:
    """Reproducible-ish, sortable run id: run_<utc-compact>_<short-uuid>."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------

class TerminationReason(str, enum.Enum):
    """Why an agent loop stopped. Every run must end with exactly one of these."""

    SUCCESS = "SUCCESS"
    FAILED_EVAL = "FAILED_EVAL"
    NO_NEW_EVIDENCE = "NO_NEW_EVIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    BLOCKED = "BLOCKED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"
    AGENT_FINISHED = "AGENT_FINISHED"


class GraderVerdict(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class FailureCategory(str, enum.Enum):
    """Extensible failure taxonomy (spec section 10).

    A trajectory may carry several of these -- do not force one label.
    """

    TASK_UNDERSTANDING = "TASK_UNDERSTANDING"
    BAD_HYPOTHESIS = "BAD_HYPOTHESIS"
    BAD_TOOL_SELECTION = "BAD_TOOL_SELECTION"
    BAD_TOOL_ARGUMENTS = "BAD_TOOL_ARGUMENTS"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    IGNORED_EVIDENCE = "IGNORED_EVIDENCE"
    STATE_LOSS = "STATE_LOSS"
    REPEATED_ACTION = "REPEATED_ACTION"
    PREMATURE_TERMINATION = "PREMATURE_TERMINATION"
    INCORRECT_VERIFICATION = "INCORRECT_VERIFICATION"
    INCOMPLETE_VERIFICATION = "INCOMPLETE_VERIFICATION"
    AGENT_DISAGREEMENT = "AGENT_DISAGREEMENT"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    GRADER_FAILURE = "GRADER_FAILURE"
    # multi-agent (reserved for V4; defined now so stored data stays stable)
    ROLE_VIOLATION = "ROLE_VIOLATION"
    INFORMATION_WITHHOLDING = "INFORMATION_WITHHOLDING"
    IGNORED_AGENT_INPUT = "IGNORED_AGENT_INPUT"
    TASK_DERAILMENT = "TASK_DERAILMENT"
    CONVERSATION_RESET = "CONVERSATION_RESET"
    REASONING_ACTION_MISMATCH = "REASONING_ACTION_MISMATCH"


# --------------------------------------------------------------------------
# tools / ACI
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSchema:
    """Provider-neutral description of one tool.

    Kept deliberately small: name, one-line purpose, JSON-schema parameters.
    Adapters translate this into whatever their provider expects.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Structured tool output.

    ``summary`` is what the agent sees -- bounded and informative.
    ``data`` holds structured fields for graders and analysis.
    ``raw_ref`` points at full output on disk when the observation was truncated,
    so the underlying data is never lost (spec section 4).
    """

    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    raw_ref: Optional[str] = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# provider interface types
# --------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: Optional[str] = None


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GraderSpec:
    """Declarative grader configuration attached to a task."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    weight: float = 1.0
    name: Optional[str] = None

    @property
    def label(self) -> str:
        return self.name or self.type


@dataclass(frozen=True)
class EnvironmentSpec:
    """How to construct the workspace the agent acts in.

    ``files`` seeds literal file contents; ``template_dir`` copies a directory.
    Both are relative to the task file's own directory.
    """

    files: dict[str, str] = field(default_factory=dict)
    template_dir: Optional[str] = None
    setup_commands: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    success_criteria: str = ""
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    graders: list[GraderSpec] = field(default_factory=list)
    tools: Optional[list[str]] = None       # None = all registered tools
    max_steps: int = 12
    trials: int = 1
    tags: list[str] = field(default_factory=list)
    source_path: Optional[str] = None

    @staticmethod
    def from_dict(d: dict[str, Any], source_path: Optional[str] = None) -> "Task":
        env_raw = d.get("environment") or {}
        env = EnvironmentSpec(
            files=dict(env_raw.get("files") or {}),
            template_dir=env_raw.get("template_dir"),
            setup_commands=list(env_raw.get("setup_commands") or []),
        )
        graders = [
            GraderSpec(
                type=g["type"],
                params=dict(g.get("params") or {}),
                required=bool(g.get("required", True)),
                weight=float(g.get("weight", 1.0)),
                name=g.get("name"),
            )
            for g in (d.get("graders") or [])
        ]
        missing = [k for k in ("id", "prompt") if not d.get(k)]
        if missing:
            raise ValueError(f"task is missing required field(s): {', '.join(missing)}")
        return Task(
            id=d["id"],
            prompt=d["prompt"],
            success_criteria=d.get("success_criteria", ""),
            environment=env,
            graders=graders,
            tools=d.get("tools"),
            max_steps=int(d.get("max_steps", 12)),
            trials=int(d.get("trials", 1)),
            tags=list(d.get("tags") or []),
            source_path=source_path,
        )


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------

@dataclass
class Step:
    """One state -> decision -> action -> observation transition."""

    step: int
    state: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""          # only what the agent explicitly emitted for logging
    latency_ms: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class GraderResult:
    grader: str
    verdict: GraderVerdict
    score: Optional[float] = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: float = 0.0
    required: bool = True
    weight: float = 1.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.verdict is GraderVerdict.PASS

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class Evaluation:
    """Aggregate of all grader results for one trajectory."""

    passed: bool = False
    score: float = 0.0
    results: list[GraderResult] = field(default_factory=list)
    unknown_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "unknown_count": self.unknown_count,
            "notes": self.notes,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class Trajectory:
    """The machine-readable record of one run. Append-only during execution."""

    trajectory_id: str
    task_id: str
    agent_config: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    prompt_version: str = ""
    started_at: str = field(default_factory=utcnow)
    ended_at: str = ""
    workspace: str = ""
    trial: int = 0

    steps: list[Step] = field(default_factory=list)
    final_output: dict[str, Any] = field(default_factory=dict)
    environment_outcome: dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[Evaluation] = None
    termination_reason: str = ""
    failure_categories: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "agent_config": self.agent_config,
            "model": self.model,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "workspace": self.workspace,
            "trial": self.trial,
            "steps": [s.to_dict() for s in self.steps],
            "final_output": self.final_output,
            "environment_outcome": self.environment_outcome,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "termination_reason": self.termination_reason,
            "failure_categories": self.failure_categories,
            "metrics": self.metrics,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Trajectory":
        t = Trajectory(
            trajectory_id=d["trajectory_id"],
            task_id=d.get("task_id", ""),
            agent_config=d.get("agent_config") or {},
            model=d.get("model", ""),
            provider=d.get("provider", ""),
            prompt_version=d.get("prompt_version", ""),
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at", ""),
            workspace=d.get("workspace", ""),
            trial=int(d.get("trial", 0)),
            termination_reason=d.get("termination_reason", ""),
            failure_categories=list(d.get("failure_categories") or []),
            metrics=d.get("metrics") or {},
        )
        t.steps = [Step(**s) for s in (d.get("steps") or [])]
        t.final_output = d.get("final_output") or {}
        t.environment_outcome = d.get("environment_outcome") or {}
        ev = d.get("evaluation")
        if ev:
            t.evaluation = Evaluation(
                passed=ev.get("passed", False),
                score=ev.get("score", 0.0),
                unknown_count=ev.get("unknown_count", 0),
                notes=ev.get("notes", ""),
                results=[
                    GraderResult(
                        grader=r["grader"],
                        verdict=GraderVerdict(r["verdict"]),
                        score=r.get("score"),
                        evidence=r.get("evidence") or [],
                        uncertainty=r.get("uncertainty", 0.0),
                        required=r.get("required", True),
                        weight=r.get("weight", 1.0),
                        error=r.get("error"),
                    )
                    for r in (ev.get("results") or [])
                ],
            )
        return t

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
