"""Failure analysis (spec section 18).

Input:  task, trajectory, outcome, grader failures
Output: root causes, the critical step, whether it was avoidable, a proposed
        fix, a recommended regression test.

**The output is a hypothesis, not ground truth.** Every record carries a
`source` field -- ``rules`` or ``model`` -- and nothing downstream is permitted
to treat either as a verdict. The deterministic signals are always computed;
the model, when one is configured, adds interpretation on top of them.

That ordering matters. A model handed a raw transcript will confabulate a
plausible story about why a run failed. A model handed *"the agent repeated an
identical action four times and never ran the tests"* is doing something much
closer to reading.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..judge.client import JudgeClient, JudgeError
from ..prompts import VersionedPrompt, register
from ..types import GraderVerdict, Task, TerminationReason, Trajectory, utcnow

ANALYZER_PROMPT = register(VersionedPrompt(
    name="failure_analyzer",
    version="v1",
    text="""\
You are analysing why an AI agent's run failed. You are producing a hypothesis
for an engineer to test -- not a verdict, and not a defence of the agent.

You are given deterministic facts already extracted from the run. Trust those
facts over your own reading of the transcript; they were computed, not inferred.

Aim for the *earliest* step where the run went wrong, not the step where the
failure became visible. Those are usually different, and the earlier one is the
one worth fixing.

Reply with a single JSON object and nothing else:

{
  "root_causes": ["<specific and mechanical, not 'the agent was confused'>"],
  "critical_step": <step number, or null if no single step>,
  "avoidable": <true|false>,
  "proposed_fix": "<a change to the prompt, tools, or task -- be concrete>",
  "recommended_regression_test": "<what a test asserting this stays fixed would check>",
  "optimizer_candidate": <true if a better prompt would plausibly fix this>,
  "categories": ["<from the taxonomy you were given>"],
  "confidence": <float 0.0-1.0>
}

If the evidence does not support a conclusion, say so in root_causes and set a
low confidence. Do not invent a cause to fill the field.""",
))


@dataclass
class FailureAnalysis:
    trajectory_id: str
    task_id: str
    source: str = "rules"                    # "rules" | "model"
    root_causes: list[str] = field(default_factory=list)
    critical_step: Optional[int] = None
    avoidable: bool = True
    proposed_fix: str = ""
    recommended_regression_test: str = ""
    optimizer_candidate: bool = False
    categories: list[str] = field(default_factory=list)
    confidence: float = 0.0
    deterministic_signals: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = ""
    model: str = ""
    error: Optional[str] = None
    analyzed_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def analyze(
    task: Task,
    traj: Trajectory,
    judge: Optional[JudgeClient] = None,
) -> FailureAnalysis:
    """Analyse one failed run. Safe to call on a passing run (returns empty)."""
    signals = extract_signals(task, traj)
    analysis = FailureAnalysis(
        trajectory_id=traj.trajectory_id,
        task_id=traj.task_id,
        deterministic_signals=signals,
        categories=list(traj.failure_categories),
        critical_step=signals.get("first_failed_step"),
    )
    if traj.evaluation and traj.evaluation.passed:
        analysis.root_causes = ["run passed -- nothing to analyse"]
        analysis.confidence = 1.0
        return analysis

    _apply_rules(analysis, signals)
    if judge is None:
        return analysis

    try:
        reply = judge.ask_json(
            ANALYZER_PROMPT.text,
            _build_user_prompt(task, traj, signals),
            prompt_id=ANALYZER_PROMPT.id,
        )
    except JudgeError as exc:
        analysis.error = f"{exc} (falling back to deterministic analysis)"
        return analysis

    return _merge_model_reply(analysis, reply)


# --------------------------------------------------------------------------
# deterministic layer
# --------------------------------------------------------------------------

def extract_signals(task: Task, traj: Trajectory) -> dict[str, Any]:
    """Facts about the run, computed rather than inferred."""
    tool_steps = [s for s in traj.steps if (s.action or {}).get("type") == "tool_call"]
    failed = [s for s in tool_steps if not (s.tool_result or {}).get("ok", True)]
    tools_used = [(s.action or {}).get("tool") for s in tool_steps]

    error_codes: dict[str, int] = {}
    for s in failed:
        code = (s.tool_result or {}).get("error") or "UNKNOWN_ERROR"
        error_codes[code] = error_codes.get(code, 0) + 1

    ev = traj.evaluation
    failing_graders = [
        {"grader": r.grader, "verdict": r.verdict.value, "required": r.required,
         "error": r.error,
         "evidence": r.evidence[:2]}
        for r in (ev.results if ev else [])
        if r.verdict is not GraderVerdict.PASS
    ]

    # Does this task involve tests at all? A research task that writes prose
    # should not be scolded for never running a test suite.
    grader_types = {g.type for g in task.graders}
    expects_tests = "tests_pass" in grader_types or any(
        g.type == "tool_used" and g.params.get("tool") == "run_tests" for g in task.graders
    )
    expects_file_changes = bool(grader_types & {"tests_pass", "file_exists", "file_contains"})

    return {
        "task_expects_tests": expects_tests,
        "task_expects_file_changes": expects_file_changes,
        "termination_reason": traj.termination_reason,
        "steps": len(traj.steps),
        "tool_calls": len(tool_steps),
        "tool_failures": len(failed),
        "first_failed_step": failed[0].step if failed else None,
        "error_codes": error_codes,
        "tools_used": tools_used,
        "distinct_tools": sorted(set(t for t in tools_used if t)),
        "repeated_actions": (traj.metrics or {}).get("repeated_actions", 0),
        "ran_tests": "run_tests" in tools_used,
        "modified_files": any(t in {"edit_file", "write_file"} for t in tools_used),
        "failing_graders": failing_graders,
        "unknown_graders": [r.grader for r in (ev.results if ev else [])
                            if r.verdict is GraderVerdict.UNKNOWN],
        "declared_success": bool((traj.final_output or {}).get("text")),
    }


def _apply_rules(analysis: FailureAnalysis, s: dict[str, Any]) -> None:
    """Cheap, auditable conclusions drawn only from computed signals."""
    causes: list[str] = []
    fixes: list[str] = []

    if s["unknown_graders"]:
        causes.append(
            f"grading was inconclusive: {', '.join(s['unknown_graders'])} returned UNKNOWN, "
            "so this run does not tell you whether the agent succeeded"
        )
        fixes.append("fix the eval before drawing conclusions about the agent")

    if s.get("task_expects_file_changes") and not s["modified_files"] and s["failing_graders"]:
        causes.append("the agent never modified any file, so the environment could not change")
        fixes.append("check whether the task prompt makes the required change explicit")

    if s.get("task_expects_tests") and s["modified_files"] and not s["ran_tests"]:
        causes.append("the agent changed files but never ran the tests, so it could not "
                      "have observed whether its change worked")
        fixes.append("strengthen the system prompt's verification instruction, or add a "
                     "required run_tests trajectory grader so this fails loudly")

    if s["repeated_actions"] >= 2:
        causes.append(f"the agent repeated an identical action {s['repeated_actions']} time(s), "
                      "gaining no new information")
        fixes.append("give the tool a clearer error, or add state to the observation so a "
                     "repeat is visibly pointless")

    for code, n in (s["error_codes"] or {}).items():
        if code == "NO_MATCH":
            causes.append(f"{n} edit(s) failed with NO_MATCH -- the agent's idea of the file "
                          "did not match its contents")
            fixes.append("require read_file_region before edit_file, or return the nearest "
                         "matching lines in the NO_MATCH error")
        elif code == "AMBIGUOUS_MATCH":
            causes.append(f"{n} edit(s) were ambiguous -- the anchor text was not unique")
            fixes.append("have edit_file suggest a uniquely-anchored alternative")
        elif code == "BAD_ARGUMENTS":
            causes.append(f"{n} tool call(s) had malformed arguments")
            fixes.append("tighten the tool's parameter descriptions")
        elif code == "GUARDRAIL":
            causes.append(f"{n} call(s) were blocked by a guardrail")
            fixes.append("if the attempt was legitimate, the workspace is too narrow; "
                         "if not, the prompt is steering the agent out of bounds")

    reason = s["termination_reason"]
    if reason == TerminationReason.MAX_ITERATIONS.value:
        causes.append("the agent ran out of steps before finishing")
        fixes.append("raise max_steps, or reduce how many steps the task needs")
    elif reason == TerminationReason.NO_NEW_EVIDENCE.value:
        causes.append("the run was stopped for looping without new evidence")
    elif reason == TerminationReason.ENVIRONMENT_ERROR.value:
        causes.append("the provider or environment failed -- this is not an agent failure")
        analysis.avoidable = False
    elif reason == TerminationReason.AGENT_FINISHED.value and s["declared_success"] \
            and s["failing_graders"]:
        causes.append("the agent reported completion while graders found the work incomplete: "
                      "it verified nothing, or misread what it saw")
        fixes.append("require the agent to quote the evidence for its claim")

    if not causes:
        causes.append("no deterministic signal explains this failure -- inspect the "
                      "transcript, and treat any model analysis as a starting hypothesis")

    analysis.root_causes = causes
    # numbered when there are several -- a run-on sentence of fixes is unreadable
    analysis.proposed_fix = (
        fixes[0] if len(fixes) == 1
        else "; ".join(f"({i}) {f}" for i, f in enumerate(fixes, 1))
    )
    analysis.optimizer_candidate = bool(
        s.get("task_expects_tests") and s["modified_files"] and not s["ran_tests"]
    ) or s["repeated_actions"] >= 2
    analysis.recommended_regression_test = _suggest_regression_test(s)
    # deterministic rules are reliable but shallow -- never claim high confidence
    analysis.confidence = 0.5 if len(causes) > 1 else 0.35


def _suggest_regression_test(s: dict[str, Any]) -> str:
    if s.get("task_expects_tests") and s["modified_files"] and not s["ran_tests"]:
        return ("a task with a required `tool_used: run_tests` grader, asserting the agent "
                "cannot pass by editing without verifying")
    if "NO_MATCH" in (s["error_codes"] or {}):
        return ("a task whose target file contains near-duplicate lines, asserting the agent "
                "reads before editing")
    if s["repeated_actions"] >= 2:
        return ("a task where the first tool call returns an empty result, asserting the agent "
                "changes approach instead of repeating")
    if s["unknown_graders"]:
        return ("a grader unit test pinning the UNKNOWN path, so an inconclusive eval can "
                "never silently read as a pass")
    return "a minimal reproduction of this task, kept in the regression suite"


# --------------------------------------------------------------------------
# model layer
# --------------------------------------------------------------------------

def _build_user_prompt(task: Task, traj: Trajectory, signals: dict[str, Any]) -> str:
    from .._taxonomy import taxonomy_names

    steps: list[str] = []
    for s in traj.steps:
        action = s.action or {}
        if action.get("type") == "tool_call":
            res = s.tool_result or {}
            status = "ok" if res.get("ok") else f"ERROR {res.get('error')}"
            obs = (s.observation or {}).get("summary", "")[:400]
            steps.append(f"[step {s.step}] {action.get('tool')} -> {status}\n{obs}")
        else:
            steps.append(f"[step {s.step}] FINAL ANSWER: {(s.rationale or '')[:400]}")

    return (
        f"TASK: {task.prompt}\n\n"
        f"SUCCESS CRITERIA: {task.success_criteria or '(none stated)'}\n\n"
        f"DETERMINISTIC FACTS (computed, trust these):\n"
        f"{json.dumps(signals, indent=2, default=str)}\n\n"
        f"FAILURE TAXONOMY (use these category names):\n{', '.join(taxonomy_names())}\n\n"
        f"TRANSCRIPT:\n" + "\n\n".join(steps)
    )


def _merge_model_reply(analysis: FailureAnalysis, reply) -> FailureAnalysis:
    d = reply.data
    rule_causes = analysis.root_causes

    model_causes = [str(c)[:400] for c in (d.get("root_causes") or []) if str(c).strip()]
    analysis.prompt_version = reply.prompt_id
    analysis.model = reply.model

    if not model_causes:
        # The judge replied but said nothing usable. Labelling this "model"
        # would credit an interpretation that was never made.
        analysis.error = "judge returned no root causes; deterministic analysis stands"
        return analysis

    analysis.source = "model"
    # keep both: the computed findings stay visible next to the interpretation
    analysis.root_causes = model_causes + [f"(deterministic) {c}" for c in rule_causes]

    step = d.get("critical_step")
    if isinstance(step, (int, float)):
        analysis.critical_step = int(step)
    if isinstance(d.get("avoidable"), bool):
        analysis.avoidable = d["avoidable"]
    if d.get("proposed_fix"):
        analysis.proposed_fix = str(d["proposed_fix"])[:1000]
    if d.get("recommended_regression_test"):
        analysis.recommended_regression_test = str(d["recommended_regression_test"])[:1000]
    if isinstance(d.get("optimizer_candidate"), bool):
        analysis.optimizer_candidate = d["optimizer_candidate"]

    from .._taxonomy import taxonomy_names
    valid = set(taxonomy_names())
    proposed = [str(c) for c in (d.get("categories") or []) if str(c) in valid]
    analysis.categories = sorted(set(analysis.categories) | set(proposed))

    try:
        analysis.confidence = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        analysis.confidence = 0.5
    return analysis


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save(analysis: FailureAnalysis, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "analysis.json"
    path.write_text(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


def load(run_dir: Path) -> Optional[FailureAnalysis]:
    path = Path(run_dir) / "analysis.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(FailureAnalysis)}
    return FailureAnalysis(**{k: v for k, v in d.items() if k in known})
