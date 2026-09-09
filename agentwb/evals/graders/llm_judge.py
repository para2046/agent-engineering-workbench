"""Model-based graders (spec section 7, priority 3).

Only reached for things code cannot check: was a synthesis grounded in the
evidence, did the answer cover the question, was the explanation clear. If a
deterministic grader can answer it, use that instead -- these cost money, take
time, and are themselves fallible.

Three design rules, all from the spec:

* **Isolated dimensions, not one universal judge.** There is no
  ``score_this_0_to_100`` grader here. Each dimension has its own rubric, its
  own prompt version, and its own registered grader.
* **Structured evidence, always.** A judge that returns a number without
  quoting what it looked at cannot be audited, and an unauditable score is an
  opinion wearing a lab coat.
* **UNKNOWN is allowed and expected.** No judge configured, unparseable reply,
  or no evidence to point at all yield UNKNOWN -- never a defaulted PASS.
"""

from __future__ import annotations

from typing import Any, Optional

from ...judge.client import JudgeError
from ...prompts import VersionedPrompt, register
from ...types import GraderResult, GraderVerdict
from .base import GradingContext, bad, grader, ok, unknown

DEFAULT_THRESHOLD = 0.7

JUDGE_SYSTEM = register(VersionedPrompt(
    name="judge_system",
    version="v1",
    text="""\
You are grading one narrow dimension of an AI agent's work. You are not deciding
whether the overall task succeeded -- other graders handle that.

Rules:
- Judge only the dimension described. Ignore everything else, including whether
  you would have solved the problem differently.
- Ground every claim in the material you were given. Quote it.
- If the material does not let you judge this dimension, say so: return
  "verdict": "UNKNOWN". An honest UNKNOWN is more useful than a guess.
- Do not reward confident writing. Reward evidence.

Reply with a single JSON object and nothing else:

{
  "score": <float 0.0-1.0>,
  "verdict": "PASS" | "FAIL" | "UNKNOWN",
  "uncertainty": <float 0.0-1.0>,
  "evidence": [{"quote": "<verbatim from the material>", "why": "<what it shows>"}],
  "reasoning": "<two sentences at most>"
}""",
))

# Each dimension is its own rubric and its own prompt version.
DIMENSIONS: dict[str, VersionedPrompt] = {
    "groundedness": register(VersionedPrompt(
        name="judge_groundedness", version="v1",
        text="Is every substantive claim in the agent's output supported by something it "
             "actually observed during the run? Penalise claims with no observation behind "
             "them, and claims that overstate what an observation showed. Do not penalise "
             "an agent for correctly reporting that something is unverified.",
    )),
    "coverage": register(VersionedPrompt(
        name="judge_coverage", version="v1",
        text="Does the output address every part of what was asked? Identify anything "
             "requested but not delivered. Do not reward extra work that was not asked for, "
             "and do not penalise it either -- judge coverage of the request only.",
    )),
    "correctness": register(VersionedPrompt(
        name="judge_correctness", version="v1",
        text="Judged against the stated success criteria, is the substance of the output "
             "correct? Point at the specific claim that is wrong if it is wrong. If "
             "correctness depends on facts you cannot see in the material, return UNKNOWN "
             "rather than assuming.",
    )),
    "instruction_following": register(VersionedPrompt(
        name="judge_instruction_following", version="v1",
        text="Did the agent follow the explicit instructions and constraints in the task, "
             "including anything it was told not to do? Quote the instruction and the "
             "behaviour when they conflict.",
    )),
    "clarity": register(VersionedPrompt(
        name="judge_clarity", version="v1",
        text="Would a competent engineer who did not watch the run understand what was "
             "done, why, and what remains uncertain? Judge communication only -- never "
             "correctness, and never length for its own sake.",
    )),
}


@grader("llm_judge")
def llm_judge(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    """Grade one isolated dimension with a model.

    params:
      dimension  one of DIMENSIONS, or omit and supply `rubric`
      rubric     custom rubric text (requires `name` for the prompt id)
      threshold  score at or above which the verdict is PASS (default 0.7)
      include    what to show the judge: "output" (default), "trajectory", "both"
    """
    judge = getattr(ctx, "judge", None)
    if judge is None:
        return unknown(
            evidence=[{"source": "config",
                       "observation": "no judge model configured for this run"}],
            error="no judge configured -- pass --judge-provider, or set judge_provider "
                  "in agentwb.json; deterministic graders still ran",
        )

    dimension = params.get("dimension")
    if dimension and dimension in DIMENSIONS:
        rubric_prompt = DIMENSIONS[dimension]
        rubric, prompt_id = rubric_prompt.text, rubric_prompt.id
    elif params.get("rubric"):
        rubric = params["rubric"]
        prompt_id = f"judge_custom_{params.get('name', 'unnamed')}:v1"
    else:
        return unknown(
            evidence=[{"source": "config", "known_dimensions": sorted(DIMENSIONS)}],
            error=f"unknown dimension {dimension!r} and no rubric supplied",
        )

    material = _material(ctx, judge, params.get("include", "output"))
    if not material.strip():
        return unknown(
            evidence=[{"source": "trajectory", "observation": "run produced nothing to judge"}],
            error="no material available for judging",
        )

    user = (
        f"DIMENSION: {dimension or params.get('name', 'custom')}\n"
        f"RUBRIC: {rubric}\n\n"
        f"TASK GIVEN TO THE AGENT:\n{ctx.task.prompt}\n\n"
        f"SUCCESS CRITERIA:\n{ctx.task.success_criteria or '(none stated)'}\n\n"
        f"MATERIAL TO JUDGE:\n{material}"
    )

    try:
        reply = judge.ask_json(JUDGE_SYSTEM.text, user, prompt_id=prompt_id)
    except JudgeError as exc:
        return unknown(
            evidence=[{"source": "judge", "prompt_id": prompt_id}],
            error=str(exc),
        )

    return _to_result(reply, prompt_id, float(params.get("threshold", DEFAULT_THRESHOLD)))


def _to_result(reply, prompt_id: str, threshold: float) -> GraderResult:
    d = reply.data
    stated = str(d.get("verdict", "")).upper()
    evidence = _normalise_evidence(d.get("evidence"))
    meta = {"source": "judge", "prompt_id": prompt_id, "model": reply.model,
            "reasoning": str(d.get("reasoning", ""))[:600]}

    if stated == "UNKNOWN":
        return unknown(evidence=[meta, *evidence],
                       error="judge reported insufficient evidence to decide")

    score = _as_float(d.get("score"))
    if score is None:
        return unknown(evidence=[meta, *evidence],
                       error=f"judge returned no usable score (got {d.get('score')!r})")

    # A verdict with nothing to point at is an opinion, not a measurement.
    if not evidence:
        return unknown(evidence=[meta],
                       error="judge returned a score with no supporting evidence")

    uncertainty = _as_float(d.get("uncertainty")) or 0.0
    result = ok(score, [meta, *evidence], uncertainty) if score >= threshold \
        else bad(score, [meta, *evidence], uncertainty)

    # The model's own verdict is advisory; the threshold decides, so that one
    # configured number governs the pass line rather than model mood.
    if stated in {"PASS", "FAIL"} and (stated == "PASS") != (result.verdict is GraderVerdict.PASS):
        meta["threshold_override"] = (
            f"judge said {stated} but score {score} vs threshold {threshold} decides"
        )
    return result


def _material(ctx: GradingContext, judge, include: str) -> str:
    parts: list[str] = []
    final = (ctx.trajectory.final_output or {}).get("text", "")
    if include in {"output", "both"} and final:
        parts.append(f"--- AGENT'S FINAL OUTPUT ---\n{final}")
    if include in {"trajectory", "both"}:
        parts.append("--- WHAT THE AGENT ACTUALLY OBSERVED ---\n" + _trajectory_digest(ctx))
    return judge.truncate_evidence("\n\n".join(parts))


def _trajectory_digest(ctx: GradingContext) -> str:
    """Compact action/observation log -- the evidence a groundedness judge needs."""
    lines: list[str] = []
    for s in ctx.trajectory.steps:
        action = s.action or {}
        if action.get("type") != "tool_call":
            continue
        obs = (s.observation or {}).get("summary", "")
        obs = obs if len(obs) <= 500 else obs[:500] + " …[elided]"
        lines.append(f"[step {s.step}] {action.get('tool')}({_brief(action.get('arguments'))})\n{obs}")
    return "\n\n".join(lines) or "(the agent took no actions)"


def _brief(arguments: Optional[dict]) -> str:
    if not arguments:
        return ""
    text = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in arguments.items())
    return text if len(text) <= 200 else text[:200] + "…"


def _normalise_evidence(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[:10]:
        if isinstance(item, dict):
            quote, why = item.get("quote"), item.get("why")
            if quote:
                out.append({"quote": str(quote)[:600], "why": str(why or "")[:300]})
        elif isinstance(item, str) and item.strip():
            out.append({"quote": item[:600], "why": ""})
    return out


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))
