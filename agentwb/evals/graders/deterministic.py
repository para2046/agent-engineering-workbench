"""Deterministic graders.

Two families:

*Outcome* graders inspect the environment after the run -- did the tests pass,
does the file exist, does it contain what it should. These are the ones you
want: they judge the result, not the route (spec section 8).

*Trajectory* graders inspect how the agent worked, and exist for the cases
where process genuinely is the requirement: tool restrictions, required
verification, cost limits.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from ...aci.guardrails import safe_env
from ...types import GraderResult
from .base import GradingContext, bad, grader, ok, unknown


# --------------------------------------------------------------------------
# outcome graders
# --------------------------------------------------------------------------

@grader("tests_pass")
def tests_pass(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    """Run the workspace test suite ourselves and believe only the exit code.

    Deliberately independent of whatever the agent claimed or of any test run
    that happened during the trajectory: the grader re-executes.
    """
    target = params.get("target")
    timeout = int(params.get("timeout", 120))
    use_pytest = _has_module("pytest")

    if use_pytest:
        cmd = [sys.executable, "-m", "pytest", "-q"] + ([target] if target else [])
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-q"] if not target else \
              [sys.executable, "-m", "unittest", target]

    try:
        proc = subprocess.run(cmd, cwd=str(ctx.workspace), capture_output=True, text=True,
                              timeout=timeout, env=safe_env())
    except subprocess.TimeoutExpired:
        return unknown(evidence=[{"cmd": " ".join(cmd)}], error=f"test run timed out after {timeout}s")
    except OSError as exc:
        return unknown(error=f"could not launch tests: {exc}")

    out = (proc.stdout or "") + (proc.stderr or "")
    evidence = [{
        "source": "test_runner",
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "observation": out[-2000:],
    }]
    if proc.returncode == 0 and _ran_any_tests(out):
        return ok(1.0, evidence)
    if proc.returncode == 0:
        # a green exit with zero tests collected proves nothing
        return unknown(evidence, error="test runner exited 0 but collected no tests")
    return bad(0.0, evidence)


@grader("file_exists")
def file_exists(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    path = params["path"]
    target = ctx.workspace / path
    exists = target.is_file()
    ev = [{"source": "filesystem", "path": path, "observation": "present" if exists else "absent"}]
    return ok(1.0, ev) if exists else bad(0.0, ev)


@grader("file_contains")
def file_contains(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    path = params["path"]
    target = ctx.workspace / path
    if not target.is_file():
        return bad(0.0, [{"source": "filesystem", "path": path, "observation": "file absent"}])
    text = target.read_text(encoding="utf-8", errors="replace")
    if "pattern" in params:
        hit = re.search(params["pattern"], text) is not None
        needle = params["pattern"]
    else:
        needle = params["text"]
        hit = needle in text
    if params.get("absent"):
        hit = not hit
    ev = [{"source": "filesystem", "path": path, "needle": needle,
           "observation": "matched" if hit else "not matched"}]
    return ok(1.0, ev) if hit else bad(0.0, ev)


@grader("output_matches")
def output_matches(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    """Check the agent's final text. Weak evidence -- keep it non-required."""
    text = (ctx.trajectory.final_output or {}).get("text", "")
    if not text:
        return unknown(error="run produced no final text")
    pattern = params.get("pattern")
    hit = re.search(pattern, text, re.I) is not None if pattern else params["text"].lower() in text.lower()
    ev = [{"source": "final_output", "needle": pattern or params.get("text"),
           "observation": text[:500]}]
    return ok(1.0, ev) if hit else bad(0.0, ev)


@grader("no_forbidden_changes")
def no_forbidden_changes(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    """Assert that protected paths were never written.

    Reads the trajectory rather than the filesystem: we care that the agent
    did not attempt the modification, not merely that the file looks unchanged.
    """
    protected = list(params.get("paths") or [])
    offences = []
    for step in ctx.trajectory.steps:
        action = step.action or {}
        if action.get("type") != "tool_call":
            continue
        if action.get("tool") not in {"write_file", "edit_file", "shell"}:
            continue
        blob = str(action.get("arguments") or {})
        for p in protected:
            if p in blob:
                offences.append({"step": step.step, "tool": action.get("tool"), "path": p})
    ev = [{"source": "trajectory", "protected": protected, "violations": offences}]
    return ok(1.0, ev) if not offences else bad(0.0, ev)


# --------------------------------------------------------------------------
# trajectory graders
# --------------------------------------------------------------------------

@grader("tool_used")
def tool_used(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    """Require (or forbid) use of a given tool. For process requirements only:
    verification, authorization, cost -- not for dictating a solution route."""
    name = params["tool"]
    forbidden = bool(params.get("forbidden", False))
    used = [s.step for s in ctx.trajectory.steps
            if (s.action or {}).get("tool") == name]
    ev = [{"source": "trajectory", "tool": name, "used_at_steps": used}]
    hit = bool(used)
    passed = (not hit) if forbidden else hit
    return ok(1.0, ev) if passed else bad(0.0, ev)


@grader("max_steps")
def max_steps(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    """Efficiency bound. Non-required by convention -- a slow pass is still a pass."""
    limit = int(params["limit"])
    used = len(ctx.trajectory.steps)
    ev = [{"source": "trajectory", "steps": used, "limit": limit}]
    return ok(1.0, ev) if used <= limit else bad(0.0, ev)


@grader("terminated_with")
def terminated_with(ctx: GradingContext, params: dict[str, Any]) -> GraderResult:
    expected = params.get("reason") or params.get("reasons")
    expected_set = {expected} if isinstance(expected, str) else set(expected or [])
    actual = ctx.trajectory.termination_reason
    ev = [{"source": "trajectory", "expected": sorted(expected_set), "observation": actual}]
    return ok(1.0, ev) if actual in expected_set else bad(0.0, ev)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _ran_any_tests(output: str) -> bool:
    if re.search(r"\d+ passed", output):
        return True
    m = re.search(r"^Ran (\d+) test", output, re.M)
    if m:
        return int(m.group(1)) > 0
    return "no tests ran" not in output.lower()
