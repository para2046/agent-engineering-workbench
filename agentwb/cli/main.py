"""agentwb command line interface.

Designed so that Claude Code can drive it as easily as a human can: every
command prints a compact human summary by default and full JSON with --json.

    agentwb run tasks/fix_divide_bug.json
    agentwb eval tasks/ --trials 3
    agentwb inspect run_017
    agentwb compare run_017 run_021
    agentwb failures --last 20
    agentwb regress
    agentwb annotate run_017 --verdict FAIL --note "grader was wrong"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..evals.harness import Harness, load_tasks
from ..experience.store import ExperienceStore
from ..experiments.comparison import compare, summarize_task_results
from ..providers.base import ProviderError, available_providers, build_provider
from ..trajectories.store import TrajectoryStore
from ..types import GraderVerdict, Trajectory

# ANSI is off unless we're on a tty -- piped output should stay clean.
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


GREEN, RED, DIM, BOLD, YELLOW = "32", "31", "2", "1", "33"


def _verdict_mark(v: str) -> str:
    return {"PASS": _c(GREEN, "PASS"), "FAIL": _c(RED, "FAIL"),
            "UNKNOWN": _c(YELLOW, "UNKNOWN")}.get(v, v)


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.settings = Settings.load(Path(args.data_dir) if args.data_dir else None)
        self.store = TrajectoryStore(self.settings.data_dir)
        self.experience = ExperienceStore(self.settings.data_dir / "experience")
        self.harness = Harness(
            self.store, self.experience, self.settings.data_dir / "workspaces"
        )
        self.json = getattr(args, "json", False)

    def provider(self, args: argparse.Namespace):
        key = args.provider or self.settings.provider
        kwargs: dict[str, Any] = {}
        if getattr(args, "model", None):
            kwargs["model"] = args.model
        elif self.settings.model:
            kwargs["model"] = self.settings.model
        return build_provider(key, **kwargs)

    def emit(self, payload: dict, text: str) -> None:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str) if self.json else text)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    tasks = load_tasks(Path(args.task))
    if not tasks:
        print(f"no task files found at {args.task}", file=sys.stderr)
        return 2
    provider = ctx.provider(args)

    results = []
    for task in tasks:
        result = ctx.harness.run_task(task, provider, trials=args.trials,
                                      tool_timeout=args.timeout)
        results.append(result)
        for tr in result.trials:
            _print_trial(ctx, tr.trajectory)

    summary = summarize_task_results(results)
    ctx.emit({"summary": summary,
              "runs": [t.trajectory.trajectory_id for r in results for t in r.trials]},
             _format_summary(summary))
    return 0 if summary["passed"] == summary["trials"] else 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Re-grade existing runs, or run a suite. Without --regrade this is `run`
    over a directory of tasks -- the capability suite."""
    if not args.regrade:
        return cmd_run(args)

    ctx = Ctx(args)
    run_id = ctx.store.resolve(args.task)
    traj = ctx.store.load(run_id)
    tasks = {t.id: t for t in load_tasks(Path(args.tasks))} if args.tasks else {}
    task = tasks.get(traj.task_id)
    if task is None:
        print(f"cannot re-grade: no task definition for {traj.task_id!r} "
              f"(pass --tasks <dir>)", file=sys.stderr)
        return 2
    ws = Path(traj.workspace)
    if not ws.is_dir():
        print(f"cannot re-grade: workspace {ws} no longer exists", file=sys.stderr)
        return 2
    traj.evaluation = ctx.harness.grade(task, traj, ws)
    ctx.store.write(traj)
    _print_trial(ctx, traj)
    return 0 if traj.evaluation.passed else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    traj = ctx.store.load(ctx.store.resolve(args.run_id))
    if ctx.json:
        print(json.dumps(traj.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 0

    ev = traj.evaluation
    print(f"{_c(BOLD, traj.trajectory_id)}  task={traj.task_id}  "
          f"{traj.provider}/{traj.model}  prompt={traj.prompt_version}")
    print(f"  started {traj.started_at}  ended {traj.ended_at}  "
          f"termination={traj.termination_reason}")
    print(f"  metrics: {json.dumps(traj.metrics)}")
    if traj.failure_categories:
        print(f"  failure labels: {_c(YELLOW, ', '.join(traj.failure_categories))}")
    print()

    for s in traj.steps:
        action = s.action or {}
        if action.get("type") == "tool_call":
            res = s.tool_result or {}
            mark = _c(GREEN, "ok") if res.get("ok") else _c(RED, "ERR")
            args_str = json.dumps(action.get("arguments", {}), default=str)
            if len(args_str) > 140:
                args_str = args_str[:140] + "…"
            print(f"  {s.step:>3}. [{mark}] {_c(BOLD, action.get('tool',''))} {_c(DIM, args_str)}")
            obs = (s.observation or {}).get("summary", "")
            for line in obs.splitlines()[: args.obs_lines]:
                print(f"        {_c(DIM, line[:160])}")
            if len(obs.splitlines()) > args.obs_lines:
                print(f"        {_c(DIM, f'... [+{len(obs.splitlines()) - args.obs_lines} lines]')}")
        else:
            print(f"  {s.step:>3}. [{_c(DIM, 'final')}] {(s.rationale or '')[:400]}")
    print()

    if ev:
        print(f"  evaluation: {_verdict_mark('PASS' if ev.passed else 'FAIL')}  "
              f"score={ev.score}")
        for r in ev.results:
            extra = f"  ({r.error})" if r.error else ""
            req = "" if r.required else _c(DIM, " [advisory]")
            print(f"    - {r.grader}: {_verdict_mark(r.verdict.value)}"
                  f" score={r.score}{req}{extra}")
        if ev.notes:
            print(f"    {_c(YELLOW, ev.notes)}")
    print(f"\n  workspace: {traj.workspace}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    base = ctx.store.load(ctx.store.resolve(args.baseline))
    cand = ctx.store.load(ctx.store.resolve(args.candidate))
    result = compare(base, cand)

    if ctx.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    print(f"{_c(BOLD, 'baseline')}  {base.trajectory_id}")
    print(f"{_c(BOLD, 'candidate')} {cand.trajectory_id}\n")
    if result.config_differences:
        print("  config differences:")
        for k, (a, b) in result.config_differences.items():
            print(f"    {k}: {a} -> {b}")
    else:
        print("  config: identical")
    o = result.outcome
    print(f"\n  outcome: {_verdict_mark('PASS' if o['baseline_passed'] else 'FAIL')}"
          f" -> {_verdict_mark('PASS' if o['candidate_passed'] else 'FAIL')}"
          f"   score {o['baseline_score']} -> {o['candidate_score']}")
    if result.metric_differences:
        print("\n  metrics:")
        for k, d in result.metric_differences.items():
            arrow = {"better": _c(GREEN, "better"), "worse": _c(RED, "worse"),
                     "same": _c(DIM, "same")}[d["direction"]]
            print(f"    {k:<15} {d['baseline']:>8} -> {d['candidate']:>8}  "
                  f"({d['delta']:+})  {arrow}")
    if result.confounded:
        print(f"\n  {_c(RED, 'CONFOUNDED_EXPERIMENT')} -- metric deltas are not attributable")
    for w in result.warnings:
        print(f"  {_c(YELLOW, 'warning')}: {w}")
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    rows = ctx.experience.failures(limit=args.last, task_id=args.task_id,
                                   category=args.category)
    if ctx.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    counts = ctx.experience.category_counts()
    if counts:
        print(_c(BOLD, "failure categories:"))
        for cat, n in counts.items():
            print(f"  {n:>4}  {cat}")
        print()
    if not rows:
        print("no failures recorded")
        return 0
    print(_c(BOLD, f"last {len(rows)} failures:"))
    for e in rows:
        cats = ", ".join(e.get("failure_categories") or []) or "-"
        print(f"  {e['trajectory_id']}  task={e['task']['id']:<24} {_c(YELLOW, cats)}")
    return 0


def cmd_regress(args: argparse.Namespace) -> int:
    """Run the regression suite: every task tagged `regression`."""
    ctx = Ctx(args)
    tasks = [t for t in load_tasks(Path(args.tasks)) if "regression" in t.tags]
    if not tasks:
        print(f"no tasks tagged 'regression' in {args.tasks}")
        return 0
    provider = ctx.provider(args)
    results = [ctx.harness.run_task(t, provider, trials=args.trials) for t in tasks]
    for r in results:
        for tr in r.trials:
            _print_trial(ctx, tr.trajectory)
    summary = summarize_task_results(results)
    ctx.emit({"summary": summary}, _format_summary(summary))
    return 0 if summary["passed"] == summary["trials"] else 1


def cmd_annotate(args: argparse.Namespace) -> int:
    """Attach a human correction to a run (spec section 29)."""
    ctx = Ctx(args)
    run_id = ctx.store.resolve(args.run_id)
    entry = ctx.experience.add_correction(
        trajectory_id=run_id,
        author=args.author,
        verdict=args.verdict,
        categories=args.category or [],
        note=args.note,
    )
    ctx.emit(entry, f"recorded human correction for {run_id} "
                    f"(verdict={args.verdict or '-'}, author={args.author})")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    tasks = load_tasks(Path(args.tasks))
    payload = [{"id": t.id, "tags": t.tags, "graders": [g.label for g in t.graders],
                "trials": t.trials, "max_steps": t.max_steps} for t in tasks]
    if ctx.json:
        print(json.dumps(payload, indent=2))
        return 0
    for t in tasks:
        tags = f" [{', '.join(t.tags)}]" if t.tags else ""
        print(f"  {_c(BOLD, t.id)}{_c(DIM, tags)}")
        print(f"      graders: {', '.join(g.label for g in t.graders) or '-'}")
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    n = ctx.store.reindex()
    print(f"reindexed {n} runs into {ctx.store.db_path}")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    for p in available_providers():
        print(f"  {p}")
    return 0


# --------------------------------------------------------------------------
# printing helpers
# --------------------------------------------------------------------------

def _print_trial(ctx: Ctx, traj: Trajectory) -> None:
    if ctx.json:
        return
    ev = traj.evaluation
    mark = _verdict_mark("PASS" if (ev and ev.passed) else "FAIL")
    fails = [r.grader for r in (ev.results if ev else []) if r.verdict is not GraderVerdict.PASS]
    detail = f"  failing: {', '.join(fails)}" if fails else ""
    print(f"  [{mark}] {traj.trajectory_id}  task={traj.task_id}  "
          f"steps={traj.metrics.get('steps')}  term={traj.termination_reason}{detail}")


def _format_summary(s: dict) -> str:
    rate = s["success_rate"] * 100
    return (f"\n{_c(BOLD, 'summary')}: {s['passed']}/{s['trials']} trials passed "
            f"({rate:.0f}%) across {s['tasks']} task(s)")


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentwb", description="Agent Engineering Workbench")
    p.add_argument("--data-dir", help="where trajectories/experience live (default ./data)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)

    def add_provider_flags(sp):
        sp.add_argument("--provider", help="mock | claude | openai (default from config)")
        sp.add_argument("--model", help="model id override")

    sp = sub.add_parser("run", help="run a task (or a directory of tasks)")
    sp.add_argument("task")
    sp.add_argument("--trials", type=int, help="override trial count")
    sp.add_argument("--timeout", type=int, default=60, help="per-tool timeout in seconds")
    add_provider_flags(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("eval", help="run a suite, or re-grade an existing run")
    sp.add_argument("task", help="task file/dir, or a run id with --regrade")
    sp.add_argument("--regrade", action="store_true", help="re-grade an existing run in place")
    sp.add_argument("--tasks", help="task directory (needed for --regrade)")
    sp.add_argument("--trials", type=int)
    sp.add_argument("--timeout", type=int, default=60)
    add_provider_flags(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("inspect", help="show a trajectory step by step")
    sp.add_argument("run_id")
    sp.add_argument("--obs-lines", type=int, default=4, help="observation lines per step")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("compare", help="compare two runs")
    sp.add_argument("baseline")
    sp.add_argument("candidate")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("failures", help="list recorded failures and their categories")
    sp.add_argument("--last", type=int, default=20)
    sp.add_argument("--task-id")
    sp.add_argument("--category")
    sp.set_defaults(func=cmd_failures)

    sp = sub.add_parser("regress", help="run every task tagged 'regression'")
    sp.add_argument("--tasks", default="tasks")
    sp.add_argument("--trials", type=int)
    add_provider_flags(sp)
    sp.set_defaults(func=cmd_regress)

    sp = sub.add_parser("annotate", help="attach a human correction to a run")
    sp.add_argument("run_id")
    sp.add_argument("--verdict", choices=["PASS", "FAIL", "UNKNOWN"])
    sp.add_argument("--category", action="append", help="reclassified failure category")
    sp.add_argument("--note", default="")
    sp.add_argument("--author", default="human")
    sp.set_defaults(func=cmd_annotate)

    sp = sub.add_parser("tasks", help="list task definitions")
    sp.add_argument("--tasks", default="tasks")
    sp.set_defaults(func=cmd_tasks)

    sp = sub.add_parser("reindex", help="rebuild the SQLite index from JSON on disk")
    sp.set_defaults(func=cmd_reindex)

    sp = sub.add_parser("providers", help="list available model providers")
    sp.set_defaults(func=cmd_providers)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
