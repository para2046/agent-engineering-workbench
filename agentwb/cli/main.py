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

from ..analysis import failure_analyzer
from ..config import Settings
from ..costs import PriceBook
from ..evals.harness import Harness, load_tasks
from ..judge.client import JudgeClient
from ..experience.retrieval import ExperienceRetriever
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
        self.judge = self._build_judge(args)
        retrieve_k = getattr(args, "retrieve", 0) or 0
        self.retriever = (
            ExperienceRetriever(self.experience, k=retrieve_k) if retrieve_k > 0 else None
        )
        self.prices = PriceBook(self.settings.model_prices)
        self.harness = Harness(
            self.store, self.experience, self.settings.data_dir / "workspaces",
            judge=self.judge,
            analyze_failures=not getattr(args, "no_analysis", False),
            retriever=self.retriever,
            prices=self.prices,
        )
        self.json = getattr(args, "json", False)

    def _build_judge(self, args: argparse.Namespace):
        """A judge is optional. Without one, model graders return UNKNOWN --
        which is the honest outcome, not a silently skipped check."""
        key = getattr(args, "judge_provider", None) or self.settings.judge_provider
        if not key:
            return None
        kwargs: dict[str, Any] = {}
        model = getattr(args, "judge_model", None) or self.settings.judge_model
        if model:
            kwargs["model"] = model
        return JudgeClient(build_provider(key, **kwargs))

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


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyse why a run failed. Re-runs analysis unless --cached is given."""
    ctx = Ctx(args)
    if args.all_failures:
        rows = ctx.experience.failures(limit=args.last)
        run_ids = [r["trajectory_id"] for r in rows]
    else:
        if not args.run_id:
            print("give a run id, or --all-failures", file=sys.stderr)
            return 2
        run_ids = [ctx.store.resolve(args.run_id)]

    if not run_ids:
        print("no failed runs to analyse")
        return 0

    tasks = {t.id: t for t in load_tasks(Path(args.tasks))}
    out = []
    for run_id in run_ids:
        traj = ctx.store.load(run_id)
        task = tasks.get(traj.task_id)
        if task is None:
            print(f"skipping {run_id}: no task definition for {traj.task_id!r} "
                  f"in {args.tasks}", file=sys.stderr)
            continue
        analysis = None
        if args.cached:
            analysis = failure_analyzer.load(ctx.store.run_dir(run_id))
        if analysis is None:
            analysis = failure_analyzer.analyze(task, traj, judge=ctx.judge)
            failure_analyzer.save(analysis, ctx.store.run_dir(run_id))
        out.append(analysis)
        if not ctx.json:
            _print_analysis(analysis)

    if ctx.json:
        print(json.dumps([a.to_dict() for a in out], indent=2, default=str))
    return 0


def _print_analysis(a) -> None:
    src = _c(DIM, f"[{a.source}]") if a.source == "rules" else _c(BOLD, f"[{a.source}]")
    print(f"\n{_c(BOLD, a.trajectory_id)}  task={a.task_id}  {src}  "
          f"confidence={a.confidence}")
    if a.source == "rules":
        print(_c(DIM, "  (deterministic signals only -- no judge model configured)"))
    if a.error:
        print(f"  {_c(YELLOW, 'note')}: {a.error}")
    print(f"  {_c(BOLD, 'root causes:')}")
    for cause in a.root_causes:
        print(f"    - {cause}")
    if a.critical_step is not None:
        print(f"  critical step: {a.critical_step}")
    print(f"  avoidable: {a.avoidable}   optimizer candidate: {a.optimizer_candidate}")
    if a.categories:
        print(f"  categories: {_c(YELLOW, ', '.join(a.categories))}")
    if a.proposed_fix:
        print(f"  {_c(BOLD, 'proposed fix:')} {a.proposed_fix}")
    if a.recommended_regression_test:
        print(f"  {_c(BOLD, 'regression test:')} {a.recommended_regression_test}")
    print(_c(DIM, "  this is a hypothesis, not a verdict -- verify before acting on it"))


def cmd_prompts(args: argparse.Namespace) -> int:
    """List registered prompt versions (spec section 12)."""
    from .. import prompts as prompt_registry
    from ..analysis import failure_analyzer as _fa  # noqa: F401 -- registers its prompt
    from ..evals.graders import llm_judge as _lj  # noqa: F401 -- registers judge prompts

    ids = prompt_registry.registered()
    if getattr(args, "json", False):
        print(json.dumps({pid: prompt_registry.get(pid).text for pid in ids}, indent=2))
        return 0
    for pid in ids:
        vp = prompt_registry.get(pid)
        first = vp.text.strip().splitlines()[0][:90]
        parent = f"  (from {vp.parent})" if vp.parent else ""
        print(f"  {_c(BOLD, pid)}{parent}")
        print(f"      {_c(DIM, first)}")
    return 0


def cmd_multi_agent(args: argparse.Namespace) -> int:
    """Run a task through the two-agent protocol, in a real workspace.

    The environment is what makes this more than a transcript: the disagreement
    ladder can run a discriminating experiment, and the graders judge what the
    agents actually left behind.
    """
    from ..experiments.runner import run_exchange
    from ..judge.client import JudgeClient
    from ..runtime.orchestrator import Orchestrator, save_exchange

    ctx = Ctx(args)
    tasks = load_tasks(Path(args.task))
    if not tasks:
        print(f"no task files found at {args.task}", file=sys.stderr)
        return 2

    def build_client(provider_key, model):
        kwargs = {"model": model} if model else {}
        return JudgeClient(build_provider(provider_key or ctx.settings.provider, **kwargs))

    # Two seats. Which provider fills which is configuration, not architecture.
    agents = {
        "researcher": build_client(args.researcher_provider, args.researcher_model),
        "engineer": build_client(args.engineer_provider, args.engineer_model),
    }

    failures = 0
    for task in tasks:
        def factory(run_experiment, _task=task):
            return Orchestrator(
                agents,
                max_rounds=args.max_rounds,
                judge=ctx.judge,
                run_experiment=run_experiment,
            )

        run = run_exchange(
            task=task,
            orchestrator_factory=factory,
            workspaces_root=ctx.settings.data_dir / "exchanges",
            grade=ctx.harness.grade if task.graders else None,
            tool_timeout=args.timeout,
        )
        save_exchange(run.exchange, ctx.settings.data_dir / "exchanges" /
                      run.run_id / "exchange.json")

        if ctx.json:
            print(json.dumps(run.to_dict(), indent=2, default=str))
        else:
            _print_exchange(run)
        if not run.passed:
            failures += 1

    return 0 if failures == 0 else 1


def _print_exchange(run) -> None:
    ex = run.exchange
    metrics = ex.metrics if isinstance(ex.metrics, dict) else ex.metrics.to_dict()
    verdict = _verdict_mark("PASS" if run.passed else "FAIL") if run.evaluation else _c(DIM, "ungraded")
    print(f"[{verdict}] {_c(BOLD, run.run_id)}  task={run.task_id}  "
          f"term={ex.termination_reason}")

    for m in ex.messages:
        print(f"  {_c(DIM, f'r{m.round}')} {_c(BOLD, m.sender)} -> {m.recipient}  "
              f"{m.type.value}")
        print(f"      {m.claim[:200]}")
        for e in m.evidence[:2]:
            print(f"      {_c(DIM, f'evidence({e.source}): {e.observation[:140]}')}")

    for v in ex.violations:
        print(f"  {_c(RED, 'violation')} r{v.get('round')} {v.get('agent')}: {v.get('problem')}")

    for d in ex.disagreements:
        resolution = d.get("resolution") or d.get("outcome", {}).get("resolution")
        basis = d.get("basis") or d.get("outcome", {}).get("basis") or ""
        colour = GREEN if str(resolution).startswith("RESOLVED") else YELLOW
        print(f"  {_c(colour, 'disagreement')}: {resolution}"
              + (f" on {basis}" if basis else ""))

    print(f"  metrics: {json.dumps(metrics)}")
    if run.evaluation:
        for r in run.evaluation.results:
            print(f"    - {r.grader}: {_verdict_mark(r.verdict.value)}")
    print(f"  workspace: {run.workspace}")


def cmd_regress_from_failure(args: argparse.Namespace) -> int:
    """Turn a recorded failure into a permanent regression task (spec section 19).

    The flywheel's closing link: a fixed bug that never becomes a test can come
    back silently, and the failure data you already paid for is the cheapest
    source of a task that would catch it.
    """
    from ..analysis import failure_analyzer as fa

    ctx = Ctx(args)
    run_id = ctx.store.resolve(args.run_id)
    traj = ctx.store.load(run_id)

    source_tasks = {t.id: t for t in load_tasks(Path(args.tasks))}
    original = source_tasks.get(traj.task_id)
    if original is None:
        print(f"no task definition for {traj.task_id!r} in {args.tasks} -- cannot "
              f"reproduce the environment it failed in", file=sys.stderr)
        return 2

    analysis = fa.load(ctx.store.run_dir(run_id))
    new_id = args.id or f"regression_{traj.task_id}_{run_id[-6:]}"

    payload = {
        "id": new_id,
        "tags": sorted(set(original.tags) | {"regression"}),
        "prompt": original.prompt,
        "success_criteria": original.success_criteria,
        "environment": {"files": dict(original.environment.files),
                        "template_dir": original.environment.template_dir},
        "max_steps": original.max_steps,
        "trials": original.trials,
        "graders": [
            {"type": g.type, "name": g.label, "required": g.required,
             "weight": g.weight, "params": g.params}
            for g in original.graders
        ],
        "derived_from": {
            "trajectory_id": run_id,
            "failure_categories": traj.failure_categories,
            "termination_reason": traj.termination_reason,
            "recommended_test": (analysis.recommended_regression_test if analysis else ""),
            "note": "generated from a recorded failure; review the graders before "
                    "trusting this as a regression bar",
        },
    }

    out = Path(args.out or args.tasks) / f"{new_id}.json"
    if out.exists() and not args.force:
        print(f"{out} already exists -- pass --force to overwrite", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ctx.emit({"created": str(out), "task_id": new_id},
             f"wrote regression task {_c(BOLD, new_id)} -> {out}\n"
             + (f"  suggested check: {analysis.recommended_regression_test}\n"
                if analysis and analysis.recommended_regression_test else "")
             + _c(DIM, "  review the graders before trusting it as a regression bar"))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Propose candidate policies, evaluate them, and run the promotion gate.

    Promotes nothing without clearing the gate, and prints why when it refuses.
    """
    from ..optimization.datasets import DEV, REGRESSION, TEST, build_dataset
    from ..optimization.optimizer import (
        ReflectiveOptimizer,
        feedback_from_analyses,
        optimize,
    )
    from ..optimization.promotion import GateConfig, PromotionGate, result_from_task_results
    from ..prompts import get as get_prompt
    from ..runtime.agent import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION
    from ..prompts import VersionedPrompt, register

    ctx = Ctx(args)
    if ctx.judge is None:
        print("optimization needs a model to propose candidates -- pass "
              "--judge-provider (e.g. --judge-provider claude)", file=sys.stderr)
        return 2

    baseline = register(VersionedPrompt(
        name=SYSTEM_PROMPT_VERSION.split(":")[0],
        version=SYSTEM_PROMPT_VERSION.split(":")[1],
        text=SYSTEM_PROMPT,
    ))

    ds = build_dataset(load_tasks(Path(args.tasks)))
    dev = ds.for_optimizer(DEV)
    if not dev:
        print(f"no tasks in the dev split of {args.tasks} -- nothing to optimize against; "
              f"run `agentwb splits` to see the assignment", file=sys.stderr)
        return 2

    # Failure evidence, from stored analyses.
    from ..analysis import failure_analyzer as fa
    analyses = []
    for row in ctx.experience.failures(limit=args.feedback_from):
        loaded = fa.load(ctx.store.run_dir(row["trajectory_id"]))
        if loaded is not None:
            analyses.append(loaded)

    provider = ctx.provider(args)

    def evaluate(policy, tasks):
        results = [ctx.harness.run_task(t, provider, trials=args.trials, policy=policy)
                   for t in tasks]
        return result_from_task_results("dev", results, ctx.prices)

    gate = PromotionGate(GateConfig(
        max_cost_per_trial=args.max_cost,
        min_trials_for_confidence=args.min_trials,
    ), prices=ctx.prices)

    result = optimize(
        baseline=baseline,
        optimizer=ReflectiveOptimizer(ctx.judge),
        feedback=feedback_from_analyses(analyses),
        evaluate=evaluate,
        gate=gate,
        dev_tasks=dev,
        test_tasks=ds.for_final_evaluation() or None,
        regression_tasks=ds.split(REGRESSION) or None,
        n_candidates=args.candidates,
    )

    if ctx.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.promoted else 1

    print(f"{_c(BOLD, 'baseline')} {result.baseline_id}")
    for c in result.rejected:
        print(f"  {_c(RED, 'rejected')} {c.id} -- {c.rejected_reason}")
    for c in result.candidates:
        decision = result.evaluated.get(c.id)
        if decision is None:
            continue
        mark = _c(GREEN, "PROMOTE") if decision.promote else _c(RED, "REJECT")
        print(f"  [{mark}] {c.id}")
        if c.rationale:
            print(f"      {_c(DIM, c.rationale[:160])}")
        for check in decision.checks:
            tick = _c(GREEN, "ok") if check.passed else _c(RED, "no")
            print(f"      [{tick}] {check.name}: {check.detail}")
        for w in decision.warnings:
            print(f"      {_c(YELLOW, 'warning')}: {w}")
    for note in result.notes:
        print(f"  {_c(DIM, note)}")
    print(f"\n{_c(BOLD, result.summary())}")
    return 0 if result.promoted else 1


def cmd_splits(args: argparse.Namespace) -> int:
    """Show how tasks divide into train/dev/test/regression."""
    from ..optimization.datasets import SPLITS, build_dataset

    ctx = Ctx(args)
    ds = build_dataset(load_tasks(Path(args.tasks)))
    if ctx.json:
        print(json.dumps({s: sorted(t.id for t in ds.split(s)) for s in SPLITS}, indent=2))
        return 0
    for name in SPLITS:
        tasks = ds.split(name)
        label = _c(BOLD, f"{name} ({len(tasks)})")
        note = _c(DIM, "  -- held out; never optimized against") if name == "test" else ""
        print(f"  {label}{note}")
        for t in tasks:
            print(f"      {t.id}")
    print(_c(DIM, "\n  Assignment is a hash of the task id, so it is stable across "
                  "machines and runs. A `split` tag overrides it."))
    return 0


def cmd_prices(args: argparse.Namespace) -> int:
    """Show configured model prices; cost is UNKNOWN without them."""
    ctx = Ctx(args)
    if not ctx.prices:
        print("no model prices configured -- cost is reported as UNKNOWN, never as zero.")
        print("\nAdd to agentwb.json:")
        print('  "model_prices": {"<model>": '
              '{"input_per_mtok": 0.00, "output_per_mtok": 0.00}}')
        return 0
    for model in ctx.prices.models:
        price = ctx.prices.price_for(model)
        print(f"  {model}: ${price.input_per_mtok}/Mtok in, "
              f"${price.output_per_mtok}/Mtok out")
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

    def add_retrieval_flags(sp):
        sp.add_argument("--retrieve", type=int, metavar="K", default=0,
                        help="inject up to K relevant past experiences from earlier "
                             "runs on OTHER tasks (0 = off). Same-task retrieval is "
                             "always refused: it would leak the answer.")

    def add_judge_flags(sp):
        sp.add_argument("--judge-provider",
                        help="provider for model graders and failure analysis; "
                             "omit and they return UNKNOWN")
        sp.add_argument("--judge-model", help="judge model id override")

    sp = sub.add_parser("run", help="run a task (or a directory of tasks)")
    sp.add_argument("task")
    sp.add_argument("--trials", type=int, help="override trial count")
    sp.add_argument("--timeout", type=int, default=60, help="per-tool timeout in seconds")
    sp.add_argument("--no-analysis", action="store_true", help="skip failure analysis")
    add_provider_flags(sp)
    add_judge_flags(sp)
    add_retrieval_flags(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("eval", help="run a suite, or re-grade an existing run")
    sp.add_argument("task", help="task file/dir, or a run id with --regrade")
    sp.add_argument("--regrade", action="store_true", help="re-grade an existing run in place")
    sp.add_argument("--tasks", help="task directory (needed for --regrade)")
    sp.add_argument("--trials", type=int)
    sp.add_argument("--timeout", type=int, default=60)
    sp.add_argument("--no-analysis", action="store_true")
    add_provider_flags(sp)
    add_judge_flags(sp)
    add_retrieval_flags(sp)
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
    sp.add_argument("--no-analysis", action="store_true")
    add_provider_flags(sp)
    add_judge_flags(sp)
    add_retrieval_flags(sp)
    sp.set_defaults(func=cmd_regress)

    sp = sub.add_parser("annotate", help="attach a human correction to a run")
    sp.add_argument("run_id")
    sp.add_argument("--verdict", choices=["PASS", "FAIL", "UNKNOWN"])
    sp.add_argument("--category", action="append", help="reclassified failure category")
    sp.add_argument("--note", default="")
    sp.add_argument("--author", default="human")
    sp.set_defaults(func=cmd_annotate)

    sp = sub.add_parser("analyze",
                        help="analyse why a run failed (hypothesis, not verdict)")
    sp.add_argument("run_id", nargs="?")
    sp.add_argument("--all-failures", action="store_true",
                    help="analyse every recorded failure")
    sp.add_argument("--last", type=int, default=10, help="with --all-failures, how many")
    sp.add_argument("--tasks", default="tasks", help="task directory")
    sp.add_argument("--cached", action="store_true", help="reuse a stored analysis if present")
    add_judge_flags(sp)
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("prompts", help="list registered prompt versions")
    sp.set_defaults(func=cmd_prompts)

    sp = sub.add_parser("optimize",
                        help="propose candidate policies and run the promotion gate")
    sp.add_argument("--tasks", default="tasks")
    sp.add_argument("--candidates", type=int, default=2, help="how many to propose")
    sp.add_argument("--trials", type=int, default=1, help="trials per task per policy")
    sp.add_argument("--feedback-from", type=int, default=10,
                    help="how many recent failures to draw evidence from")
    sp.add_argument("--max-cost", type=float, default=None,
                    help="cost/trial ceiling; UNKNOWN cost blocks promotion when set")
    sp.add_argument("--min-trials", type=int, default=20,
                    help="below this, results are flagged as noise")
    add_provider_flags(sp)
    add_judge_flags(sp)
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("multi-agent",
                        help="run a task through the two-agent protocol in a real workspace")
    sp.add_argument("task")
    sp.add_argument("--max-rounds", type=int, default=4)
    sp.add_argument("--timeout", type=int, default=60)
    sp.add_argument("--researcher-provider", help="provider for the research seat")
    sp.add_argument("--researcher-model")
    sp.add_argument("--engineer-provider", help="provider for the engineering seat")
    sp.add_argument("--engineer-model")
    add_provider_flags(sp)
    add_judge_flags(sp)
    sp.set_defaults(func=cmd_multi_agent)

    sp = sub.add_parser("regression-from-failure",
                        help="turn a recorded failure into a permanent regression task")
    sp.add_argument("run_id")
    sp.add_argument("--tasks", default="tasks")
    sp.add_argument("--out", help="directory to write into (default --tasks)")
    sp.add_argument("--id", help="task id for the new regression task")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_regress_from_failure)

    sp = sub.add_parser("splits", help="show train/dev/test/regression assignment")
    sp.add_argument("--tasks", default="tasks")
    sp.set_defaults(func=cmd_splits)

    sp = sub.add_parser("prices", help="show configured model prices")
    sp.set_defaults(func=cmd_prices)

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
