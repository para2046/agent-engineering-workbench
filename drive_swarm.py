"""Host a swarm: seed a work queue, let any number of workers claim tasks,
grade everything they hand back.

    python drive_swarm.py swarms/demo tasks/a.json tasks/b.json tasks/c.json \
        --after c=a,b --timeout 600

Workers are any external processes -- Claude Code subagents, humans, other
CLIs. A worker's whole contract:

    1. claim:   rename <swarm>/open/<id>.json -> <swarm>/claimed/<id>.<you>.json
                (rename failing means someone else won; try another ticket)
    2. work:    read the ticket; do the task in ticket["workspace"];
                touch your claim file now and then to keep the lease
    3. submit:  write <swarm>/done/<id>.<you>.json with
                {"task_id","worker","report","claims_success"}; delete your claim
    4. repeat until <swarm>/CLOSED exists

The host loop: release tickets whose dependencies passed, requeue stale
claims, and grade every submission by running the task's graders against the
workspace. Workers' success claims are recorded but decide nothing --
`graded/<id>.json` is written by the host alone, and dependent work unblocks
only on the host's PASS.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentwb.evals.harness import Harness, load_task          # noqa: E402
from agentwb.experience.store import ExperienceStore          # noqa: E402
from agentwb.swarm.queue import SwarmQueue, Ticket            # noqa: E402
from agentwb.trajectories.store import TrajectoryStore        # noqa: E402
from agentwb.types import Trajectory, new_run_id, utcnow      # noqa: E402


def build_workspace(task, root: Path) -> Path:
    ws = root / task.id
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    base = Path(task.source_path).parent if task.source_path else Path.cwd()
    if task.environment.template_dir:
        shutil.copytree((base / task.environment.template_dir).resolve(), ws,
                        dirs_exist_ok=True)
    for rel, content in task.environment.files.items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return ws


def grade(harness: Harness, task, workspace: Path, submission: dict) -> dict:
    """The host's verdict: run the task's graders against the real workspace.

    The trajectory here is a stub carrying the worker's report as the final
    output -- swarm workers act directly on files, so there is no tool
    transcript. Outcome graders (tests_pass, file_exists, ...) are unaffected;
    trajectory-only graders correctly return their failure/UNKNOWN rather than
    pretending a transcript existed.
    """
    traj = Trajectory(trajectory_id=new_run_id(), task_id=task.id,
                      provider="swarm", model=str(submission.get("worker", "?")),
                      workspace=str(workspace), started_at=utcnow())
    traj.final_output = {"text": str(submission.get("report", ""))}
    evaluation = harness.grade(task, traj, workspace)
    return {
        "worker": submission.get("worker"),
        "worker_claimed_success": bool(submission.get("claims_success")),
        "passed": evaluation.passed,
        "score": evaluation.score,
        "graders": [{"grader": r.grader, "verdict": r.verdict.value}
                    for r in evaluation.results],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("swarm", help="swarm directory (created if missing)")
    ap.add_argument("tasks", nargs="+", help="task JSON files to seed")
    ap.add_argument("--after", action="append", default=[],
                    help="dependency, e.g. c=a,b  (task c waits for a and b to PASS)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="host gives up after this many seconds")
    ap.add_argument("--lease", type=float, default=300.0,
                    help="seconds before an untouched claim is requeued")
    args = ap.parse_args()

    deps: dict[str, list[str]] = {}
    for spec in args.after:
        target, _, prereqs = spec.partition("=")
        deps[target.strip()] = [p.strip() for p in prereqs.split(",") if p.strip()]

    swarm_root = Path(args.swarm)
    queue = SwarmQueue(swarm_root, lease_seconds=args.lease)
    data_root = swarm_root / "host_data"
    harness = Harness(TrajectoryStore(data_root),
                      ExperienceStore(data_root / "experience"),
                      data_root / "unused_ws")

    tasks = {}
    for task_path in args.tasks:
        task = load_task(Path(task_path))
        tasks[task.id] = task
        ws = build_workspace(task, swarm_root / "workspaces")
        queue.seed(Ticket(
            task_id=task.id,
            task_file=str(Path(task_path).resolve()),
            workspace=str(ws.resolve()),
            instructions=("Do the task described in task_file, working directly in "
                          "`workspace`. Verify your work there (run the tests "
                          "yourself); the host re-runs all graders afterwards and "
                          "its verdict is the only one that counts."),
            after=deps.get(task.id, []),
        ))
    queue.release_unblocked()

    print(f"swarm open at {swarm_root}  tickets: {', '.join(tasks)}")
    for target, prereqs in deps.items():
        print(f"  {target} waits for: {', '.join(prereqs)}")
    print("workers may join at any time -- claim open/, submit to done/\n")

    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            for task_id in queue.requeue_stale_claims():
                print(f"  requeued stale claim: {task_id}")
            for submission in queue.pending_grades():
                task = tasks[submission["task_id"]]
                ticket_ws = swarm_root / "workspaces" / task.id
                verdict = grade(harness, task, ticket_ws, submission)
                queue.record_verdict(task.id, verdict["passed"], verdict)
                flag = "PASS" if verdict["passed"] else "FAIL"
                disagree = ("" if verdict["passed"] == verdict["worker_claimed_success"]
                            else "  <- disagrees with the worker's own claim")
                print(f"  graded {task.id}: {flag} (worker {submission['worker']})"
                      f"{disagree}")
            for task_id in queue.release_unblocked():
                print(f"  unblocked: {task_id} (dependencies passed)")

            status = queue.status()
            if status["graded"] == len(tasks):
                break
            time.sleep(2.0)
    finally:
        queue.close()

    status = queue.status()
    print(f"\nswarm complete: {status['passed']} passed, {status['failed']} failed, "
          f"{len(tasks) - status['graded']} ungraded")
    summary = {"status": status, "at": utcnow(),
               "verdicts": {tid: queue.verdict_of(tid) for tid in tasks}}
    (swarm_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if status["passed"] == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
