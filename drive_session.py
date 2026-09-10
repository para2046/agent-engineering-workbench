"""Drive a workbench run with a human (or assistant) as the model.

    python drive_session.py tasks/fix_divide_bug_unguided.json

Each pass replays the answers given so far and stops at the first unanswered
turn, printing the prompt the model would have received. Append your reply to
`data/session/answers.json` and run again. When every turn is answered, the run
completes, the graders execute against the environment, and the verdict prints.

The graders are the point. They re-run the test suite themselves, so whatever
the transcript claims, the environment is what decides.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentwb.evals.harness import Harness, load_task           # noqa: E402
from agentwb.experience.store import ExperienceStore           # noqa: E402
from agentwb.providers.session import NeedsAnswer, SessionProvider  # noqa: E402
from agentwb.trajectories.store import TrajectoryStore         # noqa: E402

DATA = Path("data")
SESSION = DATA / "session"


def show_pending(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{'=' * 70}\nTURN {payload['turn']} -- needs an answer\n{'=' * 70}")

    if payload["turn"] == 0:
        print("\n--- SYSTEM PROMPT ---")
        print(payload["system"])
        print("\n--- TOOLS AVAILABLE ---")
        for t in payload["tools"]:
            required = t["parameters"].get("required") or []
            props = ", ".join(t["parameters"].get("properties", {}))
            print(f"  {t['name']}({props})"
                  + (f"  required: {', '.join(required)}" if required else ""))
            print(f"      {t['description']}")

    print("\n--- CONVERSATION SO FAR ---")
    for m in payload["messages"]:
        if m["role"] == "user" and not m.get("tool"):
            print(f"\n[user]\n{m['content']}")
        elif m["role"] == "assistant":
            for c in (m.get("tool_calls") or []):
                print(f"\n[assistant] calls {c['name']}({json.dumps(c['arguments'])})")
            if m["content"]:
                print(f"\n[assistant] {m['content']}")
        elif m["role"] == "tool":
            print(f"\n[observation from {m.get('tool')}]\n{m['content']}")

    print(f"\n{'=' * 70}")
    print("Append one of these to data/session/answers.json, then re-run:")
    print('  {"tool": "run_tests", "arguments": {}}')
    print('  {"text": "what I changed and the evidence it worked"}')
    print("=" * 70)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    task = load_task(Path(sys.argv[1]))
    SESSION.mkdir(parents=True, exist_ok=True)
    if not (SESSION / "answers.json").is_file():
        (SESSION / "answers.json").write_text("[]", encoding="utf-8")

    store = TrajectoryStore(DATA)
    harness = Harness(store, ExperienceStore(DATA / "experience"), DATA / "workspaces")
    provider = SessionProvider(root=str(SESSION))

    answered = len(provider.load_answers())
    print(f"task: {task.id}   answers so far: {answered}")

    try:
        result = harness.run_task(task, provider, trials=1)
    except NeedsAnswer as need:
        show_pending(need.pending_path)
        return 10          # distinct code: not an error, just not finished

    traj = result.trials[0].trajectory
    ev = traj.evaluation
    print(f"\n{'=' * 70}\nRUN COMPLETE: {traj.trajectory_id}")
    print(f"termination: {traj.termination_reason}   steps: {traj.metrics.get('steps')}")
    print(f"\nVERDICT: {'PASS' if ev and ev.passed else 'FAIL'}   score={ev.score if ev else None}")
    for r in (ev.results if ev else []):
        flag = "" if r.required else "  [advisory]"
        print(f"  - {r.grader}: {r.verdict.value}{flag}"
              + (f"   ({r.error})" if r.error else ""))
    print(f"\nworkspace: {traj.workspace}")
    store.close()
    return 0 if (ev and ev.passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
