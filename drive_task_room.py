"""Host a task room: N seats, one task, one shared workspace, one verdict.

    python drive_task_room.py rooms/build tasks_room/joint_stats.json \
        --seats 3 --max-rounds 6 --timeout 300

This process is the host: it builds ONE workspace from the task, opens N
seats, round-robins their turns, and finally grades the workspace with the
task's own graders. Seats coordinate through the public, append-only
<room>/board.jsonl; each seat's private history stays in its own directory
and is never shown to another seat. Agents' done-reports are recorded but
decide nothing -- episode.json carries the host's verdict alone.

A seat is filled one of two ways:

* **external occupant** (default): a file mailbox exactly like the discussion
  room -- any process watches <room>/<seat>/prompt.json and writes
  <room>/<seat>/reply.json (the prompt documents the reply format).
* **in-process provider**: `--seat-provider agent2=claude-cli:haiku` puts a
  model from any registered provider directly in the seat, no external
  process needed. Repeat the flag per seat; seats it does not name stay
  mailboxes, so one episode can mix API models, CLI models, subagents, and
  humans freely.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentwb.evals.harness import Harness, load_task             # noqa: E402
from agentwb.experience.store import ExperienceStore             # noqa: E402
from agentwb.judge.client import JudgeClient                     # noqa: E402
from agentwb.protocols.room import RoomSeat                      # noqa: E402
from agentwb.protocols.task_room import TaskRoom, parse_seat_names  # noqa: E402
from agentwb.providers.base import build_provider                # noqa: E402
from agentwb.trajectories.store import TrajectoryStore           # noqa: E402
from agentwb.types import Trajectory, new_run_id, utcnow         # noqa: E402


def build_workspace(task, ws: Path) -> Path:
    """One shared workspace for the whole room (same recipe as the swarm)."""
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


def grade(harness: Harness, task, workspace: Path, episode: dict) -> dict:
    """The host's verdict: the task's graders against the real workspace.

    The trajectory is a stub carrying the seats' combined reports as the
    final output -- room seats act directly on files, so there is no tool
    transcript. Outcome graders are unaffected; trajectory-only graders
    correctly return their failure/UNKNOWN rather than pretending a
    transcript existed. (Same pattern as drive_swarm.py.)
    """
    reports = "\n".join(
        f"[{seat}] {info.get('report')}"
        for seat, info in episode["seats"].items() if info.get("report"))
    traj = Trajectory(trajectory_id=new_run_id(), task_id=task.id,
                      provider="task_room",
                      model="+".join(episode["seats"]),
                      workspace=str(workspace), started_at=utcnow())
    traj.final_output = {"text": reports}
    evaluation = harness.grade(task, traj, workspace)
    return {
        "passed": evaluation.passed,
        "score": evaluation.score,
        "graders": [{"grader": r.grader, "verdict": r.verdict.value}
                    for r in evaluation.results],
        "seats_claiming_done": [seat for seat, info in episode["seats"].items()
                                if info.get("done_reason") == "reported done"],
    }


def parse_seat_providers(specs: list[str]) -> dict[str, tuple[str, str]]:
    """`agent2=claude-cli:haiku` -> {"agent2": ("claude-cli", "haiku")}."""
    out: dict[str, tuple[str, str]] = {}
    for spec in specs:
        seat, sep, rest = spec.partition("=")
        if not sep or not rest:
            raise SystemExit(f"--seat-provider wants name=providerkey[:model], "
                             f"got {spec!r}")
        key, _, model = rest.partition(":")
        out[seat.strip()] = (key.strip(), model.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("room", help="room directory (created if missing)")
    ap.add_argument("task", help="task JSON (harness format)")
    ap.add_argument("--seats", default="",
                    help="a count (`4` -> agent1..agent4) or comma-separated "
                         "names; default 3 seats")
    ap.add_argument("--seat-provider", action="append", default=[],
                    metavar="NAME=KEY[:MODEL]",
                    help="fill a seat with an in-process provider, e.g. "
                         "agent1=claude-cli:sonnet; unnamed seats stay "
                         "external file mailboxes")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds to wait for each mailbox seat's reply")
    args = ap.parse_args()

    seat_names = parse_seat_names(args.seats)
    providers = parse_seat_providers(args.seat_provider)
    unknown = set(providers) - set(seat_names)
    if unknown:
        raise SystemExit(f"--seat-provider names seats that do not exist: "
                         f"{', '.join(sorted(unknown))}")

    task = load_task(Path(args.task))
    room_dir = Path(args.room)
    workspace = build_workspace(task, room_dir / "workspace")

    seats: dict[str, object] = {}
    for name in seat_names:
        if name in providers:
            key, model = providers[name]
            kwargs = {"model": model} if model else {}
            seats[name] = JudgeClient(build_provider(key, **kwargs))
        else:
            seats[name] = RoomSeat(room_dir, name, timeout=args.timeout)

    kinds = ", ".join(f"{n}={'provider:' + providers[n][0] if n in providers else 'mailbox'}"
                      for n in seat_names)
    print(f"task room open at {room_dir}  task={task.id}")
    print(f"seats: {kinds}")
    print(f"workspace (shared): {workspace}")
    print("mailbox seats: answer <room>/<seat>/prompt.json with <seat>/reply.json\n")

    room = TaskRoom(room_dir, task_prompt=task.prompt, workspace=workspace,
                    seats=seats, max_rounds=args.max_rounds, task_id=task.id)
    try:
        episode = room.run()
    finally:
        room.close()

    data_root = room_dir / "host_data"
    harness = Harness(TrajectoryStore(data_root),
                      ExperienceStore(data_root / "experience"),
                      data_root / "unused_ws")
    episode["verdict"] = grade(harness, task, workspace, episode)
    path = room.save_episode(episode)

    v = episode["verdict"]
    flag = "PASS" if v["passed"] else "FAIL"
    print(f"episode over: {episode['termination_reason']} after "
          f"{episode['rounds_used']} round(s), {episode['board_messages']} "
          f"board message(s)")
    for seat, info in episode["seats"].items():
        print(f"  {seat}: turns={info['turns']} works={info['works']} "
              f"posts={info['posts']} violations={info['violations']} "
              f"done={info['done_reason'] or 'no'}")
    print(f"verdict: {flag} score={v['score']:.2f}  "
          f"graders: " + ", ".join(f"{g['grader']}={g['verdict']}"
                                   for g in v["graders"]))
    print(f"episode saved: {path}")
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
