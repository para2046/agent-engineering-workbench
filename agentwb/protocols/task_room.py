"""A task room: N seats, ONE task, ONE shared workspace, one honest grader.

The discussion room (room.py) gives external agents seats in a *conversation*;
the swarm (swarm/queue.py) gives them *claims on separate pieces of work*. The
task room is the missing combination: several agents on the SAME task in the
SAME workspace, coordinating in the open and judged by nobody's word.

Layout on disk (one directory per episode):

    room/
      board.jsonl              PUBLIC, append-only: every published message
      <seat>/prompt.json       a mailbox seat's current turn (external occupants)
      <seat>/reply.json        the occupant's one-JSON answer
      <seat>/private/history.jsonl  that seat's private record -- never shown
                                    to any other seat
      episode.json             board + per-seat turn counts + the HOST's verdict
      CLOSED                   the episode is over

Three properties carry the design:

**The board is the only shared channel, and it only grows.** A seat's turn
shows it the task, the full public board, and its OWN private history --
nothing of anyone else's. Coordination therefore has to happen in messages a
reader can audit afterwards, and no seat can quietly rewrite what it told the
others: publishing appends a line to ``board.jsonl``, and nothing in this
module can edit or remove one.

**Seats are clients, not vendors.** A seat is anything with the JudgeClient
shape -- ``ask_json(system, user) -> reply.data`` -- so an in-process model
(any provider), a file-mailbox occupant (``room.RoomSeat``: an external CLI,
a subagent, a human), and a scripted fixture all sit at the same table in the
same episode. Mixing vendors is configuration, not architecture.

**Everyone's word together decides nothing.** All N seats can post FINAL
reports agreeing the work is done; the host still runs the task's graders
against the workspace they actually left behind, and ``episode.json`` records
that verdict alone. This is the swarm's honesty boundary, held when the
workers share one workspace and can talk each other into anything.

A seat's turn is one JSON object:

    {"action": "work", "note": "<what I changed and why>"}
        after editing files in the workspace directly; a seat with no file
        access (an in-process model) may instead include
        "files": {"relative/path": "full new content"} and the host writes
        them -- confined to the workspace, never outside it.
    {"action": "post", "message": {"type": "<TYPE>", ...}}
        publish one typed message to the board.
    {"action": "done", "report": "<what I did and the evidence it works>"}
        leave the table; the episode ends when every seat has.

Turns round-robin with a hard round budget. A seat that stops speaking the
protocol wastes its turn (recorded as a violation, privately); a mailbox seat
that times out is retired with the reason on record rather than stalling the
table for everyone else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from ..judge.client import JudgeError
from ..types import utcnow

ACTIONS = ("work", "post", "done")

SEAT_SYSTEM = """\
You are seat {seat}, one of {n} agents solving the SAME task together in one
shared workspace at {workspace}.

Each turn you receive the task, the PUBLIC BOARD (everything any seat has
published), and your own PRIVATE history. Other seats never see your private
history -- if the others need to know something, publish it.

Reply with EXACTLY ONE JSON object and nothing else (no markdown fences):

  {{"action": "work", "note": "<what you changed in the workspace and why>"}}
      -- after directly editing files in the workspace yourself. If you cannot
         touch files directly, include "files": {{"relative/path.py": "<full
         new content>"}} and the host will write them for you (workspace
         paths only).
  {{"action": "post", "message": {{"type": "<PLAN|STATUS|QUESTION|BLOCKER|FINAL_REPORT|...>", ...}}}}
      -- publish one typed message to the shared board.
  {{"action": "done", "report": "<what you did and the evidence it works>"}}
      -- leave the table. Do this once your part is finished and verified.

Coordinate via the board: claim a piece before working on it, and say what you
finished. The host grades the WORKSPACE with the task's own graders after the
episode -- reports and posts decide nothing, so verify in the workspace before
claiming anything."""


@runtime_checkable
class SeatClient(Protocol):
    """What a seat must quack like: JudgeClient, RoomSeat, or a test script."""

    def ask_json(self, system: str, user: str, prompt_id: str = "") -> Any:
        ...


class TaskRoom:
    """Host-side state machine for one episode. Grading stays with the caller
    (see drive_task_room.py) so this module never has to import the harness --
    the room runs the conversation; the verdict is someone else's job."""

    def __init__(
        self,
        root: Path,
        task_prompt: str,
        workspace: Path,
        seats: dict[str, SeatClient],
        max_rounds: int = 8,
        task_id: str = "",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "CLOSED").unlink(missing_ok=True)
        self.task_prompt = task_prompt
        self.workspace = Path(workspace)
        self.seats = dict(seats)
        self.max_rounds = max_rounds
        self.task_id = task_id
        self.board_path = self.root / "board.jsonl"
        self._seq = 0
        self._stats: dict[str, dict[str, Any]] = {
            seat: {"turns": 0, "works": 0, "posts": 0, "violations": 0,
                   "done": False, "done_reason": None, "report": None}
            for seat in self.seats
        }

    # -- the public board (append-only) ------------------------------------
    def post(self, seat: str, message: dict[str, Any]) -> dict[str, Any]:
        """Append one typed message to the board. There is no edit and no
        delete, on purpose: the board is the audit trail of the coordination,
        and a seat must not be able to rewrite what the others acted on."""
        if not isinstance(message, dict) or not str(message.get("type") or "").strip():
            raise ValueError("a board message is a JSON object with a non-empty "
                             "'type' field")
        self._seq += 1
        entry = {"seq": self._seq, "sender": seat, "at": utcnow(),
                 "message": message}
        payload = json.dumps(entry, ensure_ascii=False) + "\n"
        # A host that died mid-append leaves a torn line with no newline; a
        # fresh append must not weld onto it and corrupt a GOOD message too.
        if self.board_path.is_file():
            with self.board_path.open("rb") as fh:
                size = fh.seek(0, 2)
                if size:
                    fh.seek(size - 1)
                    if fh.read(1) != b"\n":
                        payload = "\n" + payload
        with self.board_path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
        return entry

    def board(self) -> list[dict[str, Any]]:
        """Read the board, tolerating torn lines.

        A crash mid-append leaves a half-written line; under append-only
        discipline a COMPLETE message can never become unreadable afterwards,
        so every undecodable line is the torn tail of some crashed append --
        at the end of the file, or mid-file once a later host has appended
        past it. Skipping them loses only messages that were never fully
        published, and keeps every episode readable. (The unhandled torn tail
        was called out by a live sonnet-vs-haiku exchange reviewing this very
        file; the fix is theirs as much as anyone's.)
        """
        if not self.board_path.is_file():
            return []
        out = []
        for line in self.board_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # -- private histories --------------------------------------------------
    def _private_note(self, seat: str, entry: dict[str, Any]) -> None:
        path = self.root / seat / "private" / "history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": utcnow(), **entry}, ensure_ascii=False) + "\n")

    def _private_history(self, seat: str) -> list[dict[str, Any]]:
        path = self.root / seat / "private" / "history.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    # -- one seat's turn ----------------------------------------------------
    def seat_prompt(self, seat: str, round_no: int) -> tuple[str, str]:
        system = SEAT_SYSTEM.format(seat=seat, n=len(self.seats),
                                    workspace=self.workspace)
        user = json.dumps({
            "task": self.task_prompt,
            "workspace": str(self.workspace),
            "round": round_no,
            "max_rounds": self.max_rounds,
            "public_board": self.board(),
            "your_private_history": self._private_history(seat),
        }, indent=2, ensure_ascii=False)
        return system, user

    def _apply_files(self, seat: str, files: Any) -> list[str]:
        """Write an in-process seat's edits, confined to the workspace.

        A model that names a path outside the workspace -- absolute, or
        climbing out via `..` -- gets a ValueError, not a write: the shared
        workspace is the whole blast radius a seat is entitled to."""
        if not isinstance(files, dict):
            raise ValueError('"files" must map relative paths to full contents')
        ws = self.workspace.resolve()
        written = []
        for rel, content in files.items():
            target = (ws / str(rel)).resolve()
            if not target.is_relative_to(ws):
                raise ValueError(f"path escapes the workspace: {rel!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
            written.append(str(rel))
        return written

    def _turn(self, seat: str, round_no: int) -> None:
        stats = self._stats[seat]
        stats["turns"] += 1
        system, user = self.seat_prompt(seat, round_no)
        try:
            reply = self.seats[seat].ask_json(system, user,
                                              prompt_id=f"task_room/{seat}/r{round_no}")
            data = reply.data if hasattr(reply, "data") else reply
        except JudgeError as exc:
            # A mailbox occupant that crashed or wandered off must not stall
            # the whole table; retire the seat with the reason on record.
            stats["done"] = True
            stats["done_reason"] = f"timeout: {exc}"
            self._private_note(seat, {"event": "retired", "reason": str(exc)})
            return

        action = data.get("action") if isinstance(data, dict) else None
        if action == "work":
            try:
                written = (self._apply_files(seat, data["files"])
                           if "files" in data else [])
            except ValueError as exc:
                stats["violations"] += 1
                self._private_note(seat, {"event": "violation", "detail": str(exc)})
                return
            stats["works"] += 1
            self._private_note(seat, {"event": "work",
                                      "note": str(data.get("note", "")),
                                      **({"files_written": written} if written else {})})
        elif action == "post":
            try:
                entry = self.post(seat, data.get("message"))
            except ValueError as exc:
                stats["violations"] += 1
                self._private_note(seat, {"event": "violation", "detail": str(exc)})
                return
            stats["posts"] += 1
            self._private_note(seat, {"event": "post", "seq": entry["seq"]})
        elif action == "done":
            stats["done"] = True
            stats["done_reason"] = "reported done"
            stats["report"] = str(data.get("report", ""))
            self._private_note(seat, {"event": "done", "report": stats["report"]})
        else:
            stats["violations"] += 1
            self._private_note(seat, {
                "event": "violation",
                "detail": f"reply was not one of {ACTIONS}: {json.dumps(data)[:200]}",
            })

    # -- the episode ---------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Round-robin every live seat until all are done or the round budget
        runs out. Returns the episode record WITHOUT a verdict -- the caller
        grades the workspace and adds it (agents' claims decide nothing, and
        neither does this loop)."""
        rounds_used = 0
        termination = "max_rounds"
        for round_no in range(1, self.max_rounds + 1):
            rounds_used = round_no
            for seat in self.seats:
                if not self._stats[seat]["done"]:
                    self._turn(seat, round_no)
            if all(s["done"] for s in self._stats.values()):
                termination = "all_done"
                break
        return {
            "task_id": self.task_id,
            "workspace": str(self.workspace),
            "max_rounds": self.max_rounds,
            "rounds_used": rounds_used,
            "termination_reason": termination,
            "board_messages": self._seq,
            "board": self.board(),
            "seats": self._stats,
            "at": utcnow(),
        }

    def close(self) -> None:
        (self.root / "CLOSED").write_text("episode complete\n", encoding="utf-8")

    def save_episode(self, episode: dict[str, Any]) -> Path:
        path = self.root / "episode.json"
        path.write_text(json.dumps(episode, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        return path


def parse_seat_names(spec: str, default_n: int = 3) -> list[str]:
    """`--seats 4` -> agent1..agent4; `--seats alice,bob` -> named seats."""
    spec = (spec or "").strip()
    if not spec:
        return [f"agent{i}" for i in range(1, default_n + 1)]
    if spec.isdigit():
        return [f"agent{i}" for i in range(1, int(spec) + 1)]
    return [s.strip() for s in spec.split(",") if s.strip()]
