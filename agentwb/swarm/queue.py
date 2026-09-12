"""A shared work queue for a swarm of agents (the coordination layer).

The room gave external agents *seats in a conversation*. This gives them
*claims on work*. Any number of workers -- Claude Code subagents, other CLIs,
humans -- can join a swarm at any time by polling one directory, claiming
tickets atomically, and submitting results. The host seeds the queue, gates
dependent work, re-queues abandoned claims, and grades every finished piece
with the real harness.

Layout on disk (one directory per swarm):

    swarm/
      open/<task_id>.json               a ticket anyone may claim
      claimed/<task_id>.<worker>.json   a claim; heartbeat via mtime
      done/<task_id>.<worker>.json      the worker's completion report
      graded/<task_id>.json             the HOST's verdict (workers never write here)
      blocked/<task_id>.json            waiting on dependencies
      CLOSED                            no more work will be added

Three properties carry the design:

**Claims are atomic, by construction.** A worker claims by renaming
``open/X.json`` to ``claimed/X.<worker>.json``. Rename either succeeds or the
file is already gone -- two workers racing get exactly one winner, with no
locks, no server, and no way to half-claim. This is the same trick that makes
maildir safe.

**Membership is dynamic because nobody tracks it.** There is no worker
registry to join or leave. A worker exists while it holds claims or polls the
directory; a crashed worker simply stops heartbeating and its claim is
re-queued after the lease expires. Scaling from one worker to twenty is
running nineteen more processes.

**Completion is a claim; the verdict is the host's.** A worker's ``done/``
report says what it believes it did. The host then runs the task's graders
against the actual workspace, exactly as for any single-agent run, and writes
``graded/``. A swarm of enthusiastic workers reporting success changes nothing
about what the graders find -- the property this whole workbench exists for,
held at swarm scale.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..types import utcnow

STATES = ("open", "claimed", "done", "graded", "blocked")


@dataclass
class Ticket:
    """One unit of claimable work."""

    task_id: str
    task_file: str                 # path to the task JSON (harness format)
    workspace: str                 # where the work happens; graders run here
    instructions: str = ""         # worker-facing note beyond the task prompt
    after: list[str] = field(default_factory=list)   # task_ids that must PASS first
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "task_file": self.task_file,
            "workspace": self.workspace, "instructions": self.instructions,
            "after": self.after, "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Ticket":
        return Ticket(
            task_id=d["task_id"], task_file=d.get("task_file", ""),
            workspace=d.get("workspace", ""), instructions=d.get("instructions", ""),
            after=list(d.get("after") or []), created_at=d.get("created_at", ""),
        )


class SwarmQueue:
    def __init__(self, root: Path, lease_seconds: float = 600.0):
        self.root = Path(root)
        self.lease_seconds = lease_seconds
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)

    # -- host side --------------------------------------------------------
    def seed(self, ticket: Ticket) -> Path:
        """Add work. Tickets with unmet dependencies wait in blocked/."""
        state = "blocked" if ticket.after else "open"
        path = self.root / state / f"{ticket.task_id}.json"
        path.write_text(json.dumps(ticket.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    def release_unblocked(self) -> list[str]:
        """Move blocked tickets whose prerequisites have all been graded PASS.

        Dependencies gate on the HOST's verdict, not on a worker's done-report:
        downstream work building on an unverified upstream claim is how a swarm
        compounds one agent's mistake into everyone's.
        """
        released = []
        for path in sorted((self.root / "blocked").glob("*.json")):
            ticket = Ticket.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if all(self.verdict_of(dep) is True for dep in ticket.after):
                target = self.root / "open" / path.name
                try:
                    path.rename(target)
                    released.append(ticket.task_id)
                except OSError:
                    continue
        return released

    def requeue_stale_claims(self) -> list[str]:
        """Return abandoned work to the pool.

        A claim whose file has not been touched within the lease is treated as
        a dead worker. Workers extend their lease by touching the claim file
        (`heartbeat`). Requeueing is also a rename, so a worker racing its own
        expiry cannot double-run: exactly one of the two moves wins.
        """
        requeued = []
        now = time.time()
        for path in sorted((self.root / "claimed").glob("*.json")):
            if now - path.stat().st_mtime <= self.lease_seconds:
                continue
            task_id = path.stem.split(".", 1)[0]
            target = self.root / "open" / f"{task_id}.json"
            try:
                path.rename(target)
                requeued.append(task_id)
            except OSError:
                continue
        return requeued

    def record_verdict(self, task_id: str, passed: bool,
                       detail: Optional[dict[str, Any]] = None) -> Path:
        payload = {"task_id": task_id, "passed": passed, "graded_at": utcnow(),
                   **(detail or {})}
        path = self.root / "graded" / f"{task_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        return path

    def verdict_of(self, task_id: str) -> Optional[bool]:
        path = self.root / "graded" / f"{task_id}.json"
        if not path.is_file():
            return None
        return bool(json.loads(path.read_text(encoding="utf-8")).get("passed"))

    def close(self) -> None:
        (self.root / "CLOSED").write_text("no further work\n", encoding="utf-8")

    @property
    def closed(self) -> bool:
        return (self.root / "CLOSED").exists()

    def status(self) -> dict[str, Any]:
        counts = {state: len(list((self.root / state).glob("*.json")))
                  for state in STATES}
        graded = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted((self.root / "graded").glob("*.json"))]
        counts["passed"] = sum(1 for g in graded if g.get("passed"))
        counts["failed"] = counts["graded"] - counts["passed"]
        return counts

    # -- worker side ------------------------------------------------------
    def claim(self, worker: str) -> Optional[Ticket]:
        """Atomically claim one open ticket, oldest first. None = nothing open.

        The rename IS the lock. Two workers racing for the same ticket both
        call rename; the filesystem lets exactly one succeed and hands the
        other an error we treat as "someone else got it, try the next".
        """
        for path in sorted((self.root / "open").glob("*.json"),
                           key=lambda p: p.stat().st_mtime):
            task_id = path.stem
            target = self.root / "claimed" / f"{task_id}.{worker}.json"
            try:
                path.rename(target)
            except OSError:
                continue                      # lost the race; next ticket
            return Ticket.from_dict(json.loads(target.read_text(encoding="utf-8")))
        return None

    def heartbeat(self, task_id: str, worker: str) -> bool:
        path = self.root / "claimed" / f"{task_id}.{worker}.json"
        if not path.is_file():
            return False                      # lease expired; work was requeued
        os.utime(path)
        return True

    def submit(self, task_id: str, worker: str, report: str,
               claims_success: bool = True) -> Path:
        """Hand the work back. This states a belief; the host's graders decide."""
        claim_path = self.root / "claimed" / f"{task_id}.{worker}.json"
        done_path = self.root / "done" / f"{task_id}.{worker}.json"
        payload = {"task_id": task_id, "worker": worker, "report": report,
                   "claims_success": claims_success, "submitted_at": utcnow()}
        done_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        claim_path.unlink(missing_ok=True)
        return done_path

    def pending_grades(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted((self.root / "done").glob("*.json")):
            task_id = path.stem.split(".", 1)[0]
            if self.verdict_of(task_id) is None:
                out.append(json.loads(path.read_text(encoding="utf-8")))
        return out
