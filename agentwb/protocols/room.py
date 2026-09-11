"""A room: protocol seats that external agents can occupy.

The orchestrator normally talks to in-process providers. A room turns each
seat into a small file-based mailbox instead, so ANY external process -- a
Claude Code subagent, another CLI model, a human with a text editor -- can sit
in it and discuss with the others through the real protocol:

    room/
      <seat>/prompt.json     <- the seat's current turn: system + context
      <seat>/reply.json      <- the occupant writes ONE protocol message here
      exchange.json          <- the finished exchange, same format as ever
      CLOSED                 <- written when the exchange ends

The occupant's contract is three sentences: wait for a prompt.json whose
`turn` you have not answered; write reply.json containing exactly one JSON
message in the format the prompt shows; repeat until CLOSED exists.

Everything else is unchanged, deliberately. Messages are schema-validated,
remits are enforced, disagreements go down the evidence-then-experiment
ladder, rounds must earn their existence, and the exchange is graded and
visualized like any other. The room adds participation, not exemptions -- a
subagent in a seat is subject to exactly the rules an in-process model is.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from ..judge.client import JudgeError


class RoomSeat:
    """The orchestrator-facing side of one seat. Quacks like a JudgeClient."""

    def __init__(self, room: Path, seat: str, timeout: float = 300.0,
                 poll_interval: float = 1.0):
        self.dir = Path(room) / seat
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seat = seat
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._turn = 0
        self.model = f"room/{seat}"
        self.provider = self          # JudgeClient-shaped surface

    # -- what the orchestrator calls ---------------------------------------
    def ask_json(self, system: str, user: str, prompt_id: str = "") -> Any:
        self._turn += 1
        prompt_path = self.dir / "prompt.json"
        reply_path = self.dir / "reply.json"
        reply_path.unlink(missing_ok=True)

        prompt_path.write_text(json.dumps({
            "turn": self._turn,
            "seat": self.seat,
            "system": system,
            "user": user,
            "instructions": (
                "Write reply.json in this directory containing exactly one JSON "
                "object in the message format shown in `system`. Then wait for "
                "the next prompt.json (its `turn` will increase) or for a CLOSED "
                "file in the room directory."
            ),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = self._try_read(reply_path)
            if data is not None:
                consumed = self.dir / f"reply.turn{self._turn}.json"
                try:
                    reply_path.replace(consumed)
                except OSError:
                    pass
                from types import SimpleNamespace
                return SimpleNamespace(data=data, raw_text=json.dumps(data),
                                       prompt_id=prompt_id, model=self.model)
            time.sleep(self.poll_interval)

        raise JudgeError(
            f"seat {self.seat!r} did not reply within {self.timeout:.0f}s -- "
            f"the occupant may have crashed or wandered off"
        )

    @staticmethod
    def _try_read(path: Path) -> Optional[dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            return None               # mid-write on the occupant's side
        if not text.strip():
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None               # partially written; poll again
        return data if isinstance(data, dict) else {"text": str(data)}


def open_room(room: Path, seats: list[str], timeout: float = 300.0) -> dict[str, RoomSeat]:
    room = Path(room)
    room.mkdir(parents=True, exist_ok=True)
    (room / "CLOSED").unlink(missing_ok=True)
    return {seat: RoomSeat(room, seat, timeout=timeout) for seat in seats}


def close_room(room: Path) -> None:
    (Path(room) / "CLOSED").write_text("exchange complete\n", encoding="utf-8")
