"""Host a discussion room where external agents occupy the protocol seats.

    python drive_room.py rooms/demo "Should trajectories live in JSONL or SQLite?" \
        --seats researcher,engineer --max-rounds 2 --timeout 300

This process is the host: it runs the real Orchestrator, but each seat is a
file mailbox (see agentwb/protocols/room.py). Any external agent -- a Claude
Code subagent, another model CLI, a human -- occupies a seat by watching
<room>/<seat>/prompt.json and writing <room>/<seat>/reply.json.

The protocol is not relaxed for guests: messages are schema-validated, remits
enforced, disagreements resolved by evidence or experiment, rounds must earn
their existence. The finished exchange lands in <room>/exchange.json, ready
for `demo/viz.py` or `demo/flow.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentwb.protocols.room import close_room, open_room          # noqa: E402
from agentwb.runtime.orchestrator import Orchestrator, save_exchange  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("room", help="room directory (created if missing)")
    ap.add_argument("topic", help="what the seats are asked to work out")
    ap.add_argument("--seats", default="researcher,engineer",
                    help="comma-separated seat names (default researcher,engineer)")
    ap.add_argument("--criteria", default="", help="success criteria shown to both seats")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds to wait for each seat's reply")
    args = ap.parse_args()

    seats = [s.strip() for s in args.seats.split(",") if s.strip()]
    room = Path(args.room)
    agents = open_room(room, seats, timeout=args.timeout)
    print(f"room open at {room}  seats: {', '.join(seats)}")
    print("waiting for occupants -- each seat answers <seat>/prompt.json "
          "with <seat>/reply.json\n")

    try:
        exchange = Orchestrator(agents, max_rounds=args.max_rounds).run(
            task_id=room.name, task_prompt=args.topic, success_criteria=args.criteria)
    finally:
        close_room(room)

    path = save_exchange(exchange, room / "exchange.json")
    print(exchange.transcript())
    print(f"\ntermination: {exchange.termination_reason}")
    print(f"metrics: {exchange.metrics}")
    print(f"exchange saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
