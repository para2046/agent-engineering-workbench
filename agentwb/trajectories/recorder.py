"""Append-only trajectory recording.

Steps are flushed as they happen, so a crashed or killed run still leaves a
readable partial trajectory on disk. The final `close` writes the complete
document and updates the SQLite index.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Step, Trajectory
from .store import TrajectoryStore


class TrajectoryRecorder:
    def __init__(self, store: TrajectoryStore):
        self.store = store

    def open(self, traj: Trajectory) -> None:
        self.store.run_dir(traj.trajectory_id).mkdir(parents=True, exist_ok=True)
        self._events(traj).write_text("", encoding="utf-8")
        self._append(traj, {"event": "run_started", "trajectory_id": traj.trajectory_id,
                            "task_id": traj.task_id, "model": traj.model,
                            "provider": traj.provider, "at": traj.started_at})

    def step(self, traj: Trajectory, step: Step) -> None:
        self._append(traj, {"event": "step", **step.to_dict()})

    def close(self, traj: Trajectory) -> Path:
        self._append(traj, {"event": "run_ended", "termination_reason": traj.termination_reason,
                            "at": traj.ended_at})
        return self.store.write(traj)

    # -- internals --------------------------------------------------------
    def _events(self, traj: Trajectory) -> Path:
        return self.store.run_dir(traj.trajectory_id) / "events.jsonl"

    def _append(self, traj: Trajectory, payload: dict) -> None:
        with self._events(traj).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
