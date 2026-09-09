"""Local-first storage: JSONL documents plus a SQLite index.

JSONL is the source of truth -- human-readable, greppable, diffable, and
trivially portable. SQLite exists only as an index so `failures --last 50` and
`compare` do not have to walk every run directory. Deleting the database is
always safe: `reindex()` rebuilds it from the JSONL files.

No vector database. Not yet, and not until retrieval volume justifies one
(spec section 22).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Optional

from ..types import Trajectory

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    trajectory_id     TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    provider          TEXT,
    model             TEXT,
    prompt_version    TEXT,
    trial             INTEGER,
    started_at        TEXT,
    ended_at          TEXT,
    passed            INTEGER,
    score             REAL,
    termination_reason TEXT,
    steps             INTEGER,
    tool_calls        INTEGER,
    tool_failures     INTEGER,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    latency_ms        INTEGER,
    failure_categories TEXT,
    path              TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_passed ON runs(passed);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""


class TrajectoryStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs_root = self.root / "trajectories"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite3"
        self._db: Optional[sqlite3.Connection] = None
        self._init_db()

    # -- paths ------------------------------------------------------------
    def run_dir(self, trajectory_id: str) -> Path:
        return self.runs_root / trajectory_id

    def doc_path(self, trajectory_id: str) -> Path:
        return self.run_dir(trajectory_id) / "trajectory.json"

    # -- db ---------------------------------------------------------------
    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _init_db(self) -> None:
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    # -- writing ----------------------------------------------------------
    def write(self, traj: Trajectory) -> Path:
        d = self.run_dir(traj.trajectory_id)
        d.mkdir(parents=True, exist_ok=True)
        path = self.doc_path(traj.trajectory_id)
        path.write_text(json.dumps(traj.to_dict(), indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        self._index(traj, path)
        return path

    def _index(self, traj: Trajectory, path: Path) -> None:
        ev = traj.evaluation
        m = traj.metrics or {}
        self.db.execute(
            """INSERT OR REPLACE INTO runs VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                traj.trajectory_id, traj.task_id, traj.provider, traj.model,
                traj.prompt_version, traj.trial, traj.started_at, traj.ended_at,
                1 if (ev and ev.passed) else 0,
                float(ev.score) if ev else None,
                traj.termination_reason,
                m.get("steps"), m.get("tool_calls"), m.get("tool_failures"),
                m.get("input_tokens"), m.get("output_tokens"), m.get("latency_ms"),
                ",".join(traj.failure_categories),
                str(path),
            ),
        )
        self.db.commit()

    # -- reading ----------------------------------------------------------
    def load(self, trajectory_id: str) -> Trajectory:
        path = self.doc_path(trajectory_id)
        if not path.is_file():
            raise FileNotFoundError(f"no trajectory {trajectory_id!r} under {self.runs_root}")
        return Trajectory.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, prefix: str) -> str:
        """Accept an unambiguous id prefix, so you can type `run_2026` not the full id."""
        if self.doc_path(prefix).is_file():
            return prefix
        matches = [p.name for p in sorted(self.runs_root.iterdir())
                   if p.is_dir() and p.name.startswith(prefix)]
        if not matches:
            raise FileNotFoundError(f"no run matching {prefix!r}")
        if len(matches) > 1:
            raise ValueError(f"{prefix!r} is ambiguous: {', '.join(matches[:6])}")
        return matches[0]

    def query(self, where: str = "", params: tuple = (), limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY started_at DESC LIMIT ?"
        rows = self.db.execute(sql, (*params, limit)).fetchall()
        return [dict(r) for r in rows]

    def iter_docs(self) -> Iterator[Trajectory]:
        for d in sorted(self.runs_root.iterdir()):
            doc = d / "trajectory.json"
            if doc.is_file():
                try:
                    yield Trajectory.from_dict(json.loads(doc.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, KeyError):
                    continue

    def reindex(self) -> int:
        """Rebuild the SQLite index from JSONL/JSON documents on disk."""
        self.db.execute("DELETE FROM runs")
        n = 0
        for traj in self.iter_docs():
            self._index(traj, self.doc_path(traj.trajectory_id))
            n += 1
        self.db.commit()
        return n
