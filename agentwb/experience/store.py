"""The experience store.

Every evaluated run becomes a reusable record:

    (situation, action, observation, outcome, evaluation, correction)

Failure data is never deleted because a later version succeeded -- the failures
are the point. They are what V2 retrieves against and what V3 optimizes on.

Storage is JSONL, append-only. Human corrections are appended as separate
records that reference the original; nothing overwrites a human label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from ..types import Task, Trajectory, utcnow


class ExperienceStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "experience.jsonl"
        self.corrections_path = self.root / "corrections.jsonl"

    # -- writing ----------------------------------------------------------
    def record(self, task: Task, traj: Trajectory) -> dict[str, Any]:
        ev = traj.evaluation
        outcome = "PASS" if (ev and ev.passed) else "FAIL"
        entry = {
            "experience_id": f"exp_{traj.trajectory_id}",
            "recorded_at": utcnow(),
            "outcome": outcome,
            "task": {
                "id": task.id,
                "prompt": task.prompt,
                "success_criteria": task.success_criteria,
                "tags": task.tags,
            },
            "situation": {
                "provider": traj.provider,
                "model": traj.model,
                "prompt_version": traj.prompt_version,
                "trial": traj.trial,
            },
            "actions": [
                {
                    "step": s.step,
                    "tool": (s.action or {}).get("tool"),
                    "arguments": (s.action or {}).get("arguments"),
                    "ok": (s.tool_result or {}).get("ok"),
                }
                for s in traj.steps
                if (s.action or {}).get("type") == "tool_call"
            ],
            "final_output": traj.final_output,
            "environment_outcome": {
                "file_count": (traj.environment_outcome or {}).get("file_count"),
            },
            "evaluation": ev.to_dict() if ev else None,
            "failure_categories": traj.failure_categories,
            "metrics": traj.metrics,
            "trajectory_id": traj.trajectory_id,
            "correction": None,
        }
        self._append(self.path, entry)
        return entry

    def add_correction(
        self,
        trajectory_id: str,
        author: str,
        verdict: Optional[str] = None,
        categories: Optional[list[str]] = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record a human correction. Appended, never merged over the original.

        Optimizers must not silently overwrite these (spec section 29), which is
        why they live in their own file with explicit provenance.
        """
        entry = {
            "correction_id": f"corr_{trajectory_id}_{int(len(list(self.iter_corrections())))}",
            "trajectory_id": trajectory_id,
            "author": author,
            "at": utcnow(),
            "human_verdict": verdict,
            "reclassified_categories": categories or [],
            "note": note,
        }
        self._append(self.corrections_path, entry)
        return entry

    # -- reading ----------------------------------------------------------
    def iter_experiences(self) -> Iterator[dict[str, Any]]:
        yield from self._iter(self.path)

    def iter_corrections(self) -> Iterator[dict[str, Any]]:
        yield from self._iter(self.corrections_path)

    def failures(self, limit: int = 50, task_id: Optional[str] = None,
                 category: Optional[str] = None) -> list[dict[str, Any]]:
        out = []
        for e in self.iter_experiences():
            if e.get("outcome") != "FAIL":
                continue
            if task_id and e.get("task", {}).get("id") != task_id:
                continue
            if category and category not in (e.get("failure_categories") or []):
                continue
            out.append(e)
        return out[-limit:][::-1]

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.iter_experiences():
            for c in e.get("failure_categories") or []:
                counts[c] = counts.get(c, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # -- internals --------------------------------------------------------
    @staticmethod
    def _append(path: Path, entry: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _iter(path: Path) -> Iterator[dict[str, Any]]:
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
