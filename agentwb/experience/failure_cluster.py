"""Grouping failures that are the same failure (spec sections 9 and 10).

Fifty failed runs are rarely fifty problems. They are usually three problems,
one of which happened forty times. A flat list hides that: the categories get
counted, the counts look alarming, and the actual shape -- *which* three -- is
invisible.

Clustering is on a **signature**, not similarity. A signature is the tuple of
observable facts that made the run fail: its categories, its termination
reason, the tool error codes it hit, and whether it verified anything. Two runs
with the same signature failed the same way, and that is a claim you can check
by reading them; two runs with a 0.83 similarity score is not.

The deliberate consequence: this never merges runs that merely look alike. It
under-clusters rather than over-clusters, because a false merge hides a real
second bug behind a first one, and that failure mode is silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Signature:
    """The observable shape of a failure."""

    categories: tuple[str, ...] = ()
    termination: str = ""
    error_codes: tuple[str, ...] = ()
    verified: bool = False
    modified_files: bool = False

    def describe(self) -> str:
        parts = []
        if self.categories:
            parts.append(", ".join(self.categories))
        if self.error_codes:
            parts.append(f"tool errors: {', '.join(self.error_codes)}")
        if self.termination:
            parts.append(f"ended {self.termination}")
        if not self.verified:
            parts.append("never verified")
        if not self.modified_files:
            parts.append("changed nothing")
        return "; ".join(parts) or "no distinguishing signal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": list(self.categories),
            "termination": self.termination,
            "error_codes": list(self.error_codes),
            "verified": self.verified,
            "modified_files": self.modified_files,
        }


@dataclass
class Cluster:
    signature: Signature
    experiences: list[dict[str, Any]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.experiences)

    @property
    def task_ids(self) -> list[str]:
        seen: list[str] = []
        for e in self.experiences:
            tid = (e.get("task") or {}).get("id", "")
            if tid and tid not in seen:
                seen.append(tid)
        return seen

    @property
    def spans_tasks(self) -> bool:
        """A cluster covering several tasks points at the agent or the tools;
        one confined to a single task usually points at that task."""
        return len(self.task_ids) > 1

    def example(self) -> Optional[dict[str, Any]]:
        return self.experiences[0] if self.experiences else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_dict(),
            "description": self.signature.describe(),
            "size": self.size,
            "task_ids": self.task_ids,
            "spans_tasks": self.spans_tasks,
            "trajectory_ids": [e.get("trajectory_id", "") for e in self.experiences],
        }


def signature_of(experience: dict[str, Any]) -> Signature:
    """Derive a failure's signature from what was recorded about it."""
    actions = experience.get("actions") or []
    tools = [a.get("tool") for a in actions]

    codes: set[str] = set()
    for action in actions:
        if action.get("ok") is False:
            code = action.get("error")
            if code:
                codes.add(str(code))

    metrics = experience.get("metrics") or {}
    termination = str(metrics.get("termination_reason")
                      or experience.get("termination_reason") or "")

    return Signature(
        categories=tuple(sorted(experience.get("failure_categories") or [])),
        termination=termination,
        error_codes=tuple(sorted(codes)),
        verified="run_tests" in tools,
        modified_files=any(t in {"edit_file", "write_file"} for t in tools),
    )


def cluster(experiences: Iterable[dict[str, Any]],
            failures_only: bool = True) -> list[Cluster]:
    """Group failures by signature, largest cluster first.

    Ties break on description so the ordering is stable across runs -- a report
    whose rows reshuffle between invocations is hard to trust and harder to diff.
    """
    groups: dict[Signature, Cluster] = {}
    for entry in experiences:
        if failures_only and entry.get("outcome") == "PASS":
            continue
        sig = signature_of(entry)
        groups.setdefault(sig, Cluster(signature=sig)).experiences.append(entry)

    return sorted(groups.values(), key=lambda c: (-c.size, c.signature.describe()))


def summarize(clusters: list[Cluster]) -> dict[str, Any]:
    """Headline numbers, including the one that matters: how concentrated the
    failures are. Thirty failures in two clusters is a very different morning
    from thirty failures in twenty-eight."""
    total = sum(c.size for c in clusters)
    largest = clusters[0].size if clusters else 0
    return {
        "failures": total,
        "clusters": len(clusters),
        "largest_cluster": largest,
        "concentration": round(largest / total, 3) if total else 0.0,
        "cross_task_clusters": sum(1 for c in clusters if c.spans_tasks),
    }
