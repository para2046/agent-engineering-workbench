"""Experience retrieval (spec section 20).

    current task -> retrieve relevant experiences -> select a small diverse set -> context

The temptation with a store full of past runs is to shovel it into the prompt.
That is what this module exists to prevent. Three constraints shape every
decision here:

**Small.** Default three experiences, each compressed to the approach taken and
the outcome. Not transcripts. A retriever that returns everything relevant has
done nothing useful.

**Diverse.** Greedy relevance alone returns five near-identical runs of the same
task. Selection penalises candidates that resemble what is already picked, so
the set spans approaches rather than repeating one.

**Uncontaminated.** This is the constraint that actually matters, and it is not
a preference. Retrieving a past run *of the task now being attempted* hands the
agent the answer and turns the eval into a lookup. Same-task retrieval is
refused unconditionally, and anything tagged held-out is excluded as a source.
An eval you have quietly leaked into cannot be un-leaked -- every score after
that is fiction.

No vector database (spec section 22). Lexical scoring over a local JSONL store
is adequate at this volume and has no infrastructure to keep alive; swap the
scorer when retrieval volume actually justifies it, not before.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..types import Task

# Tags that mark an experience as ineligible as a retrieval *source*.
HELD_OUT_TAGS = frozenset({"holdout", "held-out", "held_out", "test", "benchmark"})

# Ignored when scoring: present in nearly every task prompt, so they carry no
# signal and quietly inflate every similarity score.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be been being
do does did doing to of in on at by for with from as it its into out up down over under
you your we our they their he she his her i me my not no so such can will would should
could may might must have has had having make makes made use used using run runs
please make sure need needs want wants task file files code
""".split())

TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass
class RetrievedExperience:
    experience_id: str
    trajectory_id: str
    task_id: str
    outcome: str                     # PASS | FAIL | CORRECTED
    score: float
    approach: list[str] = field(default_factory=list)
    lesson: str = ""
    corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "outcome": self.outcome,
            "score": round(self.score, 4),
            "corrected": self.corrected,
        }


@dataclass
class RetrievalResult:
    items: list[RetrievedExperience] = field(default_factory=list)
    considered: int = 0
    excluded_same_task: int = 0
    excluded_held_out: int = 0
    query_terms: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> dict[str, Any]:
        """Recorded in the trajectory so any run can be audited for what it was
        shown -- retrieval that leaves no trace is unfalsifiable."""
        return {
            "retrieved": [i.to_dict() for i in self.items],
            "considered": self.considered,
            "excluded_same_task": self.excluded_same_task,
            "excluded_held_out": self.excluded_held_out,
            "query_terms": self.query_terms[:20],
        }

    def as_context(self) -> str:
        """Render for injection. Empty string when nothing was retrieved, so the
        caller can skip the section entirely rather than showing a bare heading."""
        if not self.items:
            return ""
        lines = [
            "RELEVANT PAST EXPERIENCE (from earlier runs on *different* tasks).",
            "Use it as a hint about approach. It is not a template, it may not "
            "apply here, and it is never evidence about the task in front of you.",
            "",
        ]
        for n, item in enumerate(self.items, 1):
            verdict = "succeeded" if item.outcome == "PASS" else (
                "was corrected by a human" if item.corrected else "failed")
            lines.append(f"{n}. task `{item.task_id}` -- {verdict}")
            if item.approach:
                lines.append(f"   approach: {' -> '.join(item.approach)}")
            if item.lesson:
                lines.append(f"   note: {item.lesson}")
        return "\n".join(lines)


class ExperienceRetriever:
    def __init__(
        self,
        store,
        k: int = 3,
        min_score: float = 0.05,
        diversity: float = 0.5,
        prefer_successful: bool = True,
    ):
        """
        k             how many experiences to inject (keep it small)
        min_score     below this, an experience is not relevant enough to show
        diversity     0 = pure relevance, 1 = maximum spread across approaches
        prefer_successful  boost PASS and human-corrected experiences
        """
        self.store = store
        self.k = k
        self.min_score = min_score
        self.diversity = diversity
        self.prefer_successful = prefer_successful

    # -- public ----------------------------------------------------------
    def retrieve(self, task: Task, k: Optional[int] = None) -> RetrievalResult:
        k = self.k if k is None else k
        query = _terms(f"{task.prompt} {task.success_criteria} {' '.join(task.tags)}")
        result = RetrievalResult(query_terms=sorted(query))
        if not query or k <= 0:
            return result

        corrected_ids = self._corrected_trajectory_ids()
        candidates: list[tuple[float, RetrievedExperience, set[str]]] = []

        for entry in self.store.iter_experiences():
            result.considered += 1
            entry_task = (entry.get("task") or {})
            entry_task_id = entry_task.get("id", "")

            # Hard exclusions. Neither is a tunable preference.
            if entry_task_id == task.id:
                result.excluded_same_task += 1
                continue
            if HELD_OUT_TAGS & {str(t).lower() for t in (entry_task.get("tags") or [])}:
                result.excluded_held_out += 1
                continue

            terms = _terms(f"{entry_task.get('prompt', '')} "
                           f"{' '.join(entry_task.get('tags') or [])}")
            if not terms:
                continue

            score = _similarity(query, terms)
            traj_id = entry.get("trajectory_id", "")
            corrected = traj_id in corrected_ids
            outcome = entry.get("outcome", "FAIL")

            if self.prefer_successful:
                if outcome == "PASS":
                    score *= 1.25
                elif corrected:
                    score *= 1.15   # a corrected failure carries the fix
                else:
                    score *= 0.85   # failures still teach, but rank below

            if score < self.min_score:
                continue

            candidates.append((
                score,
                RetrievedExperience(
                    experience_id=entry.get("experience_id", ""),
                    trajectory_id=traj_id,
                    task_id=entry_task_id,
                    outcome="CORRECTED" if corrected else outcome,
                    score=score,
                    approach=_approach(entry),
                    lesson=_lesson(entry),
                    corrected=corrected,
                ),
                terms,
            ))

        result.items = self._select_diverse(candidates, k)
        return result

    # -- internals -------------------------------------------------------
    def _select_diverse(self, candidates, k: int) -> list[RetrievedExperience]:
        """Maximal-marginal-relevance selection.

        Pure top-k returns the same run five times when a task has been retried.
        Each pick is scored on relevance minus its similarity to what is already
        chosen, so the set covers different approaches.
        """
        remaining = sorted(candidates, key=lambda c: -c[0])
        chosen: list[RetrievedExperience] = []
        chosen_terms: list[set[str]] = []

        while remaining and len(chosen) < k:
            best_i, best_value = 0, -math.inf
            for i, (score, item, terms) in enumerate(remaining):
                redundancy = max((_similarity(terms, t) for t in chosen_terms), default=0.0)
                value = (1 - self.diversity) * score - self.diversity * redundancy
                if value > best_value:
                    best_i, best_value = i, value
            score, item, terms = remaining.pop(best_i)
            chosen.append(item)
            chosen_terms.append(terms)
        return chosen

    def _corrected_trajectory_ids(self) -> set[str]:
        try:
            return {c.get("trajectory_id", "") for c in self.store.iter_corrections()}
        except AttributeError:
            return set()


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------

def _terms(text: str) -> set[str]:
    return {t for t in TOKEN.findall((text or "").lower())
            if len(t) > 2 and t not in STOPWORDS}


def _similarity(a: set[str], b: set[str]) -> float:
    """Jaccard overlap. Crude, transparent, and good enough at this volume --
    and unlike an embedding it can be explained to whoever asks why a given
    experience was shown."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _approach(entry: dict[str, Any]) -> list[str]:
    """The sequence of tools used, de-duplicated consecutively.

    Deliberately the tools and not their arguments: arguments carry file paths
    and literal edits from the other task, which is noise here at best and a
    leak at worst.
    """
    tools: list[str] = []
    for action in (entry.get("actions") or []):
        name = action.get("tool")
        if name and (not tools or tools[-1] != name):
            tools.append(name)
    return tools[:8]


def _lesson(entry: dict[str, Any]) -> str:
    """One line on why it failed, drawn from the recorded categories."""
    cats = entry.get("failure_categories") or []
    if entry.get("outcome") == "PASS" or not cats:
        return ""
    return f"previously failed with {', '.join(cats[:3])}"
