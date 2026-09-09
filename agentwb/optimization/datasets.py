"""Dataset splits for optimization (spec section 11).

    train / development / test / regression

The spec states the rule plainly: **never optimize directly against the final
test set.** This module is where that stops being an intention and becomes a
mechanism.

Two properties do the work:

*Assignment is deterministic.* A task lands in a split by a hash of its id, so
the same task is in the same split on every machine, every run, forever. A
split that shuffles between runs is not a held-out set — it is a slow leak.

*The test split is not handed out casually.* ``for_optimizer()`` returns train
and dev and will raise if asked for test. Reading the test set requires
``for_final_evaluation()``, which is a different call with a name you cannot
type by accident.

Explicit `split` tags on a task always win over hashing, so a curated
regression suite stays curated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..types import Task

TRAIN, DEV, TEST, REGRESSION = "train", "dev", "test", "regression"
SPLITS = (TRAIN, DEV, TEST, REGRESSION)

# Tags that pin a task to a split explicitly.
SPLIT_TAGS = {
    "train": TRAIN,
    "dev": DEV, "development": DEV,
    "test": TEST, "holdout": TEST, "held-out": TEST, "held_out": TEST, "benchmark": TEST,
    "regression": REGRESSION,
}

DEFAULT_RATIOS = {TRAIN: 0.6, DEV: 0.2, TEST: 0.2}


class ContaminationError(RuntimeError):
    """Raised when something tries to reach the held-out set through a door
    that is not meant to open onto it."""


@dataclass
class Dataset:
    train: list[Task] = field(default_factory=list)
    dev: list[Task] = field(default_factory=list)
    test: list[Task] = field(default_factory=list)
    regression: list[Task] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.train) + len(self.dev) + len(self.test) + len(self.regression)

    def counts(self) -> dict[str, int]:
        return {TRAIN: len(self.train), DEV: len(self.dev),
                TEST: len(self.test), REGRESSION: len(self.regression)}

    def split(self, name: str) -> list[Task]:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r} (expected one of {', '.join(SPLITS)})")
        return getattr(self, name)

    # -- controlled access ------------------------------------------------
    def for_optimizer(self, name: str = TRAIN) -> list[Task]:
        """Splits an optimizer is allowed to see: train and dev only.

        Asking this for the test set is a mistake worth failing loudly on --
        by the time a leaked benchmark shows up as an unexplained score, the
        evidence of how it happened is long gone.
        """
        if name in (TEST, REGRESSION):
            raise ContaminationError(
                f"the {name!r} split is not available to an optimizer -- optimizing "
                f"against it destroys its value as held-out evidence. Use "
                f"for_final_evaluation() once, after the candidate is chosen."
            )
        return self.split(name)

    def for_final_evaluation(self) -> list[Task]:
        """The held-out test set. Read once, after optimization has finished."""
        return self.test

    def task_ids(self, name: str) -> set[str]:
        return {t.id for t in self.split(name)}

    def assert_disjoint(self) -> None:
        """No task may appear in two splits. Cheap to check, expensive to miss."""
        seen: dict[str, str] = {}
        for name in SPLITS:
            for task in self.split(name):
                if task.id in seen:
                    raise ContaminationError(
                        f"task {task.id!r} is in both {seen[task.id]!r} and {name!r} splits"
                    )
                seen[task.id] = name


def assign_split(task: Task, ratios: Optional[dict[str, float]] = None) -> str:
    """Which split a task belongs to.

    An explicit tag wins. Otherwise a hash of the task id decides -- stable
    across machines and runs, which a random shuffle is not.
    """
    for tag in task.tags:
        mapped = SPLIT_TAGS.get(str(tag).lower())
        if mapped:
            return mapped

    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.get(s, 0.0) for s in (TRAIN, DEV, TEST))
    if total <= 0:
        return TRAIN

    digest = hashlib.sha256(task.id.encode("utf-8")).hexdigest()
    position = (int(digest[:8], 16) / 0xFFFFFFFF) * total

    cumulative = 0.0
    for name in (TRAIN, DEV, TEST):
        cumulative += ratios.get(name, 0.0)
        if position < cumulative:
            return name
    return TEST


def build_dataset(tasks: Iterable[Task], ratios: Optional[dict[str, float]] = None) -> Dataset:
    """Split a task collection. A regression-tagged task goes only to
    regression -- a curated suite is not sampling material."""
    ds = Dataset()
    for task in tasks:
        getattr(ds, assign_split(task, ratios)).append(task)
    ds.assert_disjoint()
    return ds
