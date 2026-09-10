"""Adversarial search (spec section 32).

    "Eventually allow dedicated simulation/search processes to actively
     discover difficult failure cases instead of waiting for production
     failures."

Every other part of this workbench waits for failure: you run a suite, some
tasks fail, you analyse them. This goes looking. It mutates a task that
currently passes into variants designed to break the agent, runs them, and
keeps the ones that do.

**The rule that makes this useful rather than noise: a variant must stay
solvable.** It is trivial to break any agent -- delete the file it needs, make
the tests contradict each other, ask for something impossible. Those failures
teach nothing and, worse, they pollute the regression suite with tasks that can
never go green, which is how a suite stops being believed.

So every mutation here preserves the task's success criteria and its graders,
and changes only the *route*: more distractors, ambiguous anchors, misleading
names, truncation pressure. A found failure then means "the agent could have
solved this and did not", which is a finding. Variants are additionally
screened by a solvability check before they are kept.

Mutations are deterministic and named. A search that produced unreproducible
variants would generate failures nobody could investigate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..types import Task


class Mutation(str, enum.Enum):
    """Named, deterministic ways to make a task harder without making it unfair."""

    ADD_DISTRACTOR_FILES = "ADD_DISTRACTOR_FILES"
    DUPLICATE_ANCHOR = "DUPLICATE_ANCHOR"
    MISLEADING_NAMES = "MISLEADING_NAMES"
    VERBOSE_NOISE = "VERBOSE_NOISE"
    WEAKEN_PROMPT = "WEAKEN_PROMPT"
    TIGHTEN_BUDGET = "TIGHTEN_BUDGET"


@dataclass
class Variant:
    task: Task
    mutation: Mutation
    rationale: str
    parent_task_id: str = ""
    solvable: Optional[bool] = None
    broke_the_agent: Optional[bool] = None
    trajectory_id: str = ""

    @property
    def is_finding(self) -> bool:
        """A finding is a variant the agent *could* have solved and did not."""
        return bool(self.solvable) and bool(self.broke_the_agent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "parent_task_id": self.parent_task_id,
            "mutation": self.mutation.value,
            "rationale": self.rationale,
            "solvable": self.solvable,
            "broke_the_agent": self.broke_the_agent,
            "is_finding": self.is_finding,
            "trajectory_id": self.trajectory_id,
        }


# --------------------------------------------------------------------------
# mutations
# --------------------------------------------------------------------------

DISTRACTOR_FILES = {
    "utils/helpers.py": '"""Assorted helpers."""\n\n\ndef divide(a, b):\n'
                        '    """Unrelated helper with a familiar name."""\n'
                        '    return a / b if b else None\n',
    "legacy/calculator.py": '"""Superseded; kept for reference."""\n\n\n'
                            'def divide(a, b):\n    if b == 0:\n        return 0\n'
                            '    return a / b\n',
    "docs/NOTES.md": "# Notes\n\nThe divide() behaviour was discussed at length "
                     "and never settled.\n",
}


def mutate(task: Task, mutation: Mutation) -> Optional[Variant]:
    """Produce one harder variant, or None when the mutation does not apply.

    Success criteria and graders are carried through untouched. Only the route
    changes -- otherwise a 'failure' would just mean the goalposts moved.
    """
    files = dict(task.environment.files)
    prompt = task.prompt
    max_steps = task.max_steps
    rationale = ""

    if mutation is Mutation.ADD_DISTRACTOR_FILES:
        if not files:
            return None
        files.update(DISTRACTOR_FILES)
        rationale = ("three plausible files, two defining a function with the same "
                     "name as the target -- tests whether the agent verifies which "
                     "file it is editing")

    elif mutation is Mutation.DUPLICATE_ANCHOR:
        target = _first_python_file(files)
        if not target:
            return None
        content = files[target]
        if "return 0" not in content:
            return None
        # A second identical line makes a naive edit anchor ambiguous, which
        # should surface as AMBIGUOUS_MATCH rather than a wrong edit.
        files[target] = content.replace(
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a + b\n\n\ndef reset(counter):\n"
            "    if counter is None:\n        return 0\n    return counter",
        )
        if files[target] == content:
            return None
        rationale = ("a second `return 0` elsewhere in the file -- a naive edit "
                     "anchor is now ambiguous and must be disambiguated")

    elif mutation is Mutation.MISLEADING_NAMES:
        target = _first_python_file(files)
        if not target:
            return None
        files[f"{target.rsplit('.', 1)[0]}_v2.py"] = (
            '"""Looks like the newer version. It is not imported by the tests."""\n\n\n'
            "def divide(a, b):\n    if b == 0:\n"
            '        raise ValueError("division by zero")\n    return a / b\n'
        )
        rationale = ("a file that looks newer and already contains the fix, but is "
                     "not the one under test -- editing it changes nothing")

    elif mutation is Mutation.VERBOSE_NOISE:
        target = _first_python_file(files)
        if not target:
            return None
        padding = "\n".join(f"# note {i}: historical detail, not relevant"
                            for i in range(1, 220))
        files[target] = f'"""Module with a long preamble."""\n{padding}\n\n' + files[target]
        rationale = ("~220 lines of preamble -- the relevant code no longer fits in "
                     "one bounded observation, so the agent must page or search")

    elif mutation is Mutation.WEAKEN_PROMPT:
        prompt = "Something in this workspace is wrong. Find it and fix it."
        rationale = ("the prompt no longer names the file or the symptom -- tests "
                     "whether the agent investigates rather than pattern-matches")

    elif mutation is Mutation.TIGHTEN_BUDGET:
        if task.max_steps <= 4:
            return None
        max_steps = max(3, task.max_steps // 2)
        rationale = (f"step budget halved to {max_steps} -- tests whether the agent "
                     f"prioritises verification when it cannot do everything")

    else:  # pragma: no cover - enum is exhaustive
        return None

    variant_task = Task(
        id=f"{task.id}__{mutation.value.lower()}",
        prompt=prompt,
        success_criteria=task.success_criteria,      # unchanged, deliberately
        environment=type(task.environment)(
            files=files,
            template_dir=task.environment.template_dir,
            setup_commands=list(task.environment.setup_commands),
        ),
        graders=list(task.graders),                  # unchanged, deliberately
        tools=task.tools,
        max_steps=max_steps,
        trials=task.trials,
        tags=sorted(set(task.tags) | {"adversarial"}),
        source_path=task.source_path,
    )
    return Variant(task=variant_task, mutation=mutation, rationale=rationale,
                   parent_task_id=task.id)


def _first_python_file(files: dict[str, str]) -> Optional[str]:
    for name in sorted(files):
        if name.endswith(".py") and not name.startswith("test_"):
            return name
    return None


def generate_variants(task: Task,
                      mutations: Optional[Iterable[Mutation]] = None) -> list[Variant]:
    """Every applicable mutation of a task, in a deterministic order."""
    out = []
    for mutation in (mutations or list(Mutation)):
        variant = mutate(task, mutation)
        if variant is not None:
            out.append(variant)
    return out


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

@dataclass
class SearchResult:
    parent_task_id: str
    variants: list[Variant] = field(default_factory=list)
    skipped_unsolvable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def findings(self) -> list[Variant]:
        return [v for v in self.variants if v.is_finding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_task_id": self.parent_task_id,
            "variants": [v.to_dict() for v in self.variants],
            "findings": [v.task.id for v in self.findings],
            "skipped_unsolvable": self.skipped_unsolvable,
            "notes": self.notes,
        }

    def summary(self) -> str:
        return (f"{len(self.findings)} finding(s) from {len(self.variants)} variant(s)"
                + (f"; {len(self.skipped_unsolvable)} discarded as unsolvable"
                   if self.skipped_unsolvable else ""))


def search(
    task: Task,
    run: Callable[[Task], bool],
    check_solvable: Optional[Callable[[Task], bool]] = None,
    mutations: Optional[Iterable[Mutation]] = None,
) -> SearchResult:
    """Hunt for variants that break the agent but remain fair.

    ``run(task) -> passed`` executes the agent under test.
    ``check_solvable(task) -> bool`` establishes the variant is winnable at all --
    a reference solver, or a run with a stronger configuration. Without it,
    variants are kept but flagged: an unscreened failure might be a discovery or
    might be an impossible task, and the difference matters.
    """
    result = SearchResult(parent_task_id=task.id)

    for variant in generate_variants(task, mutations):
        if check_solvable is not None:
            variant.solvable = bool(check_solvable(variant.task))
            if not variant.solvable:
                # Breaking an agent with an impossible task is not a finding,
                # and keeping it would poison the regression suite with a task
                # that can never go green.
                result.skipped_unsolvable.append(variant.task.id)
                continue
        else:
            variant.solvable = True
            result.notes.append(
                f"{variant.task.id}: kept without a solvability check -- treat a "
                f"failure here as a lead, not a confirmed weakness"
            )

        variant.broke_the_agent = not bool(run(variant.task))
        result.variants.append(variant)

    return result
