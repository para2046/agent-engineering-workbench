"""Research mode (spec section 26).

    QUESTION -> candidate hypotheses -> evidence gathering -> rank hypotheses
             -> identify disagreement -> design discriminative experiment
             -> execute -> evaluate -> update belief

The workbench already had the second half: `protocols.disagreement` designs a
discriminating experiment and lets the environment decide. What was missing is
the first half -- generating several hypotheses at once and *ranking* them.

The ranking rule is the whole point, and it is deliberately not the obvious one.

**Hypotheses are ranked by evidence, never by stated confidence.** A model
asked how sure it is will answer fluently and the number will correlate with
how good the sentence sounded. Ranking on it means the best-written hypothesis
wins, which is how a research loop converges confidently on the wrong thing. So
the score here is built from countable facts: how many distinct observations
support it, whether those observations came from the environment or from
another claim, whether it names what would refute it, and whether it admits
what it does not know. Stated confidence enters only as a small tie-breaker,
and a claim whose confidence outruns its evidence is *penalised* for the gap.

**Belief updates require new evidence.** A round that gathered nothing new
cannot change the ranking, however much re-reasoning happens. `update_belief`
refuses to move on unchanged evidence, so the loop cannot talk itself into a
conclusion.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..judge.client import JudgeClient, JudgeError
from ..prompts import VersionedPrompt, register
from ..protocols.disagreement import Disagreement, DisagreementOutcome, Resolution, resolve
from ..protocols.schemas import AgentMessage, Evidence, MessageType, Verification

HYPOTHESIS_GENERATOR = register(VersionedPrompt(
    name="research_hypothesis_generator",
    version="v1",
    text="""\
Given a question and the evidence gathered so far, propose competing
explanations. Competing is the requirement: if your candidates could all be
true at once, you have produced one hypothesis in three phrasings and the
experiment that follows will discriminate nothing.

For each, state what observation would show it to be WRONG. A candidate you
cannot describe being wrong about does not belong in the list.

Do not rank them and do not say which you prefer. Ranking happens on evidence,
not on preference, and yours would only add noise to it.

Reply with a single JSON object and nothing else:

{
  "hypotheses": [
    {
      "claim": "<one sentence>",
      "supported_by": ["<which given observation supports this, verbatim>"],
      "unknowns": ["<what you cannot determine from what you were given>"],
      "confidence": <0.0-1.0>,
      "verification": {"metric": "<what to measure>",
                       "expected_if_correct": "<...>",
                       "expected_if_wrong": "<...>"}
    }
  ]
}""",
))


class Belief(str, enum.Enum):
    UNDECIDED = "UNDECIDED"
    LEADING = "LEADING"           # ranked first, but not established
    ESTABLISHED = "ESTABLISHED"   # survived a discriminating experiment
    REFUTED = "REFUTED"


@dataclass
class Hypothesis:
    claim: str
    evidence: list[Evidence] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    stated_confidence: float = 0.0
    verification: Optional[Verification] = None
    belief: Belief = Belief.UNDECIDED
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    hypothesis_id: str = ""

    @property
    def falsifiable(self) -> bool:
        return self.verification is not None and self.verification.is_discriminative()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "belief": self.belief.value,
            "score": round(self.score, 4),
            "score_breakdown": self.score_breakdown,
            "stated_confidence": self.stated_confidence,
            "falsifiable": self.falsifiable,
            "evidence": [e.to_dict() for e in self.evidence],
            "unknowns": self.unknowns,
        }

    def as_message(self, sender: str = "research") -> AgentMessage:
        """Adapt to the protocol shape so the disagreement ladder can take it."""
        return AgentMessage(
            sender=sender, recipient="environment", type=MessageType.HYPOTHESIS,
            claim=self.claim, evidence=list(self.evidence),
            confidence=self.stated_confidence, unknowns=list(self.unknowns),
            verification=self.verification, message_id=self.hypothesis_id or "",
        )


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

# Environment observations outrank claims about observations. The ladder is the
# same one section 16 uses for decisions.
ENVIRONMENT_SOURCES = frozenset({
    "run_tests", "shell", "experiment", "tests", "metrics", "filesystem",
    "test_runner", "read_file_region", "search_code", "list_files",
})


def rank(hypotheses: Iterable[Hypothesis]) -> list[Hypothesis]:
    """Score and order by evidence. Highest first; stable on ties.

    Every component is a countable fact about the hypothesis. Nothing here
    reads how convincing the claim sounds, because that is the input a research
    loop must not optimise against.
    """
    scored = list(hypotheses)
    for h in scored:
        distinct = {f"{e.source}:{e.observation[:120]}" for e in h.evidence}
        grounded = {e for e in h.evidence if _is_environment(e.source)}

        breakdown = {
            # each distinct observation counts, with diminishing returns: the
            # tenth citation of the same fact is not ten times the evidence
            "evidence": min(len(distinct), 5) * 0.20,
            # an observation from the environment beats a report of one
            "grounded": min(len(grounded), 3) * 0.15,
            # a claim that cannot be wrong cannot be tested
            "falsifiable": 0.25 if h.falsifiable else 0.0,
            # naming your unknowns is a mark of a usable hypothesis
            "admits_unknowns": 0.10 if h.unknowns else 0.0,
            # confidence is a tie-breaker only, capped small
            "confidence": min(max(h.stated_confidence, 0.0), 1.0) * 0.05,
            # and confidence outrunning evidence is a penalty, not a bonus
            "overconfidence": -_overconfidence_penalty(h, distinct),
        }
        h.score_breakdown = {k: round(v, 4) for k, v in breakdown.items()}
        h.score = round(sum(breakdown.values()), 4)

    scored.sort(key=lambda h: (-h.score, h.claim))
    for i, h in enumerate(scored):
        if h.belief in (Belief.ESTABLISHED, Belief.REFUTED):
            continue
        h.belief = Belief.LEADING if i == 0 and h.score > 0 else Belief.UNDECIDED
    return scored


def _is_environment(source: str) -> bool:
    s = (source or "").lower()
    return any(token in s for token in ENVIRONMENT_SOURCES)


def _overconfidence_penalty(h: Hypothesis, distinct: set[str]) -> float:
    """Penalise the gap between how sure a hypothesis sounds and what backs it.

    Without this, stated confidence is a free variable a model can raise at no
    cost, and the loop rewards exactly the behaviour it should distrust.
    """
    if not distinct and h.stated_confidence > 0.5:
        return (h.stated_confidence - 0.5) * 0.6
    if len(distinct) == 1 and h.stated_confidence > 0.8:
        return (h.stated_confidence - 0.8) * 0.4
    return 0.0


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

@dataclass
class ResearchRound:
    number: int
    hypotheses: list[Hypothesis] = field(default_factory=list)
    experiment: Optional[dict[str, Any]] = None
    outcome: Optional[DisagreementOutcome] = None
    new_evidence: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.number,
            "new_evidence": self.new_evidence,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "experiment": self.experiment,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "note": self.note,
        }


@dataclass
class ResearchResult:
    question: str
    rounds: list[ResearchRound] = field(default_factory=list)
    conclusion: Optional[Hypothesis] = None
    termination_reason: str = ""

    @property
    def established(self) -> bool:
        return bool(self.conclusion and self.conclusion.belief is Belief.ESTABLISHED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "termination_reason": self.termination_reason,
            "established": self.established,
            "conclusion": self.conclusion.to_dict() if self.conclusion else None,
            "rounds": [r.to_dict() for r in self.rounds],
        }

    def summary(self) -> str:
        if self.established:
            return f"ESTABLISHED: {self.conclusion.claim}"
        if self.conclusion:
            return (f"LEADING (not established): {self.conclusion.claim} "
                    f"-- {self.termination_reason}")
        return f"no conclusion -- {self.termination_reason}"


def generate(question: str, evidence: list[Evidence], judge: JudgeClient,
             n: int = 3) -> list[Hypothesis]:
    """Ask for competing explanations of the evidence so far."""
    observations = "\n".join(f"- ({e.source}) {e.observation}" for e in evidence) \
        or "(no observations gathered yet)"
    user = (f"QUESTION: {question}\n\nEVIDENCE SO FAR:\n{observations}\n\n"
            f"Propose up to {n} competing explanations.")
    try:
        reply = judge.ask_json(HYPOTHESIS_GENERATOR.text, user,
                               prompt_id=HYPOTHESIS_GENERATOR.id)
    except JudgeError:
        return []

    by_source = {e.observation[:120]: e for e in evidence}
    out: list[Hypothesis] = []
    for i, raw in enumerate(reply.data.get("hypotheses") or []):
        if not isinstance(raw, dict) or not str(raw.get("claim") or "").strip():
            continue
        supporting = []
        for quote in (raw.get("supported_by") or []):
            match = by_source.get(str(quote)[:120])
            supporting.append(match or Evidence(source="cited", observation=str(quote)[:400]))

        v = raw.get("verification")
        verification = None
        if isinstance(v, dict):
            verification = Verification(
                metric=str(v.get("metric") or ""),
                expected_if_correct=str(v.get("expected_if_correct") or ""),
                expected_if_wrong=str(v.get("expected_if_wrong") or ""),
            )
        out.append(Hypothesis(
            claim=str(raw["claim"])[:600],
            evidence=supporting,
            unknowns=[str(u)[:300] for u in (raw.get("unknowns") or [])][:6],
            stated_confidence=_as_float(raw.get("confidence")),
            verification=verification,
            hypothesis_id=f"h{i + 1}",
        ))
    return out[:n]


def update_belief(ranked: list[Hypothesis], new_evidence: int) -> str:
    """Whether the ranking may move. Returns a note explaining the decision.

    A round that gathered nothing new cannot change what is believed, however
    much re-reasoning happened inside it. Allowing that is how a loop converges
    on its own eloquence.
    """
    if new_evidence > 0:
        return f"{new_evidence} new observation(s) -- ranking updated"
    return ("no new evidence this round -- belief unchanged; re-reasoning over the "
            "same observations is not an update")


def investigate(
    question: str,
    judge: JudgeClient,
    initial_evidence: Optional[list[Evidence]] = None,
    gather: Optional[Callable[[str], tuple[bool, str]]] = None,
    run_experiment: Optional[Callable[[str], tuple[bool, str]]] = None,
    max_rounds: int = 3,
    n_hypotheses: int = 3,
) -> ResearchResult:
    """Run the loop until something is established, or it stops honestly.

    ``run_experiment(action) -> (ok, observation)`` executes in the real
    environment. Without it the loop can rank hypotheses but can never
    establish one, and it says so rather than promoting the leader.
    """
    result = ResearchResult(question=question)
    evidence: list[Evidence] = list(initial_evidence or [])
    seen = {f"{e.source}:{e.observation[:120]}" for e in evidence}

    for number in range(1, max(1, max_rounds) + 1):
        rnd = ResearchRound(number=number)

        hypotheses = generate(question, evidence, judge, n_hypotheses)
        if not hypotheses:
            rnd.note = "no hypotheses proposed"
            result.rounds.append(rnd)
            result.termination_reason = "NO_HYPOTHESES"
            break

        rnd.hypotheses = rank(hypotheses)
        result.conclusion = rnd.hypotheses[0]

        # Two top hypotheses that differ is exactly the situation the
        # disagreement ladder exists for: settle it by observation.
        if len(rnd.hypotheses) >= 2 and run_experiment is not None:
            top, second = rnd.hypotheses[0], rnd.hypotheses[1]
            disagreement = Disagreement(task_id=question,
                                        a=top.as_message("hypothesis_a"),
                                        b=second.as_message("hypothesis_b"))
            outcome = resolve(disagreement, judge=judge, run_experiment=run_experiment)
            rnd.outcome = outcome
            rnd.experiment = outcome.experiment

            for e in outcome.evidence:
                key = f"{e.source}:{e.observation[:120]}"
                if key not in seen:
                    seen.add(key)
                    evidence.append(e)
                    rnd.new_evidence += 1

            if outcome.resolution is Resolution.RESOLVED_BY_EXPERIMENT:
                winner = top if outcome.winning_agent == "hypothesis_a" else second
                loser = second if winner is top else top
                winner.belief = Belief.ESTABLISHED
                loser.belief = Belief.REFUTED
                result.conclusion = winner
                rnd.note = update_belief(rnd.hypotheses, rnd.new_evidence)
                result.rounds.append(rnd)
                result.termination_reason = "ESTABLISHED_BY_EXPERIMENT"
                break

        # Otherwise gather more, if we were given a way to.
        if gather is not None:
            ok, observation = gather(question)
            key = f"gather:{observation[:120]}"
            if observation and key not in seen:
                seen.add(key)
                evidence.append(Evidence(source="gather", observation=observation[:2000]))
                rnd.new_evidence += 1

        rnd.note = update_belief(rnd.hypotheses, rnd.new_evidence)
        result.rounds.append(rnd)

        if rnd.new_evidence == 0:
            result.termination_reason = "NO_NEW_EVIDENCE"
            break
    else:
        result.termination_reason = "MAX_ROUNDS"

    if not result.termination_reason:
        result.termination_reason = "MAX_ROUNDS"
    return result


def _as_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
