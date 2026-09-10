"""Structured messages between agents (spec section 14).

Agents exchange typed artifacts, not conversation. That constraint is the whole
design: free-form chat between two capable models produces fluent agreement,
drifts off task, and leaves nothing you can audit afterwards. A HYPOTHESIS that
must carry evidence, a confidence, its unknowns, and a stated way to check it
cannot be produced by agreeing pleasantly.

The field that does the most work is ``verification``:

    expected_if_correct / expected_if_wrong

A claim whose author cannot say what would distinguish it from its negation is
not a hypothesis, it is a preference. Requiring both halves up front is what
later makes disagreement resolvable by experiment rather than by argument --
the discriminating observation was named before anyone knew who would win.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..types import utcnow


class MessageType(str, enum.Enum):
    TASK = "TASK"
    QUESTION = "QUESTION"
    HYPOTHESIS = "HYPOTHESIS"
    EVIDENCE = "EVIDENCE"
    IMPLEMENTATION_PLAN = "IMPLEMENTATION_PLAN"
    EXPERIMENT_PROPOSAL = "EXPERIMENT_PROPOSAL"
    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    CRITIQUE = "CRITIQUE"
    BLOCKER = "BLOCKER"
    DECISION_REQUEST = "DECISION_REQUEST"
    FINAL_REPORT = "FINAL_REPORT"


# Above this, a claim with no evidence behind it is a reasoning/action
# mismatch rather than legitimate uncertainty.
CONFIDENCE_NEEDS_EVIDENCE = 0.8

# Types that make a factual claim, and therefore must say how they could be wrong.
CLAIM_TYPES = frozenset({
    MessageType.HYPOTHESIS,
    MessageType.EXPERIMENT_PROPOSAL,
})


class ProtocolViolation(ValueError):
    """A message did not satisfy the schema its type requires."""


@dataclass(frozen=True)
class Evidence:
    """One observation, tied to where it came from.

    ``source`` is a tool name, file, run id, or metric -- something checkable.
    An observation with no source is an assertion.
    """

    source: str
    observation: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "observation": self.observation}


@dataclass(frozen=True)
class Verification:
    """How to tell whether a claim is right.

    Both halves are required. If ``expected_if_correct`` and
    ``expected_if_wrong`` are the same, the check discriminates nothing and the
    schema refuses it.
    """

    metric: str
    expected_if_correct: str
    expected_if_wrong: str

    def is_discriminative(self) -> bool:
        a = (self.expected_if_correct or "").strip().lower()
        b = (self.expected_if_wrong or "").strip().lower()
        return bool(a) and bool(b) and a != b

    def to_dict(self) -> dict[str, str]:
        return {
            "metric": self.metric,
            "expected_if_correct": self.expected_if_correct,
            "expected_if_wrong": self.expected_if_wrong,
        }


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    type: MessageType
    claim: str = ""
    task_id: str = ""
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:10]}")
    evidence: list[Evidence] = field(default_factory=list)
    confidence: Optional[float] = None
    unknowns: list[str] = field(default_factory=list)
    recommended_action: str = ""
    verification: Optional[Verification] = None
    in_reply_to: Optional[str] = None
    round: int = 0
    at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "task_id": self.task_id,
            "type": self.type.value,
            "claim": self.claim,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "unknowns": self.unknowns,
            "recommended_action": self.recommended_action,
            "verification": self.verification.to_dict() if self.verification else None,
            "in_reply_to": self.in_reply_to,
            "round": self.round,
            "at": self.at,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AgentMessage":
        verification = None
        if d.get("verification"):
            v = d["verification"]
            verification = Verification(
                metric=v.get("metric", ""),
                expected_if_correct=v.get("expected_if_correct", ""),
                expected_if_wrong=v.get("expected_if_wrong", ""),
            )
        return AgentMessage(
            sender=d.get("sender", ""),
            recipient=d.get("recipient", ""),
            type=MessageType(d.get("type", "QUESTION")),
            claim=d.get("claim", ""),
            task_id=d.get("task_id", ""),
            message_id=d.get("message_id") or f"msg_{uuid.uuid4().hex[:10]}",
            evidence=[Evidence(source=e.get("source", ""), observation=e.get("observation", ""))
                      for e in (d.get("evidence") or []) if isinstance(e, dict)],
            confidence=_as_confidence(d.get("confidence")),
            unknowns=[str(u) for u in (d.get("unknowns") or [])],
            recommended_action=d.get("recommended_action", ""),
            verification=verification,
            in_reply_to=d.get("in_reply_to"),
            round=int(d.get("round", 0)),
        )


def validate(message: AgentMessage) -> list[str]:
    """Return schema problems. Empty list means the message is well-formed."""
    problems: list[str] = []

    if not message.sender or not message.recipient:
        problems.append("sender and recipient are both required")
    if message.sender and message.sender == message.recipient:
        problems.append("a message addressed to its own sender is not a handoff")

    if message.type in CLAIM_TYPES:
        if not message.claim.strip():
            problems.append(f"a {message.type.value} must state a claim")
        if message.verification is None:
            problems.append(
                f"a {message.type.value} must say how it could be checked -- supply "
                f"verification with expected_if_correct and expected_if_wrong"
            )
        elif not message.verification.is_discriminative():
            problems.append(
                "verification does not discriminate: expected_if_correct and "
                "expected_if_wrong must differ, or the check cannot tell you anything"
            )
        # A hypothesis is allowed to be unevidenced -- that is what makes it a
        # hypothesis, and `verification` is how it stays checkable. What is not
        # allowed is near-certainty with nothing observed behind it: that is a
        # reasoning/action mismatch, and it is how an unfounded claim gets
        # treated downstream as though it were a finding.
        if not message.evidence and (message.confidence or 0) > CONFIDENCE_NEEDS_EVIDENCE:
            problems.append(
                f"confidence {message.confidence} with no evidence -- state the "
                f"observations behind it, or lower the confidence to match what "
                f"you actually know"
            )

    if message.confidence is not None and not 0.0 <= message.confidence <= 1.0:
        problems.append(f"confidence {message.confidence} is outside 0.0-1.0")

    if message.type is MessageType.EXPERIMENT_RESULT and not message.evidence:
        problems.append("an EXPERIMENT_RESULT must carry the observations it produced")

    return problems


def require_valid(message: AgentMessage) -> AgentMessage:
    problems = validate(message)
    if problems:
        raise ProtocolViolation(
            f"{message.type.value} from {message.sender}: " + "; ".join(problems)
        )
    return message


def _as_confidence(value: Any) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
