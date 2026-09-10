"""Role definitions for the default two-agent configuration (spec section 13).

    Claude  -> IMPLEMENTATION / CODEBASE agent
    OpenAI  -> RESEARCH / CRITIQUE / EXPERIMENT DESIGN agent

Roles are configurable and carry no vendor assumption: a ``Role`` is a name, a
remit, a set of message types it may send, and a client. Which provider sits
behind that client is a configuration detail, and the orchestrator never asks.

The allow-list is the part that does real work. Without it "structured
delegation" is a suggestion, and the predictable outcome is both agents doing
the same job — the researcher starts editing files, the engineer starts
speculating, and you are paying twice for one agent's work while calling it
collaboration. A message outside a role's remit is recorded as a
ROLE_VIOLATION and never delivered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import MessageType

# Types any role may send: asking, reporting what was seen, flagging a problem.
_COMMON: frozenset[MessageType] = frozenset({
    MessageType.QUESTION,
    MessageType.EVIDENCE,
    MessageType.EXPERIMENT_RESULT,
    MessageType.CRITIQUE,
    MessageType.BLOCKER,
    MessageType.DECISION_REQUEST,
    MessageType.FINAL_REPORT,
})

# IMPLEMENTATION_PLAN belongs to whoever owns the codebase. A researcher that
# writes the plan has quietly become a second engineer, and the disagreement
# protocol then has nobody independent left to check the work.
ROLE_ALLOWED: dict[str, frozenset[MessageType]] = {
    "engineer": _COMMON | {MessageType.IMPLEMENTATION_PLAN, MessageType.HYPOTHESIS},
    "researcher": _COMMON | {MessageType.HYPOTHESIS, MessageType.EXPERIMENT_PROPOSAL},
}

DEFAULT_REMITS: dict[str, str] = {
    "engineer": "implementation and codebase work: inspect the environment, make "
                "changes, run tests, and report only what you actually observed",
    "researcher": "research, critique and experiment design: examine claims, name "
                  "what is unverified, and propose the experiment that would "
                  "distinguish competing explanations",
}


@dataclass
class RoleConfig:
    """What a role is allowed to be and do."""

    name: str
    remit: str = ""
    allowed: Optional[frozenset[MessageType]] = None
    model_hint: str = ""     # advisory only; the orchestrator never reads it

    def __post_init__(self) -> None:
        if not self.remit:
            self.remit = DEFAULT_REMITS.get(self.name, "unspecified remit")
        if self.allowed is None:
            # An unknown role gets the common set rather than everything.
            # Defaulting to permissive would make the allow-list decorative.
            self.allowed = ROLE_ALLOWED.get(self.name, _COMMON)

    def may_send(self, message_type: MessageType) -> bool:
        return message_type in (self.allowed or frozenset())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "remit": self.remit,
            "allowed": sorted(t.value for t in (self.allowed or frozenset())),
        }


@dataclass
class Role:
    """A configured role bound to the client that plays it."""

    config: RoleConfig
    client: Any = None          # a JudgeClient, or anything with ask_json()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def remit(self) -> str:
        return self.config.remit

    def may_send(self, message_type: MessageType) -> bool:
        return self.config.may_send(message_type)


def default_roles(engineer_client: Any, researcher_client: Any) -> list[Role]:
    """The spec's default pairing. Order matters: the researcher opens.

    The researcher going first is deliberate. Ask the engineer to lead and it
    starts editing before anyone has said what the problem is, which is the
    expensive direction to be wrong in.
    """
    return [
        Role(RoleConfig(name="researcher"), researcher_client),
        Role(RoleConfig(name="engineer"), engineer_client),
    ]
