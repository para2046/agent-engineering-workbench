"""Bounded multi-agent orchestration (spec sections 13-17).

`protocols.schemas` says what an agent may state, `protocols.researcher_engineer`
says who may state it, and `protocols.disagreement` says how a conflict gets
settled. This runs the exchange.

The spec is blunt about what this must not become: "Do NOT implement
unrestricted free-form conversation", and "Prevent endless Claude -> GPT ->
Claude -> GPT". Three mechanisms enforce that, and they are the whole module:

**Every turn is a schema-valid artifact or it is not a turn.** A reply that
fails validation is recorded against its sender and never delivered. The other
agent is never asked to interpret malformed input, because an agent handed
garbage will helpfully invent a reading of it and proceed with confidence.

**A seat may only send what its remit allows.** Without that, both agents drift
into the same job and you pay twice for one agent's work while calling it
collaboration -- and the independent check, the only reason to run two agents,
quietly disappears.

**A round must earn its existence.** After each exchange the orchestrator asks
whether anything genuinely new arrived. If not, the run stops with
NO_NEW_EVIDENCE. Two agents restating themselves in better words is the
characteristic multi-agent failure, and it is expensive precisely because the
transcript looks like progress.

Nothing here decides truth by agreement. Two agents concurring is correlated
guessing, not evidence; conflicts go to the disagreement ladder, which prefers
a discriminating experiment to an opinion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..judge.client import JudgeError
from ..prompts import VersionedPrompt, register
from ..protocols import disagreement as disagreement_mod
from ..protocols.researcher_engineer import DEFAULT_REMITS, RoleConfig
from ..protocols.schemas import AgentMessage, MessageType, validate
from ..types import FailureCategory, TerminationReason, utcnow

ROLE_PROMPT = register(VersionedPrompt(
    name="multi_agent_role",
    version="v1",
    text="""\
You are one of two agents working on a shared task. You are the {role} agent.

Your remit: {remit}

You may send only these message types: {allowed}

Rules of the exchange:
- Stay inside your remit. Do not do the other agent's job -- ask for it.
- Every message is a structured artifact, not conversation. No pleasantries.
- A claim you have not observed is a HYPOTHESIS. Label it as one and say what
  observation would show it to be wrong.
- Match your confidence to your evidence. High confidence with nothing observed
  behind it is recorded as a reasoning/action mismatch, not as conviction.
- If you have nothing new -- no new evidence, no experiment result, no claim
  not already on the table -- say so with type BLOCKER. Restating your previous
  position in different words wastes a round and is not progress.
- Never assert what you did not observe. "I could not determine X" is useful;
  a confident guess is not.

Reply with a single JSON object and nothing else:

{{
  "type": "<one of the allowed types>",
  "claim": "<one sentence>",
  "evidence": [{{"source": "<where it came from>", "observation": "<what was seen>"}}],
  "confidence": <0.0-1.0>,
  "unknowns": ["<what you could not determine>"],
  "recommended_action": "<what should happen next>",
  "verification": {{"metric": "<what to measure>",
                   "expected_if_correct": "<...>",
                   "expected_if_wrong": "<...>"}}
}}

`verification` is required for HYPOTHESIS and EXPERIMENT_PROPOSAL: a claim that
cannot be wrong cannot be checked.""",
))


@dataclass
class Exchange:
    """The record of one multi-agent run."""

    task_id: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    rounds: int = 0
    termination_reason: str = ""
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    duplicate_work: int = 0
    started_at: str = field(default_factory=utcnow)
    ended_at: str = ""

    @property
    def metrics(self) -> dict[str, Any]:
        """Spec section 27's multi-agent metrics."""
        unresolved = sum(1 for d in self.disagreements
                         if d.get("resolution") in {"UNRESOLVED", "HUMAN_REQUIRED"})
        return {
            "rounds": self.rounds,
            "handoffs": len(self.messages),
            "disagreements": len(self.disagreements),
            "unresolved_disagreements": unresolved,
            "protocol_violations": len(self.violations),
            "duplicate_work": self.duplicate_work,
            "evidence_items": sum(len(m.evidence) for m in self.messages),
            "blockers": sum(1 for m in self.messages if m.type is MessageType.BLOCKER),
            "ignored_evidence": self._ignored_evidence(),
            "verification_failures": self._verification_failures(),
        }

    def _ignored_evidence(self) -> int:
        """Claims restated after a disagreement already ruled against them.

        This is the failure where a conflict is settled by experiment and an
        agent carries on as though it were not -- the expensive one, because
        the exchange looks like it is progressing while one side is arguing
        against a result already on the table.
        """
        losers: list[str] = []
        for record in self.disagreements:
            outcome = record.get("outcome", record)
            if not str(outcome.get("resolution", "")).startswith("RESOLVED"):
                continue
            winner_id = outcome.get("winner")
            pair = [record.get("a"), record.get("b")]
            for message_id in pair:
                if message_id and message_id != winner_id:
                    losers.append(message_id)

        losing_claims = {_normalise(m.claim) for m in self.messages
                         if m.message_id in losers and m.claim}
        if not losing_claims:
            return 0

        # Only messages sent after the ruling can ignore it.
        resolved_after = False
        count = 0
        for m in self.messages:
            if m.message_id in losers:
                resolved_after = True
                continue
            if resolved_after and _normalise(m.claim) in losing_claims:
                count += 1
        return count

    def _verification_failures(self) -> int:
        """Closing claims with nothing observed behind them.

        A FINAL_REPORT is the one message that asserts the work is done. One
        carrying no evidence is the multi-agent form of the single-agent
        failure this whole system exists to catch: a conclusion asserted rather
        than shown.
        """
        return sum(1 for m in self.messages
                   if m.type is MessageType.FINAL_REPORT and not m.evidence)

    @property
    def final_report(self) -> Optional[AgentMessage]:
        for m in reversed(self.messages):
            if m.type is MessageType.FINAL_REPORT:
                return m
        return None

    def transcript(self) -> str:
        """Human-readable exchange, bounded like every other observation here."""
        lines = [f"exchange for task {self.task_id} "
                 f"({self.rounds} round(s), {self.termination_reason})"]
        for m in self.messages:
            lines.append(f"  [r{m.round}] {m.sender} -> {m.recipient}  "
                         f"{m.type.value}: {m.claim}")
            for e in m.evidence[:3]:
                lines.append(f"        evidence({e.source}): {e.observation[:160]}")
        for v in self.violations:
            lines.append(f"  [r{v.get('round')}] {v.get('agent')} "
                         f"{v.get('category')}: {v.get('problem')}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rounds": self.rounds,
            "termination_reason": self.termination_reason,
            "metrics": self.metrics,
            "messages": [m.to_dict() for m in self.messages],
            "disagreements": self.disagreements,
            "violations": self.violations,
            "failure_categories": self.failure_categories,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class Orchestrator:
    """Runs a bounded, structured exchange between two or more seats."""

    def __init__(
        self,
        agents: dict[str, Any],
        roles: Optional[dict[str, str]] = None,
        max_rounds: int = 4,
        judge=None,
        run_experiment: Optional[Callable[[str], tuple[bool, str]]] = None,
    ):
        """
        agents          seat name -> a client with ask_json()
        roles           seat name -> remit text, overriding the defaults
        max_rounds      hard ceiling; the soft stop is "nothing new arrived"
        judge           used only to design a discriminating experiment
        run_experiment  executes that experiment in the real environment
        """
        if len(agents) < 2:
            raise ValueError(
                "multi-agent orchestration needs at least two agents -- with one "
                "there is no independent check, which is the only reason to pay "
                "for a second seat"
            )
        self.agents = agents
        self.roles = {**DEFAULT_REMITS, **(roles or {})}
        # One RoleConfig per seat. Without these the allow-list in
        # protocols.researcher_engineer would be documentation, not a rule.
        self.role_configs = {
            name: RoleConfig(name=name, remit=self.roles.get(name, ""))
            for name in agents
        }
        # Clamped rather than honoured: max_rounds=0 reads as either unbounded
        # or never-starting, and neither is something this module may do.
        self.max_rounds = max(1, int(max_rounds))
        self.judge = judge
        self.run_experiment = run_experiment

    # -- the loop --------------------------------------------------------
    def run(self, task_id: str, task_prompt: str, success_criteria: str = "") -> Exchange:
        exchange = Exchange(task_id=task_id)
        seen_claims: set[str] = set()
        seen_evidence: set[str] = set()

        for round_no in range(1, self.max_rounds + 1):
            exchange.rounds = round_no
            new_this_round = 0

            for seat in self.agents:
                message = self._turn(seat, task_prompt, success_criteria,
                                     exchange, round_no)
                if message is None:
                    continue
                exchange.messages.append(message)
                new_this_round += self._absorb(message, seen_claims, seen_evidence, exchange)

            self._check_disagreement(exchange)

            if exchange.final_report is not None:
                exchange.termination_reason = TerminationReason.SUCCESS.value
                break

            if new_this_round == 0:
                # Spec section 17: another round has to be earned.
                exchange.termination_reason = TerminationReason.NO_NEW_EVIDENCE.value
                break
        else:
            exchange.termination_reason = TerminationReason.MAX_ITERATIONS.value

        exchange.ended_at = utcnow()
        self._classify(exchange)
        return exchange

    # -- one turn --------------------------------------------------------
    def _turn(self, seat: str, task_prompt: str, success_criteria: str,
              exchange: Exchange, round_no: int) -> Optional[AgentMessage]:
        config = self.role_configs[seat]
        allowed = ", ".join(sorted(t.value for t in (config.allowed or [])))
        system = ROLE_PROMPT.text.format(role=seat, remit=config.remit, allowed=allowed)
        user = self._context(task_prompt, success_criteria, exchange, seat)

        try:
            reply = self.agents[seat].ask_json(system, user, prompt_id=ROLE_PROMPT.id)
        except JudgeError as exc:
            return self._violation(exchange, seat, round_no, str(exc),
                                   FailureCategory.ROLE_VIOLATION.value, "unusable")

        payload = dict(reply.data or {})
        if not payload.get("type"):
            return self._violation(exchange, seat, round_no,
                                   "reply carried no message type",
                                   FailureCategory.ROLE_VIOLATION.value, "unusable")

        payload["sender"] = seat
        payload["recipient"] = self._other(seat)
        payload["task_id"] = exchange.task_id
        payload["round"] = round_no

        try:
            message = AgentMessage.from_dict(payload)
        except (ValueError, KeyError) as exc:
            return self._violation(exchange, seat, round_no,
                                   f"unparseable message: {exc}",
                                   FailureCategory.ROLE_VIOLATION.value, "unusable")

        # Remit is checked before schema. Sending something you are not allowed
        # to send is a different failure from sending it badly, and collapsing
        # the two hides which one is actually happening.
        if not config.may_send(message.type):
            return self._violation(
                exchange, seat, round_no,
                f"{seat} sent {message.type.value}, which is outside its remit",
                FailureCategory.ROLE_VIOLATION.value, "remit")

        problems = validate(message)
        if problems:
            return self._violation(
                exchange, seat, round_no, "; ".join(problems),
                FailureCategory.REASONING_ACTION_MISMATCH.value, "schema")
        return message

    def _violation(self, exchange: Exchange, seat: str, round_no: int,
                   problem: str, category: str, kind: str) -> None:
        exchange.violations.append({
            "round": round_no, "agent": seat, "kind": kind,
            "problem": problem, "category": category,
        })
        return None

    def _context(self, task_prompt: str, success_criteria: str,
                 exchange: Exchange, seat: str) -> str:
        lines = [f"TASK: {task_prompt}"]
        if success_criteria:
            lines.append(f"SUCCESS CRITERIA: {success_criteria}")
        lines += ["", "EXCHANGE SO FAR:"]
        if not exchange.messages:
            lines.append("  (nothing yet -- you are opening)")
        for m in exchange.messages[-8:]:
            lines.append(f"  [{m.sender} -> {m.recipient}] {m.type.value}: {m.claim}")
            for e in m.evidence[:3]:
                lines.append(f"      evidence({e.source}): {e.observation[:200]}")
            if m.unknowns:
                lines.append(f"      unknowns: {'; '.join(m.unknowns[:3])}")
        lines += ["", f"You are the {seat} agent. Reply with one JSON message."]
        return "\n".join(lines)

    def _other(self, seat: str) -> str:
        for name in self.agents:
            if name != seat:
                return name
        return seat

    # -- bookkeeping -----------------------------------------------------
    @staticmethod
    def _absorb(message: AgentMessage, seen_claims: set[str], seen_evidence: set[str],
                exchange: Exchange) -> int:
        """Count what was genuinely new; record what was merely restated.

        Claims are normalised before comparison, so rephrasing does not read as
        novelty -- which is exactly the move a stuck agent makes.
        """
        new = 0
        key = _normalise(message.claim)
        if key:
            if key in seen_claims:
                exchange.duplicate_work += 1
            else:
                seen_claims.add(key)
                new += 1

        for e in message.evidence:
            ekey = _normalise(f"{e.source} {e.observation}")
            if ekey and ekey not in seen_evidence:
                seen_evidence.add(ekey)
                new += 1

        if message.type is MessageType.EXPERIMENT_RESULT:
            new += 1        # a result is always new information
        return new

    def _check_disagreement(self, exchange: Exchange) -> None:
        found = disagreement_mod.detect(exchange.messages, task_id=exchange.task_id)
        if found is None:
            return
        outcome = disagreement_mod.resolve(
            found, judge=self.judge, run_experiment=self.run_experiment)
        exchange.disagreements.append(_to_plain(outcome))

    @staticmethod
    def _classify(exchange: Exchange) -> None:
        """Deterministic multi-agent failure labels (spec section 10)."""
        labels: set[str] = set(exchange.failure_categories)
        for v in exchange.violations:
            if v.get("category"):
                labels.add(v["category"])
        if any(d.get("resolution") in {"UNRESOLVED", "HUMAN_REQUIRED"}
               for d in exchange.disagreements):
            labels.add(FailureCategory.AGENT_DISAGREEMENT.value)
        if exchange.duplicate_work:
            labels.add(FailureCategory.REPEATED_ACTION.value)
        if exchange.termination_reason == TerminationReason.NO_NEW_EVIDENCE.value:
            labels.add(FailureCategory.REPEATED_ACTION.value)
        if exchange.messages and not any(m.evidence for m in exchange.messages):
            labels.add(FailureCategory.INCOMPLETE_VERIFICATION.value)
        exchange.failure_categories = sorted(labels)


def _normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Deliberately aggressive: the question is whether the *substance* is new,
    and a stuck agent's second attempt usually differs only in wording.
    """
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _to_plain(outcome: Any) -> dict[str, Any]:
    """Flatten a DisagreementOutcome, resolving enums to their values."""
    if hasattr(outcome, "to_dict"):
        data = outcome.to_dict()
    else:
        data = dict(getattr(outcome, "__dict__", {}) or {})
    return {k: (v.value if hasattr(v, "value") else v) for k, v in data.items()}


def save_exchange(exchange: Exchange, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(exchange.to_dict(), indent=2, ensure_ascii=False,
                              default=str), encoding="utf-8")
    return path


# The exchange record has been referred to by both names during development.
# Aliased rather than renamed: callers should not have to care, and a rename
# that breaks imports is a worse outcome than one extra line here.
ExchangeResult = Exchange
