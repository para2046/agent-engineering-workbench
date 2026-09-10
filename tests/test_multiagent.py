"""Multi-agent protocol tests.

The spec's warning is that multi-agent systems *add* coordination, verification
and termination failure modes rather than removing them. These tests pin the
mechanism meant to contain each: schema validation and role remits
(coordination), disagreement-by-experiment (verification), and bounded rounds
that must be earned (termination).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentwb.judge.client import JudgeClient
from agentwb.judge.mock import FakeJudgeProvider
from agentwb.protocols.disagreement import Resolution, detect, resolve
from agentwb.protocols.researcher_engineer import (
    DEFAULT_REMITS,
    ROLE_ALLOWED,
    Role,
    RoleConfig,
    default_roles,
)
from agentwb.protocols.schemas import (
    AgentMessage,
    Evidence,
    MessageType,
    ProtocolViolation,
    Verification,
    require_valid,
    validate,
)
from agentwb.runtime.orchestrator import Exchange, Orchestrator, save_exchange
from agentwb.types import FailureCategory, TerminationReason


def verification(a="tests pass", b="tests still fail") -> Verification:
    return Verification(metric="test suite", expected_if_correct=a, expected_if_wrong=b)


def hypothesis(sender="researcher", claim="the pool is exhausted by per-item lookups",
               evidence=None, confidence=0.7, **kw) -> AgentMessage:
    """A well-formed hypothesis. Confidence above 0.5 requires evidence, so the
    default fixture carries some -- an unevidenced 0.7 is a schema violation."""
    if evidence is None and confidence > 0.8:
        evidence = [Evidence("metrics", "pool saturated at 100/100")]
    return AgentMessage(
        sender=sender, recipient="engineer", type=MessageType.HYPOTHESIS,
        claim=claim, evidence=evidence or [], confidence=confidence,
        verification=kw.pop("verification", verification()), **kw,
    )


def client(*replies) -> JudgeClient:
    return JudgeClient(FakeJudgeProvider(list(replies)))


def msg(mtype, claim, evidence=None, confidence=0.4, verification_=None) -> dict:
    out = {"type": mtype, "claim": claim, "confidence": confidence}
    if evidence:
        out["evidence"] = evidence
    if verification_ is not None:
        out["verification"] = verification_
    elif mtype in ("HYPOTHESIS", "EXPERIMENT_PROPOSAL"):
        out["verification"] = {"metric": "m", "expected_if_correct": "a",
                               "expected_if_wrong": "b"}
    return out


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

class TestMessageSchema(unittest.TestCase):
    def test_a_well_formed_hypothesis_validates(self):
        self.assertEqual(validate(hypothesis()), [])

    def test_hypothesis_without_verification_is_rejected(self):
        self.assertTrue(any("how it could be checked" in p
                            for p in validate(hypothesis(verification=None))))

    def test_verification_that_predicts_the_same_either_way_is_rejected(self):
        """A check whose two outcomes are identical discriminates nothing."""
        m = hypothesis(verification=verification("it works", "it works"))
        self.assertTrue(any("does not discriminate" in p for p in validate(m)))

    def test_near_certainty_without_evidence_is_rejected(self):
        """A hypothesis may be unevidenced; near-certainty may not."""
        self.assertTrue(any("no evidence" in p
                            for p in validate(hypothesis(confidence=0.95, evidence=[]))))

    def test_moderate_confidence_without_evidence_is_allowed(self):
        """Being unevidenced is what makes it a hypothesis; verification keeps
        it checkable."""
        self.assertEqual(validate(hypothesis(confidence=0.3, evidence=[])), [])
        self.assertEqual(validate(hypothesis(confidence=0.7, evidence=[])), [])

    def test_confidence_outside_range_is_rejected(self):
        m = hypothesis(evidence=[Evidence("tests", "3 failed")])
        m.confidence = 1.4
        self.assertTrue(any("outside" in p for p in validate(m)))

    def test_message_to_itself_is_not_a_handoff(self):
        m = hypothesis()
        m.recipient = m.sender
        self.assertTrue(any("own sender" in p for p in validate(m)))

    def test_experiment_result_must_carry_observations(self):
        m = AgentMessage(sender="engineer", recipient="researcher",
                         type=MessageType.EXPERIMENT_RESULT, claim="ran it")
        self.assertTrue(any("must carry the observations" in p for p in validate(m)))

    def test_require_valid_raises(self):
        with self.assertRaises(ProtocolViolation):
            require_valid(hypothesis(verification=None))

    def test_round_trip_preserves_the_message(self):
        m = hypothesis(evidence=[Evidence("tests", "3 failed")], unknowns=["why it is slow"])
        self.assertEqual(AgentMessage.from_dict(m.to_dict()).to_dict(), m.to_dict())


# --------------------------------------------------------------------------
# roles
# --------------------------------------------------------------------------

class TestRoles(unittest.TestCase):
    def test_engineer_may_implement_and_researcher_may_not(self):
        self.assertTrue(RoleConfig(name="engineer").may_send(MessageType.IMPLEMENTATION_PLAN))
        self.assertFalse(RoleConfig(name="researcher").may_send(MessageType.IMPLEMENTATION_PLAN))

    def test_researcher_may_propose_experiments_and_engineer_may_not(self):
        self.assertTrue(RoleConfig(name="researcher").may_send(MessageType.EXPERIMENT_PROPOSAL))
        self.assertFalse(RoleConfig(name="engineer").may_send(MessageType.EXPERIMENT_PROPOSAL))

    def test_unknown_role_gets_the_common_set_not_everything(self):
        """Defaulting an unrecognised seat to permissive would make the
        allow-list decorative."""
        archivist = RoleConfig(name="archivist")
        self.assertTrue(archivist.may_send(MessageType.EVIDENCE))
        self.assertFalse(archivist.may_send(MessageType.IMPLEMENTATION_PLAN))
        self.assertFalse(archivist.may_send(MessageType.EXPERIMENT_PROPOSAL))

    def test_remit_defaults_are_filled_in(self):
        self.assertEqual(RoleConfig(name="engineer").remit, DEFAULT_REMITS["engineer"])
        self.assertIn("unspecified", RoleConfig(name="nobody").remit)

    def test_explicit_remit_and_allow_list_are_respected(self):
        cfg = RoleConfig(name="auditor", remit="read only",
                         allowed=frozenset({MessageType.CRITIQUE}))
        self.assertEqual(cfg.remit, "read only")
        self.assertTrue(cfg.may_send(MessageType.CRITIQUE))
        self.assertFalse(cfg.may_send(MessageType.EVIDENCE))

    def test_roles_carry_no_vendor_assumption(self):
        for name, remit in DEFAULT_REMITS.items():
            for vendor in ("claude", "openai", "anthropic", "gpt"):
                self.assertNotIn(vendor, remit.lower(), f"{name} names a vendor")

    def test_default_pairing_puts_the_researcher_first(self):
        """Ask the engineer to lead and it edits before anyone has said what
        the problem is -- the expensive direction to be wrong in."""
        roles = default_roles(engineer_client=client(), researcher_client=client())
        self.assertEqual([r.name for r in roles], ["researcher", "engineer"])
        self.assertIsInstance(roles[0], Role)

    def test_role_serialises_its_permissions(self):
        d = RoleConfig(name="researcher").to_dict()
        self.assertIn("EXPERIMENT_PROPOSAL", d["allowed"])
        self.assertNotIn("IMPLEMENTATION_PLAN", d["allowed"])


# --------------------------------------------------------------------------
# disagreement
# --------------------------------------------------------------------------

class TestDisagreementDetection(unittest.TestCase):
    def test_conflicting_hypotheses_are_detected(self):
        d = detect([
            hypothesis("researcher", "the connection pool is exhausted by per-item lookups"),
            hypothesis("engineer", "a database index was dropped during the migration"),
        ])
        self.assertIsNotNone(d)
        self.assertEqual(set(d.agents), {"researcher", "engineer"})

    def test_agreeing_hypotheses_are_not_a_disagreement(self):
        claim = "the connection pool is exhausted by per-item currency lookups"
        self.assertIsNone(detect([hypothesis("researcher", claim), hypothesis("engineer", claim)]))

    def test_one_agent_disagreeing_with_itself_is_ignored(self):
        self.assertIsNone(detect([
            hypothesis("researcher", "the pool is exhausted"),
            hypothesis("researcher", "an index was dropped during migration"),
        ]))

    def test_non_hypothesis_messages_are_ignored(self):
        self.assertIsNone(detect([
            AgentMessage(sender="a", recipient="b", type=MessageType.QUESTION, claim="why?"),
            AgentMessage(sender="b", recipient="a", type=MessageType.CRITIQUE, claim="unclear"),
        ]))


class TestDisagreementResolution(unittest.TestCase):
    def _pair(self, a_evidence=None, b_evidence=None):
        return detect([
            hypothesis("researcher", "the connection pool is exhausted by per-item lookups",
                       evidence=a_evidence or []),
            hypothesis("engineer", "a database index was dropped during the migration",
                       evidence=b_evidence or []),
        ])

    def test_grounded_hypothesis_beats_ungrounded_one(self):
        out = resolve(self._pair(a_evidence=[Evidence("metrics", "pool 100/100, cpu flat")]))
        self.assertIs(out.resolution, Resolution.RESOLVED_BY_EVIDENCE)
        self.assertEqual(out.winning_agent, "researcher")
        self.assertEqual(out.basis, "primary evidence")

    def test_no_judge_and_no_evidence_means_human_required(self):
        """It never falls back to the more confident or more fluent agent."""
        self.assertIs(resolve(self._pair()).resolution, Resolution.HUMAN_REQUIRED)

    def test_experiment_decides_when_evidence_cannot(self):
        judge = client(
            {"discriminative": True, "experiment": "SHOW INDEXES ON orders",
             "metric": "index list", "expected_if_a": "index present",
             "expected_if_b": "index missing", "reasoning": "they predict different rows"},
            {"supports": "A", "reasoning": "the index is present, so B is out"},
        )
        out = resolve(self._pair(), judge=judge,
                      run_experiment=lambda cmd: (True, "orders_idx  BTREE  active"))
        self.assertIs(out.resolution, Resolution.RESOLVED_BY_EXPERIMENT)
        self.assertEqual(out.winning_agent, "researcher")
        self.assertEqual(out.basis, "environment outcome")
        self.assertTrue(out.evidence)

    def test_a_designed_experiment_that_is_never_run_decides_nothing(self):
        judge = client({"discriminative": True, "experiment": "check the index",
                        "metric": "m", "expected_if_a": "x", "expected_if_b": "y"})
        out = resolve(self._pair(), judge=judge, run_experiment=None)
        self.assertIs(out.resolution, Resolution.HUMAN_REQUIRED)
        self.assertIn("unrun experiment", out.basis)

    def test_non_discriminative_design_is_refused_even_when_claimed_otherwise(self):
        """The model asserts discriminative: true, but both predictions match."""
        judge = client({"discriminative": True, "experiment": "look at it", "metric": "m",
                        "expected_if_a": "slow", "expected_if_b": "slow"})
        out = resolve(self._pair(), judge=judge, run_experiment=lambda c: (True, "slow"))
        self.assertIs(out.resolution, Resolution.HUMAN_REQUIRED)
        self.assertIn("no available observation", out.basis)

    def test_unclear_result_leaves_it_unresolved(self):
        judge = client(
            {"discriminative": True, "experiment": "run it", "metric": "m",
             "expected_if_a": "fast", "expected_if_b": "slow"},
            {"supports": "UNCLEAR", "reasoning": "the output matched neither"},
        )
        out = resolve(self._pair(), judge=judge, run_experiment=lambda c: (True, "inconclusive"))
        self.assertIs(out.resolution, Resolution.UNRESOLVED)
        self.assertIsNone(out.winning_agent)

    def test_the_designer_is_never_asked_who_is_right(self):
        provider = FakeJudgeProvider([{"discriminative": False, "experiment": ""}])
        resolve(self._pair(), judge=JudgeClient(provider), run_experiment=lambda c: (True, ""))
        sent = provider.prompts[0]
        self.assertIn("HYPOTHESIS A", sent)
        self.assertNotIn("which is correct", sent.lower())
        self.assertNotIn("more convincing", sent.lower())


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

class TestOrchestrator(unittest.TestCase):
    @staticmethod
    def _agents(researcher_replies, engineer_replies):
        """Role name -> the client that plays it. Which provider sits behind a
        seat is configuration; the orchestrator never asks."""
        return {"researcher": client(*researcher_replies),
                "engineer": client(*engineer_replies)}

    def test_needs_at_least_two_agents(self):
        with self.assertRaises(ValueError):
            Orchestrator({"solo": client({})})

    def test_final_report_ends_the_run(self):
        ex = Orchestrator(self._agents(
            [msg("HYPOTHESIS", "the pool is exhausted by per-item lookups")],
            [msg("FINAL_REPORT", "fixed and verified",
                 evidence=[{"source": "tests", "observation": "12 passed"}], confidence=0.8)],
        )).run("t", "fix the latency")
        self.assertEqual(ex.termination_reason, TerminationReason.SUCCESS.value)
        self.assertIsNotNone(ex.final_report)
        self.assertEqual(ex.metrics["rounds"] if isinstance(ex.metrics, dict) else ex.metrics.rounds, 1)

    def test_restating_the_same_claim_stops_the_run(self):
        """Two agents agreeing in nicer words is not progress."""
        same = msg("CRITIQUE", "we should look into the database")
        ex = Orchestrator(self._agents([same] * 6, [same] * 6), max_rounds=4).run(
            "t", "fix the latency")
        self.assertEqual(ex.termination_reason, TerminationReason.NO_NEW_EVIDENCE.value)
        self.assertLess(ex.metrics["rounds"] if isinstance(ex.metrics, dict) else ex.metrics.rounds, 4)
        self.assertIn(FailureCategory.REPEATED_ACTION.value, ex.failure_categories)

    def test_max_rounds_is_enforced_when_evidence_keeps_arriving(self):
        replies = [msg("CRITIQUE", f"observation number {i}",
                       evidence=[{"source": "log", "observation": f"line {i}"}])
                   for i in range(1, 9)]
        ex = Orchestrator(self._agents(replies, replies), max_rounds=3).run("t", "investigate")
        self.assertEqual(ex.termination_reason, TerminationReason.MAX_ITERATIONS.value)
        self.assertEqual(ex.metrics["rounds"] if isinstance(ex.metrics, dict) else ex.metrics.rounds, 3)

    def test_out_of_remit_message_is_a_role_violation_and_is_not_delivered(self):
        ex = Orchestrator(self._agents(
            [msg("IMPLEMENTATION_PLAN", "I will edit the file myself")],
            [msg("BLOCKER", "waiting on the researcher")],
        ), max_rounds=1).run("t", "fix it")
        role_violations = [v for v in ex.violations
                           if v["category"] == FailureCategory.ROLE_VIOLATION.value]
        self.assertTrue(role_violations)
        self.assertIn("IMPLEMENTATION_PLAN", role_violations[0]["problem"])
        # the offending artifact never reaches the other agent
        self.assertNotIn(MessageType.IMPLEMENTATION_PLAN, [m.type for m in ex.messages])

    def test_schema_violation_is_recorded_and_the_message_is_not_delivered(self):
        """A confident claim with no evidence never reaches the other agent."""
        ex = Orchestrator(self._agents(
            [msg("HYPOTHESIS", "certain it is the pool", confidence=0.99)],  # no evidence
            [msg("BLOCKER", "waiting")],
        ), max_rounds=1).run("t", "fix it")
        self.assertTrue(ex.violations)
        self.assertNotIn(MessageType.HYPOTHESIS, [m.type for m in ex.messages])

    def test_unparseable_reply_is_recorded_not_crashed(self):
        ex = Orchestrator(self._agents(
            ["I think it's probably the database"],
            [msg("BLOCKER", "waiting")],
        ), max_rounds=1).run("t", "fix it")
        self.assertTrue(ex.violations)

    def test_round_cap_cannot_be_removed(self):
        """max_rounds=0 would mean unbounded or never-starting; neither is allowed."""
        o = Orchestrator(self._agents([msg("BLOCKER", "x")], [msg("BLOCKER", "y")]),
                         max_rounds=0)
        self.assertGreaterEqual(o.max_rounds, 1)

    def test_metrics_cover_the_spec_section_27_fields(self):
        ex = Orchestrator(self._agents(
            [msg("HYPOTHESIS", "pool exhaustion from per-item lookups")],
            [msg("IMPLEMENTATION_PLAN", "add a cache",
                 evidence=[{"source": "code", "observation": "lookup per item"}])],
        ), max_rounds=1).run("t", "fix it")
        m = ex.metrics if isinstance(ex.metrics, dict) else ex.metrics.to_dict()
        for field_name in ("rounds", "handoffs", "disagreements",
                           "unresolved_disagreements", "protocol_violations",
                           "duplicate_work", "evidence_items"):
            self.assertIn(field_name, m)
        self.assertEqual(m["rounds"], 1)

    def test_conflicting_hypotheses_are_resolved_by_evidence(self):
        ex = Orchestrator(self._agents(
            [msg("HYPOTHESIS", "the connection pool is exhausted", confidence=0.6,
                 evidence=[{"source": "metrics", "observation": "pool 100/100"}])],
            [msg("HYPOTHESIS", "a database index was dropped during the migration",
                 confidence=0.4)],
        ), max_rounds=1).run("t", "diagnose the latency")
        self.assertEqual((ex.metrics["disagreements"] if isinstance(ex.metrics, dict) else ex.metrics.disagreements), 1)
        record = ex.disagreements[0]
        resolution = record.get("resolution") or record.get("outcome", {}).get("resolution")
        self.assertEqual(resolution, Resolution.RESOLVED_BY_EVIDENCE.value)
        self.assertEqual((ex.metrics["unresolved_disagreements"] if isinstance(ex.metrics, dict) else ex.metrics.unresolved_disagreements), 0)

    def test_unresolved_disagreement_is_labelled(self):
        """An exchange that ended without settling a conflict says so."""
        judge = client({"discriminative": False, "experiment": "",
                        "reasoning": "nothing distinguishes them"})
        ex = Orchestrator(self._agents(
            [msg("HYPOTHESIS", "the connection pool is exhausted", confidence=0.4)],
            [msg("HYPOTHESIS", "a database index was dropped in migration",
                 confidence=0.4)],
        ), max_rounds=1, judge=judge).run("t", "diagnose")
        self.assertEqual((ex.metrics["disagreements"] if isinstance(ex.metrics, dict) else ex.metrics.disagreements), 1)
        self.assertEqual((ex.metrics["unresolved_disagreements"] if isinstance(ex.metrics, dict) else ex.metrics.unresolved_disagreements), 1)
        self.assertIn(FailureCategory.AGENT_DISAGREEMENT.value, ex.failure_categories)

    def test_exchange_serialises_and_saves(self):
        ex = Orchestrator(self._agents(
            [msg("QUESTION", "which service regressed?")],
            [msg("BLOCKER", "no access to the metrics")],
        ), max_rounds=1).run("t", "investigate")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_exchange(ex, Path(tmp) / "exchange.json")
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertIn("metrics", data)
            self.assertEqual(data["task_id"], "t")


if __name__ == "__main__":
    unittest.main()
