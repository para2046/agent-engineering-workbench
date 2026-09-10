"""Research mode, adversarial search, failure clustering, and concurrent trials.

Three of these four have a rule that is the whole point of the module, and the
tests concentrate there:

* research ranks on evidence, never on stated confidence;
* adversarial search only counts a failure if the variant stayed solvable;
* concurrent trials refuse a provider that cannot be safely shared.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agentwb.adversarial.search import Mutation, generate_variants, mutate, search
from agentwb.evals.harness import Harness, load_task
from agentwb.experience.failure_cluster import cluster, signature_of, summarize
from agentwb.experience.store import ExperienceStore
from agentwb.judge.client import JudgeClient
from agentwb.judge.mock import FakeJudgeProvider
from agentwb.protocols.schemas import Evidence, Verification
from agentwb.providers.mock import RuleProvider
from agentwb.research.mode import Belief, Hypothesis, investigate, rank, update_belief
from agentwb.trajectories.store import TrajectoryStore
from agentwb.types import ModelResponse, Task, Usage


def hyp(claim, evidence=None, confidence=0.5, unknowns=None, falsifiable=True) -> Hypothesis:
    return Hypothesis(
        claim=claim,
        evidence=evidence or [],
        unknowns=unknowns or [],
        stated_confidence=confidence,
        verification=Verification("m", "a", "b") if falsifiable else None,
        hypothesis_id=claim[:8],
    )


def client(*replies) -> JudgeClient:
    return JudgeClient(FakeJudgeProvider(list(replies)))


# --------------------------------------------------------------------------
# research: ranking
# --------------------------------------------------------------------------

class TestHypothesisRanking(unittest.TestCase):
    def test_evidence_beats_confidence(self):
        """The rule the whole module exists for: a confident claim with nothing
        behind it must lose to a modest one with observations."""
        confident = hyp("it is the cache", confidence=0.99)
        grounded = hyp("it is the pool", confidence=0.3,
                       evidence=[Evidence("metrics", "pool 100/100"),
                                 Evidence("run_tests", "3 failed")])
        ranked = rank([confident, grounded])
        self.assertEqual(ranked[0].claim, "it is the pool")
        self.assertIs(ranked[0].belief, Belief.LEADING)

    def test_environment_evidence_outranks_cited_claims(self):
        grounded = hyp("a", evidence=[Evidence("run_tests", "x")])
        hearsay = hyp("b", evidence=[Evidence("cited", "x")])
        self.assertEqual(rank([hearsay, grounded])[0].claim, "a")

    def test_a_falsifiable_hypothesis_outranks_an_unfalsifiable_one(self):
        checkable = hyp("a", falsifiable=True)
        vague = hyp("b", falsifiable=False)
        self.assertEqual(rank([vague, checkable])[0].claim, "a")

    def test_overconfidence_without_evidence_is_penalised(self):
        modest = hyp("a", confidence=0.4)
        overconfident = hyp("b", confidence=0.99)
        ranked = rank([overconfident, modest])
        self.assertLess(ranked[-1].score_breakdown["overconfidence"], 0)
        self.assertEqual(ranked[0].claim, "a")

    def test_admitting_unknowns_helps(self):
        honest = hyp("a", unknowns=["why the lookup is slow"])
        silent = hyp("b")
        self.assertEqual(rank([silent, honest])[0].claim, "a")

    def test_repeating_one_observation_is_not_more_evidence(self):
        once = hyp("a", evidence=[Evidence("run_tests", "same fact")])
        many = hyp("b", evidence=[Evidence("run_tests", "same fact")] * 5)
        ranked = rank([once, many])
        self.assertEqual(ranked[0].score_breakdown["evidence"],
                         ranked[1].score_breakdown["evidence"])

    def test_ranking_is_stable_on_ties(self):
        a, b = hyp("alpha"), hyp("beta")
        self.assertEqual([h.claim for h in rank([b, a])],
                         [h.claim for h in rank([a, b])])

    def test_established_belief_survives_reranking(self):
        h = hyp("a")
        h.belief = Belief.ESTABLISHED
        self.assertIs(rank([h, hyp("b")])[0].belief, Belief.ESTABLISHED)


class TestBeliefUpdate(unittest.TestCase):
    def test_new_evidence_permits_an_update(self):
        self.assertIn("updated", update_belief([], new_evidence=2))

    def test_no_new_evidence_blocks_an_update(self):
        """Re-reasoning over the same observations is not an update."""
        note = update_belief([], new_evidence=0)
        self.assertIn("belief unchanged", note)
        self.assertIn("not an update", note)


class TestInvestigate(unittest.TestCase):
    def _generator(self, *claims):
        return {"hypotheses": [
            {"claim": c, "supported_by": [], "unknowns": ["u"], "confidence": 0.5,
             "verification": {"metric": "m", "expected_if_correct": "x",
                              "expected_if_wrong": "y"}}
            for c in claims]}

    def test_an_experiment_can_establish_a_hypothesis(self):
        judge = client(
            self._generator("the pool is exhausted", "an index was dropped"),
            {"discriminative": True, "experiment": "check the pool", "metric": "m",
             "expected_if_a": "saturated", "expected_if_b": "idle"},
            {"supports": "A", "reasoning": "the pool reads saturated"},
        )
        result = investigate("why is checkout slow?", judge,
                             run_experiment=lambda c: (True, "pool 100/100"),
                             max_rounds=1)
        self.assertTrue(result.established)
        self.assertEqual(result.termination_reason, "ESTABLISHED_BY_EXPERIMENT")
        self.assertIs(result.conclusion.belief, Belief.ESTABLISHED)

    def test_without_an_experiment_runner_nothing_is_established(self):
        """It reports a leader and says it is not established, rather than
        promoting it."""
        judge = client(self._generator("a", "b"))
        result = investigate("q", judge, max_rounds=1)
        self.assertFalse(result.established)
        self.assertIn("LEADING (not established)", result.summary())

    def test_a_round_with_no_new_evidence_stops_the_loop(self):
        judge = client(self._generator("a", "b"))
        result = investigate("q", judge, max_rounds=5)
        self.assertEqual(result.termination_reason, "NO_NEW_EVIDENCE")
        self.assertEqual(len(result.rounds), 1)

    def test_no_hypotheses_is_reported_not_crashed(self):
        result = investigate("q", client("not json"), max_rounds=2)
        self.assertEqual(result.termination_reason, "NO_HYPOTHESES")
        self.assertIsNone(result.conclusion)


# --------------------------------------------------------------------------
# adversarial search
# --------------------------------------------------------------------------

class AdversarialCase(unittest.TestCase):
    def setUp(self):
        self.task = load_task(Path("tasks/fix_divide_bug.json"))


class TestMutations(AdversarialCase):
    def test_every_mutation_preserves_the_success_criteria(self):
        """Changing the goalposts would make any 'failure' meaningless."""
        for variant in generate_variants(self.task):
            with self.subTest(mutation=variant.mutation):
                self.assertEqual(variant.task.success_criteria, self.task.success_criteria)

    def test_every_mutation_preserves_the_graders(self):
        for variant in generate_variants(self.task):
            with self.subTest(mutation=variant.mutation):
                self.assertEqual([g.type for g in variant.task.graders],
                                 [g.type for g in self.task.graders])

    def test_variants_are_tagged_adversarial(self):
        for variant in generate_variants(self.task):
            self.assertIn("adversarial", variant.task.tags)

    def test_variant_ids_record_their_parent(self):
        variant = mutate(self.task, Mutation.WEAKEN_PROMPT)
        self.assertEqual(variant.parent_task_id, self.task.id)
        self.assertTrue(variant.task.id.startswith(self.task.id))

    def test_mutation_is_deterministic(self):
        a = mutate(self.task, Mutation.ADD_DISTRACTOR_FILES)
        b = mutate(self.task, Mutation.ADD_DISTRACTOR_FILES)
        self.assertEqual(a.task.environment.files, b.task.environment.files)

    def test_every_mutation_carries_a_rationale(self):
        for variant in generate_variants(self.task):
            self.assertTrue(variant.rationale, variant.mutation)

    def test_an_inapplicable_mutation_returns_none(self):
        tiny = Task(id="t", prompt="p", max_steps=3)
        self.assertIsNone(mutate(tiny, Mutation.TIGHTEN_BUDGET))
        self.assertIsNone(mutate(tiny, Mutation.ADD_DISTRACTOR_FILES))


class TestSearch(AdversarialCase):
    def test_a_broken_but_solvable_variant_is_a_finding(self):
        result = search(self.task, run=lambda t: False, check_solvable=lambda t: True)
        self.assertTrue(result.findings)
        self.assertTrue(all(v.is_finding for v in result.variants))

    def test_an_unsolvable_variant_is_discarded_not_reported(self):
        """Breaking an agent with an impossible task is not a discovery, and
        keeping it would poison the regression suite."""
        result = search(self.task, run=lambda t: False, check_solvable=lambda t: False)
        self.assertEqual(result.findings, [])
        self.assertEqual(result.variants, [])
        self.assertTrue(result.skipped_unsolvable)

    def test_a_variant_the_agent_survives_is_not_a_finding(self):
        result = search(self.task, run=lambda t: True, check_solvable=lambda t: True)
        self.assertEqual(result.findings, [])
        self.assertTrue(result.variants)

    def test_without_a_solvability_check_findings_are_flagged_as_leads(self):
        result = search(self.task, run=lambda t: False)
        self.assertTrue(result.notes)
        self.assertIn("lead, not a confirmed weakness", result.notes[0])

    def test_search_can_be_restricted_to_named_mutations(self):
        result = search(self.task, run=lambda t: True, check_solvable=lambda t: True,
                        mutations=[Mutation.WEAKEN_PROMPT])
        self.assertEqual([v.mutation for v in result.variants], [Mutation.WEAKEN_PROMPT])


# --------------------------------------------------------------------------
# failure clustering
# --------------------------------------------------------------------------

def experience(task_id="t", outcome="FAIL", categories=None, tools=None,
               error=None, termination="AGENT_FINISHED") -> dict:
    actions = [{"tool": t, "ok": error is None} for t in (tools or ["run_tests"])]
    if error:
        actions[-1] = {"tool": actions[-1]["tool"], "ok": False, "error": error}
    return {
        "trajectory_id": f"run_{task_id}_{outcome}_{len(actions)}",
        "outcome": outcome,
        "task": {"id": task_id, "prompt": "p"},
        "failure_categories": categories or [],
        "actions": actions,
        "metrics": {"termination_reason": termination},
    }


class TestFailureClustering(unittest.TestCase):
    def test_identical_failures_group_together(self):
        rows = [experience(categories=["REPEATED_ACTION"]) for _ in range(3)]
        clusters = cluster(rows)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 3)

    def test_different_failures_stay_apart(self):
        rows = [experience(categories=["REPEATED_ACTION"]),
                experience(categories=["GRADER_FAILURE"])]
        self.assertEqual(len(cluster(rows)), 2)

    def test_passes_are_excluded_by_default(self):
        self.assertEqual(cluster([experience(outcome="PASS")]), [])

    def test_a_cluster_spanning_tasks_is_flagged(self):
        """Those point at the agent or the tools, not at one task."""
        rows = [experience(task_id="a", categories=["REPEATED_ACTION"]),
                experience(task_id="b", categories=["REPEATED_ACTION"])]
        self.assertTrue(cluster(rows)[0].spans_tasks)

    def test_tool_error_codes_separate_signatures(self):
        rows = [experience(tools=["edit_file"], error="NO_MATCH"),
                experience(tools=["edit_file"], error="AMBIGUOUS_MATCH")]
        self.assertEqual(len(cluster(rows)), 2)

    def test_signature_describes_itself(self):
        sig = signature_of(experience(categories=["REPEATED_ACTION"], tools=["list_files"]))
        text = sig.describe()
        self.assertIn("REPEATED_ACTION", text)
        self.assertIn("never verified", text)

    def test_summary_reports_concentration(self):
        rows = [experience(categories=["A"]) for _ in range(9)] + \
               [experience(categories=["B"])]
        stats = summarize(cluster(rows))
        self.assertEqual(stats["failures"], 10)
        self.assertEqual(stats["clusters"], 2)
        self.assertEqual(stats["concentration"], 0.9)

    def test_clusters_are_ordered_largest_first(self):
        rows = [experience(categories=["A"])] + [experience(categories=["B"])] * 3
        self.assertEqual(cluster(rows)[0].size, 3)


# --------------------------------------------------------------------------
# concurrent trials
# --------------------------------------------------------------------------

class StatelessProvider:
    name, model = "stateless", "stateless-v1"

    def generate(self, system, messages, tools):
        time.sleep(0.02)
        return ModelResponse(text="done", usage=Usage(10, 10))


class TestConcurrentTrials(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.store = TrajectoryStore(root)
        self.harness = Harness(self.store, ExperienceStore(root / "e"), root / "w")
        self.task = load_task(Path("tasks/fix_divide_bug.json"))

    def tearDown(self):
        self.store.close()
        try:
            self._tmp.cleanup()
        except (PermissionError, OSError):
            pass          # Windows holds the sqlite handle briefly

    def test_a_stateful_provider_is_refused_rather_than_silently_wrong(self):
        """Sharing per-run state across threads produced 0/4 where serial gave
        4/4. A quietly wrong success rate is worse than no parallelism."""
        with self.assertRaises(ValueError) as cm:
            self.harness.run_task(self.task, RuleProvider(), trials=4, concurrency=4)
        self.assertIn("per-run state", str(cm.exception))

    def test_a_stateless_provider_runs_concurrently(self):
        result = self.harness.run_task(self.task, StatelessProvider(),
                                       trials=4, concurrency=4)
        self.assertEqual(len(result.trials), 4)

    def test_trial_order_is_deterministic_despite_completion_order(self):
        """A trial list that reshuffles between runs makes `compare` undiffable."""
        result = self.harness.run_task(self.task, StatelessProvider(),
                                       trials=4, concurrency=4)
        self.assertEqual([t.trajectory.trial for t in result.trials], [0, 1, 2, 3])

    def test_every_concurrent_trial_reaches_the_index(self):
        self.harness.run_task(self.task, StatelessProvider(), trials=4, concurrency=4)
        self.assertEqual(len(self.store.query()), 4)

    def test_concurrency_of_one_still_works(self):
        result = self.harness.run_task(self.task, RuleProvider(), trials=2, concurrency=1)
        self.assertEqual(len(result.trials), 2)


if __name__ == "__main__":
    unittest.main()
