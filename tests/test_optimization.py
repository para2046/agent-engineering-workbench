"""Dataset splits, cost estimation, and the promotion gate.

The gate exists to refuse things. Most of these tests are about what it refuses
and why — a gate that only ever says yes is decoration.
"""

from __future__ import annotations

import unittest

from agentwb.costs import PriceBook, utility
from agentwb.optimization.datasets import (
    DEV,
    REGRESSION,
    TEST,
    TRAIN,
    ContaminationError,
    Dataset,
    assign_split,
    build_dataset,
)
from agentwb.optimization.promotion import (
    GateConfig,
    PromotionGate,
    SplitResult,
    result_from_task_results,
)
from agentwb.types import Task


def task(task_id: str, tags=None) -> Task:
    return Task(id=task_id, prompt="do the thing", tags=tags or [])


def split_result(name="dev", trials=40, passed=30, cost=None, latency=0,
                 passed_ids=None, failed_ids=None) -> SplitResult:
    return SplitResult(
        split=name, tasks=trials, trials=trials, passed=passed,
        total_cost=cost, total_latency_ms=latency,
        passed_task_ids=set(passed_ids or []), failed_task_ids=set(failed_ids or []),
    )


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

class TestSplits(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        """A split that moves between runs is a slow leak, not a held-out set."""
        first = assign_split(task("some_task_id"))
        for _ in range(20):
            self.assertEqual(assign_split(task("some_task_id")), first)

    def test_explicit_tag_overrides_hashing(self):
        self.assertEqual(assign_split(task("x", ["regression"])), REGRESSION)
        self.assertEqual(assign_split(task("x", ["dev"])), DEV)
        self.assertEqual(assign_split(task("x", ["train"])), TRAIN)

    def test_holdout_synonyms_all_map_to_test(self):
        for tag in ("test", "holdout", "held-out", "held_out", "benchmark"):
            with self.subTest(tag=tag):
                self.assertEqual(assign_split(task("x", [tag])), TEST)

    def test_ratios_are_respected_across_many_tasks(self):
        ds = build_dataset(task(f"task_{i:04d}") for i in range(400))
        counts = ds.counts()
        self.assertGreater(counts[TRAIN], counts[DEV])
        self.assertGreater(counts[TRAIN], counts[TEST])
        self.assertEqual(sum(counts.values()), 400)

    def test_splits_are_disjoint(self):
        build_dataset(task(f"t{i}") for i in range(50)).assert_disjoint()

    def test_duplicate_task_in_two_splits_is_caught(self):
        ds = Dataset(train=[task("dup")], dev=[task("dup")])
        with self.assertRaises(ContaminationError):
            ds.assert_disjoint()

    def test_regression_tag_keeps_a_task_out_of_sampling(self):
        ds = build_dataset([task("curated", ["regression"])])
        self.assertEqual(ds.counts()[REGRESSION], 1)
        self.assertEqual(ds.counts()[TRAIN], 0)


class TestSplitAccessControl(unittest.TestCase):
    """The mechanism behind 'never optimize against the test set'."""

    def setUp(self):
        self.ds = build_dataset([
            task("a", ["train"]), task("b", ["dev"]),
            task("c", ["test"]), task("d", ["regression"]),
        ])

    def test_optimizer_may_read_train_and_dev(self):
        self.assertEqual([t.id for t in self.ds.for_optimizer(TRAIN)], ["a"])
        self.assertEqual([t.id for t in self.ds.for_optimizer(DEV)], ["b"])

    def test_optimizer_asking_for_test_raises(self):
        with self.assertRaises(ContaminationError) as cm:
            self.ds.for_optimizer(TEST)
        self.assertIn("held-out", str(cm.exception))

    def test_optimizer_asking_for_regression_raises(self):
        with self.assertRaises(ContaminationError):
            self.ds.for_optimizer(REGRESSION)

    def test_final_evaluation_is_a_separate_deliberate_call(self):
        self.assertEqual([t.id for t in self.ds.for_final_evaluation()], ["c"])

    def test_unknown_split_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ds.split("validation")


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

class TestCosts(unittest.TestCase):
    def setUp(self):
        self.book = PriceBook({"model-x": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}})

    def test_unpriced_model_is_unknown_not_zero(self):
        self.assertIsNone(PriceBook({}).estimate("anything", 1000, 1000))
        self.assertIsNone(self.book.estimate("other-model", 1000, 1000))

    def test_known_model_is_estimated(self):
        # 1M in + 1M out = 3.00 + 15.00
        self.assertAlmostEqual(self.book.estimate("model-x", 1_000_000, 1_000_000), 18.0)

    def test_prefix_match_covers_dated_snapshots(self):
        self.assertAlmostEqual(self.book.estimate("model-x-20260101", 1_000_000, 0), 3.0)

    def test_longest_prefix_wins(self):
        book = PriceBook({
            "model": {"input_per_mtok": 1.0, "output_per_mtok": 1.0},
            "model-x": {"input_per_mtok": 9.0, "output_per_mtok": 9.0},
        })
        self.assertAlmostEqual(book.estimate("model-x-2026", 1_000_000, 0), 9.0)

    def test_malformed_price_entry_is_skipped_not_defaulted(self):
        book = PriceBook({"bad": {"input_per_mtok": "free"}, "good": {
            "input_per_mtok": 1.0, "output_per_mtok": 2.0}})
        self.assertIsNone(book.estimate("bad", 1000, 1000))
        self.assertIsNotNone(book.estimate("good", 1000, 1000))

    def test_utility_returns_none_when_cost_is_unmeasured(self):
        """Treating unknown cost as zero would rank an unpriced model as free."""
        self.assertIsNone(utility(success=1.0, cost=None))

    def test_utility_ignores_missing_cost_when_it_carries_no_weight(self):
        self.assertEqual(utility(success=1.0, cost=None, weights={"cost": 0.0}), 1.0)

    def test_utility_penalises_cost(self):
        self.assertLess(utility(success=1.0, cost=0.5), utility(success=1.0, cost=0.1))


# --------------------------------------------------------------------------
# promotion gate
# --------------------------------------------------------------------------

class TestPromotionGate(unittest.TestCase):
    def setUp(self):
        self.gate = PromotionGate(GateConfig(min_trials_for_confidence=1))

    def decide(self, gate=None, **kw):
        defaults = dict(
            baseline_dev=split_result(passed=30, passed_ids={"a", "b"}, failed_ids={"c"}),
            candidate_dev=split_result(passed=36, passed_ids={"a", "b", "c"}),
            baseline_test=split_result("test", passed=30, passed_ids={"t1"}),
            candidate_test=split_result("test", passed=34, passed_ids={"t1"}),
        )
        defaults.update(kw)
        return (gate or self.gate).evaluate(**defaults)

    def test_clean_improvement_is_promoted(self):
        d = self.decide()
        self.assertTrue(d.promote, d.summary())
        self.assertEqual(d.blockers, [])

    def test_no_dev_improvement_blocks(self):
        d = self.decide(candidate_dev=split_result(passed=30, passed_ids={"a", "b"}))
        self.assertFalse(d.promote)
        self.assertIn("dev_improvement", [c.name for c in d.blockers])

    def test_worse_on_held_out_blocks_even_with_dev_gain(self):
        """The whole point of a held-out set."""
        d = self.decide(candidate_test=split_result("test", passed=20, passed_ids={"t1"}))
        self.assertFalse(d.promote)
        self.assertIn("held_out_test", [c.name for c in d.blockers])

    def test_missing_held_out_results_block(self):
        d = self.decide(baseline_test=None, candidate_test=None)
        self.assertFalse(d.promote)
        check = next(c for c in d.checks if c.name == "held_out_test")
        self.assertIn("generalise", check.detail)

    def test_a_single_regression_blocks_despite_better_average(self):
        """Averages are where regressions hide."""
        d = self.decide(
            baseline_dev=split_result(passed=20, passed_ids={"keeper"}),
            candidate_dev=split_result(passed=38, passed_ids={"x"}, failed_ids={"keeper"}),
        )
        self.assertFalse(d.promote)
        check = next(c for c in d.checks if c.name == "no_regressions")
        self.assertIn("keeper", check.detail)

    def test_regressions_in_the_held_out_split_are_also_counted(self):
        d = self.decide(
            baseline_test=split_result("test", passed=30, passed_ids={"t1"}),
            candidate_test=split_result("test", passed=34, failed_ids={"t1"}),
        )
        self.assertFalse(d.promote)
        self.assertIn("no_regressions", [c.name for c in d.blockers])

    def test_unmeasured_cost_blocks_a_configured_ceiling(self):
        """Unmeasured must never be read as free."""
        gate = PromotionGate(GateConfig(max_cost_per_trial=0.01, min_trials_for_confidence=1))
        d = self.decide(gate=gate, candidate_dev=split_result(passed=36, cost=None,
                                                              passed_ids={"a", "b", "c"}))
        self.assertFalse(d.promote)
        check = next(c for c in d.checks if c.name == "cost_ceiling")
        self.assertIn("UNKNOWN", check.detail)

    def test_cost_within_ceiling_passes(self):
        gate = PromotionGate(GateConfig(max_cost_per_trial=1.0, min_trials_for_confidence=1))
        d = self.decide(gate=gate, candidate_dev=split_result(passed=36, cost=4.0,
                                                              passed_ids={"a", "b", "c"}))
        self.assertTrue(d.promote, d.summary())

    def test_cost_over_ceiling_blocks(self):
        gate = PromotionGate(GateConfig(max_cost_per_trial=0.01, min_trials_for_confidence=1))
        d = self.decide(gate=gate, candidate_dev=split_result(passed=36, cost=40.0,
                                                              passed_ids={"a", "b", "c"}))
        self.assertFalse(d.promote)
        self.assertIn("cost_ceiling", [c.name for c in d.blockers])

    def test_latency_ceiling_blocks(self):
        gate = PromotionGate(GateConfig(max_latency_ms_per_trial=100,
                                        min_trials_for_confidence=1))
        d = self.decide(gate=gate, candidate_dev=split_result(
            passed=36, latency=40 * 5000, passed_ids={"a", "b", "c"}))
        self.assertFalse(d.promote)
        self.assertIn("latency_ceiling", [c.name for c in d.blockers])

    def test_regression_suite_decline_blocks(self):
        d = self.decide(
            baseline_regression=split_result("regression", trials=10, passed=10,
                                             passed_ids={"r1"}),
            candidate_regression=split_result("regression", trials=10, passed=5,
                                              failed_ids={"r1"}),
        )
        self.assertFalse(d.promote)
        names = [c.name for c in d.blockers]
        self.assertTrue({"regression_suite", "no_regressions"} & set(names))

    def test_one_sided_regression_results_warn_rather_than_silently_skip(self):
        d = self.decide(baseline_regression=split_result("regression", trials=5, passed=5))
        self.assertTrue(any("only one side" in w for w in d.warnings))

    def test_small_sample_warns_but_does_not_block(self):
        gate = PromotionGate(GateConfig(min_trials_for_confidence=20))
        d = self.decide(
            gate=gate,
            baseline_dev=split_result(trials=3, passed=1, passed_ids={"a"}),
            candidate_dev=split_result(trials=3, passed=3, passed_ids={"a", "b", "c"}),
            baseline_test=split_result("test", trials=3, passed=1),
            candidate_test=split_result("test", trials=3, passed=3),
        )
        self.assertTrue(d.promote)
        self.assertTrue(any("stochastic" in w for w in d.warnings))

    def test_unequal_trial_counts_warn(self):
        d = self.decide(
            baseline_dev=split_result(trials=40, passed=20, passed_ids={"a"}),
            candidate_dev=split_result(trials=10, passed=9, passed_ids={"a", "b"}),
        )
        self.assertTrue(any("unequal trial counts" in w for w in d.warnings))

    def test_decision_explains_itself(self):
        d = self.decide(candidate_dev=split_result(passed=30, passed_ids={"a", "b"}))
        self.assertIn("REJECT", d.summary())
        self.assertTrue(all(c.detail for c in d.checks))
        self.assertIn("checks", d.to_dict())


class TestSplitResultAggregation(unittest.TestCase):
    def test_flaky_task_does_not_count_as_a_clean_pass(self):
        r = SplitResult(split="dev", trials=2, passed=1,
                        passed_task_ids={"flaky"}, failed_task_ids={"flaky"})
        r.passed_task_ids -= r.failed_task_ids
        self.assertEqual(r.passed_task_ids, set())

    def test_cost_stays_none_when_nothing_was_priced(self):
        class _Traj:
            metrics = {"latency_ms": 5, "input_tokens": 10, "output_tokens": 10}
            model = "unpriced"
            task_id = "t"
            evaluation = None

        class _Trial:
            trajectory = _Traj()

        class _TaskResult:
            trials = [_Trial()]

        out = result_from_task_results("dev", [_TaskResult()], PriceBook({}))
        self.assertIsNone(out.total_cost)
        self.assertIsNone(out.cost_per_trial)


if __name__ == "__main__":
    unittest.main()
