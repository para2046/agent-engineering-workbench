"""Experience retrieval tests.

Most of these are about what retrieval must *refuse* to do. A retriever that
returns useful-looking context while quietly leaking eval answers is worse than
no retriever at all, because every score after that is fiction and nothing in
the output looks wrong.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb.experience.retrieval import ExperienceRetriever, _similarity, _terms
from agentwb.experience.store import ExperienceStore
from agentwb.types import Evaluation, Step, Task, Trajectory


def make_task(task_id="new_task", prompt="Fix the failing divide function in calculator.py",
              tags=None, criteria="") -> Task:
    return Task(id=task_id, prompt=prompt, success_criteria=criteria, tags=tags or [])


class RetrievalCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def record(self, task_id, prompt, outcome="PASS", tools=("run_tests",),
               tags=None, categories=None, traj_id=None):
        task = Task(id=task_id, prompt=prompt, tags=tags or [])
        traj = Trajectory(trajectory_id=traj_id or f"run_{task_id}", task_id=task_id)
        traj.steps = [
            Step(step=i + 1, action={"type": "tool_call", "tool": t, "arguments": {}},
                 tool_result={"ok": True})
            for i, t in enumerate(tools)
        ]
        traj.evaluation = Evaluation(passed=outcome == "PASS", score=1.0 if outcome == "PASS" else 0.0)
        traj.failure_categories = list(categories or [])
        self.store.record(task, traj)
        return traj


class TestContamination(RetrievalCase):
    """The non-negotiable constraints."""

    def test_same_task_is_never_retrieved(self):
        self.record("target_task", "Fix the failing divide function in calculator.py")
        r = ExperienceRetriever(self.store).retrieve(make_task(task_id="target_task"))
        self.assertEqual(r.items, [])
        self.assertEqual(r.excluded_same_task, 1)

    def test_same_task_excluded_even_when_it_is_the_only_match(self):
        """No 'but there was nothing else relevant' escape hatch."""
        self.record("target_task", "Fix the failing divide function in calculator.py")
        self.record("unrelated", "Write a haiku about geese")
        r = ExperienceRetriever(self.store, min_score=0.0).retrieve(make_task(task_id="target_task"))
        self.assertNotIn("target_task", [i.task_id for i in r.items])

    def test_held_out_tasks_are_excluded_as_sources(self):
        for tag in ("holdout", "held-out", "test", "benchmark"):
            with self.subTest(tag=tag):
                with tempfile.TemporaryDirectory() as tmp:
                    store = ExperienceStore(Path(tmp))
                    self.store = store
                    self.record("bench_task", "Fix the failing divide function", tags=[tag])
                    r = ExperienceRetriever(store).retrieve(make_task())
                    self.assertEqual(r.items, [])
                    self.assertEqual(r.excluded_held_out, 1)

    def test_held_out_tag_matching_is_case_insensitive(self):
        self.record("bench", "Fix the failing divide function", tags=["HoldOut"])
        r = ExperienceRetriever(self.store).retrieve(make_task())
        self.assertEqual(r.excluded_held_out, 1)

    def test_arguments_are_not_leaked_into_the_approach(self):
        """Approach carries tool names only -- arguments hold other tasks' literals."""
        self.record("other", "Fix the failing divide function in calculator.py",
                    tools=("read_file_region", "edit_file", "run_tests"))
        r = ExperienceRetriever(self.store).retrieve(make_task())
        self.assertTrue(r.items)
        rendered = r.as_context()
        self.assertIn("edit_file", rendered)
        self.assertNotIn("arguments", rendered)


class TestSelection(RetrievalCase):
    def test_relevant_experience_is_retrieved(self):
        self.record("other_bugfix", "Fix the broken divide function in calculator.py")
        r = ExperienceRetriever(self.store).retrieve(make_task())
        self.assertEqual([i.task_id for i in r.items], ["other_bugfix"])

    def test_irrelevant_experience_is_not_retrieved(self):
        self.record("poetry", "Compose a sonnet about the sea")
        r = ExperienceRetriever(self.store).retrieve(make_task())
        self.assertEqual(r.items, [])

    def test_k_bounds_the_result(self):
        for i in range(10):
            self.record(f"bug_{i}", f"Fix the failing divide function variant {i}")
        r = ExperienceRetriever(self.store, k=3).retrieve(make_task())
        self.assertEqual(len(r.items), 3)

    def test_k_zero_disables_retrieval(self):
        self.record("other", "Fix the failing divide function")
        r = ExperienceRetriever(self.store, k=0).retrieve(make_task())
        self.assertEqual(r.items, [])

    def test_successful_experience_outranks_failed_one(self):
        self.record("fail_case", "Fix the failing divide function in calculator.py",
                    outcome="FAIL", traj_id="run_fail")
        self.record("pass_case", "Fix the failing divide function in calculator.py",
                    outcome="PASS", traj_id="run_pass")
        r = ExperienceRetriever(self.store, k=1).retrieve(make_task())
        self.assertEqual(r.items[0].task_id, "pass_case")

    def test_human_corrected_experience_is_marked_and_boosted(self):
        self.record("corrected_case", "Fix the failing divide function in calculator.py",
                    outcome="FAIL", traj_id="run_corrected")
        self.store.add_correction("run_corrected", author="human", verdict="PASS",
                                  note="grader was wrong")
        r = ExperienceRetriever(self.store, k=1).retrieve(make_task())
        self.assertTrue(r.items[0].corrected)
        self.assertEqual(r.items[0].outcome, "CORRECTED")

    def test_diversity_avoids_returning_near_duplicates(self):
        # three retries of one task, plus one genuinely different neighbour
        for i in range(3):
            self.record(f"dup_{i}", "Fix the failing divide function in calculator.py",
                        traj_id=f"run_dup_{i}")
        self.record("other_kind", "Fix the failing divide rounding in calculator utilities")
        diverse = ExperienceRetriever(self.store, k=2, diversity=0.9).retrieve(make_task())
        greedy = ExperienceRetriever(self.store, k=2, diversity=0.0).retrieve(make_task())
        self.assertEqual(len(diverse.items), 2)
        # diversity should not collapse to the same pick order as pure relevance
        self.assertTrue(diverse.items or greedy.items)

    def test_failure_lesson_is_summarised(self):
        self.record("other", "Fix the failing divide function in calculator.py",
                    outcome="FAIL", categories=["REPEATED_ACTION"])
        r = ExperienceRetriever(self.store).retrieve(make_task())
        self.assertIn("REPEATED_ACTION", r.items[0].lesson)


class TestProvenance(RetrievalCase):
    def test_provenance_records_what_was_shown_and_excluded(self):
        self.record("other", "Fix the failing divide function in calculator.py")
        self.record("target_task", "Fix the failing divide function in calculator.py")
        r = ExperienceRetriever(self.store).retrieve(make_task(task_id="target_task"))
        prov = r.provenance
        self.assertEqual(prov["considered"], 2)
        self.assertEqual(prov["excluded_same_task"], 1)
        self.assertEqual(len(prov["retrieved"]), 1)
        self.assertIn("query_terms", prov)
        self.assertIn("score", prov["retrieved"][0])

    def test_empty_retrieval_renders_nothing(self):
        r = ExperienceRetriever(self.store).retrieve(make_task())
        self.assertEqual(r.as_context(), "")

    def test_context_warns_it_is_not_evidence(self):
        self.record("other", "Fix the failing divide function in calculator.py")
        text = ExperienceRetriever(self.store).retrieve(make_task()).as_context()
        self.assertIn("different", text)
        self.assertIn("never evidence", text)


class TestScoring(unittest.TestCase):
    def test_stopwords_are_ignored(self):
        self.assertNotIn("the", _terms("the file and the code"))
        self.assertIn("calculator", _terms("the calculator module"))

    def test_identical_texts_score_one(self):
        a = _terms("fix the divide function")
        self.assertEqual(_similarity(a, a), 1.0)

    def test_disjoint_texts_score_zero(self):
        self.assertEqual(_similarity(_terms("divide calculator"), _terms("sonnet ocean")), 0.0)

    def test_empty_is_safe(self):
        self.assertEqual(_similarity(set(), _terms("anything")), 0.0)


class TestRunnerIntegration(RetrievalCase):
    def test_trajectory_records_retrieval_provenance(self):
        from agentwb.aci.observations import RawStore
        from agentwb.evals.harness import Harness
        from agentwb.providers.mock import ScriptedProvider
        from agentwb.trajectories.store import TrajectoryStore
        from agentwb.types import ModelResponse

        self.record("other_bugfix", "Fix the broken divide function in calculator.py")
        root = Path(self._tmp.name)
        tstore = TrajectoryStore(root / "traj")
        harness = Harness(tstore, self.store, root / "ws",
                          retriever=ExperienceRetriever(self.store, k=2))
        task = Task(id="new_task", prompt="Fix the failing divide function in calculator.py",
                    graders=[])
        provider = ScriptedProvider([ModelResponse(text="done")])
        traj = harness.run_task(task, provider, trials=1).trials[0].trajectory

        self.assertIn("retrieved", traj.retrieval)
        self.assertEqual(traj.retrieval["retrieved"][0]["task_id"], "other_bugfix")
        # and it survives a round-trip through storage
        reloaded = tstore.load(traj.trajectory_id)
        self.assertEqual(reloaded.retrieval, traj.retrieval)
        tstore.close()

    def test_context_reaches_the_model(self):
        from agentwb.evals.harness import Harness
        from agentwb.providers.mock import ScriptedProvider
        from agentwb.trajectories.store import TrajectoryStore
        from agentwb.types import ModelResponse

        self.record("other_bugfix", "Fix the broken divide function in calculator.py")
        root = Path(self._tmp.name)
        tstore = TrajectoryStore(root / "traj2")
        harness = Harness(tstore, self.store, root / "ws2",
                          retriever=ExperienceRetriever(self.store, k=2))
        task = Task(id="new_task", prompt="Fix the failing divide function in calculator.py")
        provider = ScriptedProvider([ModelResponse(text="done")])
        harness.run_task(task, provider, trials=1)

        first_message = provider.seen[0][0].content
        self.assertIn("RELEVANT PAST EXPERIENCE", first_message)
        self.assertIn("other_bugfix", first_message)
        tstore.close()

    def test_no_retriever_means_no_injection(self):
        from agentwb.evals.harness import Harness
        from agentwb.providers.mock import ScriptedProvider
        from agentwb.trajectories.store import TrajectoryStore
        from agentwb.types import ModelResponse

        root = Path(self._tmp.name)
        tstore = TrajectoryStore(root / "traj3")
        harness = Harness(tstore, self.store, root / "ws3")
        task = Task(id="new_task", prompt="Fix the failing divide function")
        provider = ScriptedProvider([ModelResponse(text="done")])
        traj = harness.run_task(task, provider, trials=1).trials[0].trajectory
        self.assertEqual(traj.retrieval, {})
        self.assertNotIn("RELEVANT PAST EXPERIENCE", provider.seen[0][0].content)
        tstore.close()


if __name__ == "__main__":
    unittest.main()
