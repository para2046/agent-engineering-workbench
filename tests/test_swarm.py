"""Swarm queue tests.

The property everything rests on is the atomic claim: two workers racing for
one ticket must produce exactly one winner, with no locks and no coordinator.
The second property is the honesty boundary: workers' success claims decide
nothing, and dependencies gate on the host's verdict alone.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from agentwb.swarm.queue import SwarmQueue, Ticket


def ticket(task_id, after=None):
    return Ticket(task_id=task_id, task_file=f"{task_id}.json",
                  workspace=f"ws/{task_id}", after=after or [])


class SwarmCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.queue = SwarmQueue(Path(self._tmp.name), lease_seconds=1.0)

    def tearDown(self):
        self._tmp.cleanup()


class TestAtomicClaims(SwarmCase):
    def test_a_claim_moves_the_ticket_out_of_open(self):
        self.queue.seed(ticket("a"))
        claimed = self.queue.claim("w1")
        self.assertEqual(claimed.task_id, "a")
        self.assertIsNone(self.queue.claim("w2"), "the ticket must be gone")

    def test_racing_workers_get_exactly_one_winner_per_ticket(self):
        """The core guarantee, tested with a genuine thread race."""
        for i in range(6):
            self.queue.seed(ticket(f"t{i}"))
        wins: list[str] = []
        lock = threading.Lock()

        def worker(name):
            while True:
                t = self.queue.claim(name)
                if t is None:
                    return
                with lock:
                    wins.append(t.task_id)

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(wins), [f"t{i}" for i in range(6)],
                         "every ticket claimed exactly once, none twice, none lost")

    def test_oldest_ticket_is_claimed_first(self):
        self.queue.seed(ticket("first"))
        time.sleep(0.05)
        self.queue.seed(ticket("second"))
        self.assertEqual(self.queue.claim("w").task_id, "first")


class TestLeases(SwarmCase):
    def test_a_dead_workers_claim_is_requeued_after_the_lease(self):
        self.queue.seed(ticket("a"))
        self.queue.claim("dead_worker")
        self.assertEqual(self.queue.requeue_stale_claims(), [])   # lease still live
        time.sleep(1.2)
        self.assertEqual(self.queue.requeue_stale_claims(), ["a"])
        self.assertEqual(self.queue.claim("survivor").task_id, "a")

    def test_heartbeat_extends_the_lease(self):
        self.queue.seed(ticket("a"))
        self.queue.claim("w1")
        time.sleep(0.7)
        self.assertTrue(self.queue.heartbeat("a", "w1"))
        time.sleep(0.7)                       # total > lease, but heartbeat reset it
        self.assertEqual(self.queue.requeue_stale_claims(), [])

    def test_heartbeat_after_requeue_reports_the_loss(self):
        """A worker that slept through its lease must learn it lost the claim."""
        self.queue.seed(ticket("a"))
        self.queue.claim("w1")
        time.sleep(1.2)
        self.queue.requeue_stale_claims()
        self.assertFalse(self.queue.heartbeat("a", "w1"))


class TestDependencies(SwarmCase):
    def test_dependent_work_starts_blocked(self):
        self.queue.seed(ticket("c", after=["a"]))
        self.assertIsNone(self.queue.claim("w"))

    def test_a_workers_success_claim_does_not_unblock_anything(self):
        """The honesty boundary at swarm scale: done/ is a belief, graded/ is
        the verdict, and dependencies gate on the verdict alone."""
        self.queue.seed(ticket("a"))
        self.queue.seed(ticket("c", after=["a"]))
        self.queue.claim("w1")
        self.queue.submit("a", "w1", "definitely finished", claims_success=True)
        self.assertEqual(self.queue.release_unblocked(), [],
                         "an ungraded submission must unblock nothing")

    def test_a_host_pass_unblocks_dependants(self):
        self.queue.seed(ticket("c", after=["a"]))
        self.queue.record_verdict("a", passed=True)
        self.assertEqual(self.queue.release_unblocked(), ["c"])
        self.assertEqual(self.queue.claim("w").task_id, "c")

    def test_a_host_fail_keeps_dependants_blocked(self):
        self.queue.seed(ticket("c", after=["a"]))
        self.queue.record_verdict("a", passed=False)
        self.assertEqual(self.queue.release_unblocked(), [])

    def test_multiple_prerequisites_all_required(self):
        self.queue.seed(ticket("c", after=["a", "b"]))
        self.queue.record_verdict("a", passed=True)
        self.assertEqual(self.queue.release_unblocked(), [])
        self.queue.record_verdict("b", passed=True)
        self.assertEqual(self.queue.release_unblocked(), ["c"])


class TestSubmissionsAndStatus(SwarmCase):
    def test_submit_clears_the_claim_and_queues_for_grading(self):
        self.queue.seed(ticket("a"))
        self.queue.claim("w1")
        self.queue.submit("a", "w1", "report text")
        pending = self.queue.pending_grades()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["worker"], "w1")
        self.assertFalse(self.queue.heartbeat("a", "w1"))

    def test_graded_work_leaves_the_pending_list(self):
        self.queue.seed(ticket("a"))
        self.queue.claim("w1")
        self.queue.submit("a", "w1", "r")
        self.queue.record_verdict("a", passed=False)
        self.assertEqual(self.queue.pending_grades(), [])
        self.assertIs(self.queue.verdict_of("a"), False)

    def test_status_counts_are_accurate(self):
        self.queue.seed(ticket("a"))
        self.queue.seed(ticket("b"))
        self.queue.seed(ticket("c", after=["a"]))
        self.queue.claim("w1")
        status = self.queue.status()
        self.assertEqual((status["open"], status["claimed"], status["blocked"]),
                         (1, 1, 1))

    def test_close_is_visible_to_workers(self):
        self.assertFalse(self.queue.closed)
        self.queue.close()
        self.assertTrue(self.queue.closed)


if __name__ == "__main__":
    unittest.main()
