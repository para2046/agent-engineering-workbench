"""Task room tests.

The properties under test are the ones the room exists for: the board only
grows and is the only shared channel; private histories never leak into
another seat's prompt; and the host's graders -- not the seats' unanimous
word -- decide the verdict. The last case runs a full scripted three-seat
episode against the real harness end to end.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agentwb.evals.harness import Harness, load_task
from agentwb.experience.store import ExperienceStore
from agentwb.protocols.task_room import TaskRoom, parse_seat_names
from agentwb.trajectories.store import TrajectoryStore

import drive_task_room


class ScriptedSeat:
    """A seat client that walks a fixed script and records every prompt it
    was shown -- which is exactly what the isolation tests need to inspect."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def ask_json(self, system: str, user: str, prompt_id: str = ""):
        self.prompts.append(user)
        reply = self.replies.pop(0) if self.replies else {"action": "done",
                                                          "report": "out of script"}
        return SimpleNamespace(data=reply)

    def saw(self, text: str) -> bool:
        return any(text in p for p in self.prompts)


class RoomCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def room(self, seats, max_rounds=4):
        return TaskRoom(self.root / "room", task_prompt="do the thing",
                        workspace=self.ws, seats=seats, max_rounds=max_rounds,
                        task_id="t")


class TestBoard(RoomCase):
    def test_posts_append_and_never_rewrite(self):
        room = self.room({"a": ScriptedSeat([])})
        room.post("a", {"type": "PLAN", "text": "first"})
        first = room.board_path.read_text(encoding="utf-8")
        room.post("a", {"type": "STATUS", "text": "second"})
        second = room.board_path.read_text(encoding="utf-8")
        self.assertTrue(second.startswith(first),
                        "an existing board line must never change")
        board = room.board()
        self.assertEqual([m["seq"] for m in board], [1, 2])
        self.assertEqual(board[0]["message"]["text"], "first")

    def test_a_torn_tail_never_takes_the_board_down(self):
        """A crash mid-append leaves one half-written final line; the board
        must stay readable (minus the message that was never fully
        published), and a later append must start on a fresh line rather
        than welding onto the torn one. Found live: a sonnet-vs-haiku
        exchange reviewing this module flagged the unhandled torn tail."""
        room = self.room({"a": ScriptedSeat([])})
        room.post("a", {"type": "PLAN", "text": "intact"})
        with room.board_path.open("a", encoding="utf-8") as fh:
            fh.write('{"seq": 2, "sender": "a", "mess')   # torn, no newline
        board = room.board()
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0]["message"]["text"], "intact")

        room.post("a", {"type": "STATUS", "text": "after the crash"})
        board = room.board()
        self.assertEqual([m["message"]["text"] for m in board],
                         ["intact", "after the crash"],
                         "append after a torn tail must start a fresh line")

    def test_untyped_messages_are_refused(self):
        room = self.room({"a": ScriptedSeat([])})
        with self.assertRaises(ValueError):
            room.post("a", {"text": "no type"})
        with self.assertRaises(ValueError):
            room.post("a", "just a string")
        self.assertEqual(room.board(), [])


class TestIsolationAndSharing(RoomCase):
    def test_private_history_stays_private_but_board_posts_are_shared(self):
        a = ScriptedSeat([
            {"action": "work", "note": "SECRET_A_NOTE"},
            {"action": "done", "report": "a done"},
        ])
        b = ScriptedSeat([
            {"action": "post", "message": {"type": "STATUS", "text": "B_PUBLIC_POST"}},
            {"action": "done", "report": "b done"},
        ])
        episode = self.room({"a": a, "b": b}).run()

        # b never sees a's private work note; a's later prompt shows b's post.
        self.assertFalse(b.saw("SECRET_A_NOTE"),
                         "a private note leaked into another seat's prompt")
        self.assertTrue(a.saw("SECRET_A_NOTE"), "a seat must see its OWN history")
        self.assertTrue(a.saw("B_PUBLIC_POST"), "board posts are shared")
        self.assertEqual(episode["termination_reason"], "all_done")

    def test_host_applied_files_are_confined_to_the_workspace(self):
        a = ScriptedSeat([
            {"action": "work", "note": "escape", "files": {"../evil.txt": "x"}},
            {"action": "done", "report": "done"},
        ])
        episode = self.room({"a": a}).run()
        self.assertEqual(episode["seats"]["a"]["violations"], 1)
        self.assertEqual(episode["seats"]["a"]["works"], 0)
        self.assertFalse((self.root / "evil.txt").exists())


class TestSeatNames(unittest.TestCase):
    def test_default_count_and_names(self):
        self.assertEqual(parse_seat_names(""), ["agent1", "agent2", "agent3"])
        self.assertEqual(parse_seat_names("2"), ["agent1", "agent2"])
        self.assertEqual(parse_seat_names("alice, bob"), ["alice", "bob"])


class HarnessCase(RoomCase):
    """Cases that grade a real workspace with the real harness."""

    def setUp(self):
        super().setUp()
        data = self.root / "host_data"
        self.store = TrajectoryStore(data)
        self.harness = Harness(self.store,
                               ExperienceStore(data / "experience"),
                               data / "unused_ws")

        self.task_path = Path(__file__).resolve().parents[1] / \
            "tasks_room" / "joint_stats.json"
        self.task = load_task(self.task_path)
        for rel, content in self.task.environment.files.items():
            (self.ws / rel).write_text(content, encoding="utf-8")

    def tearDown(self):
        self.store.close()
        super().tearDown()


class TestGradingIndependence(HarnessCase):
    def test_unanimous_success_claims_do_not_move_the_verdict(self):
        """Every seat reports success; the workspace still fails the suite.
        The verdict must come from the graders, not the chorus."""
        seats = {name: ScriptedSeat([{"action": "done",
                                      "report": "all tests pass, great work team"}])
                 for name in ("a", "b", "c")}
        episode = self.room(seats).run()
        verdict = drive_task_room.grade(self.harness, self.task, self.ws, episode)
        self.assertFalse(verdict["passed"])
        self.assertEqual(sorted(verdict["seats_claiming_done"]), ["a", "b", "c"])


class TestScriptedEpisode(HarnessCase):
    def test_three_seats_jointly_implement_the_module(self):
        """A full offline episode: agent1 edits the workspace directly (the
        external-occupant path), agent2 and agent3 hand the host their edits
        via "files" (the in-process path); everyone coordinates on the board;
        the host's graders pass the result on its merits."""
        header = "def _need(values):\n    if not values:\n        raise ValueError('empty')\n"
        mean_src = header + "\n\ndef mean(values):\n    _need(values)\n    return sum(values) / len(values)\n"
        median_src = ("\n\ndef median(values):\n    _need(values)\n"
                      "    s = sorted(values)\n    n = len(s)\n"
                      "    mid = n // 2\n"
                      "    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2\n")
        mode_src = ("\n\ndef mode(values):\n    _need(values)\n"
                    "    from collections import Counter\n"
                    "    counts = Counter(values)\n"
                    "    best = max(counts.values())\n"
                    "    return min(v for v, c in counts.items() if c == best)\n")

        ws = self.ws

        class DirectEditor(ScriptedSeat):
            """agent1 acts like an external occupant: it touches the shared
            workspace itself and only reports a note."""

            def ask_json(self, system, user, prompt_id=""):
                if self.replies and self.replies[0].get("_write"):
                    (ws / "stats.py").write_text(mean_src, encoding="utf-8")
                return super().ask_json(system, user, prompt_id)

        agent1 = DirectEditor([
            {"action": "post", "message": {"type": "PLAN",
                                           "text": "I take mean(); agent2 median, agent3 mode"}},
            {"_write": True, "action": "work",
             "note": "wrote _need() guard and mean() directly in stats.py"},
            {"action": "done", "report": "mean() in place, guard shared"},
        ])
        agent2 = ScriptedSeat([
            {"action": "post", "message": {"type": "STATUS", "text": "taking median()"}},
            {"action": "work", "note": "appended median()",
             "files": {"stats.py": mean_src + median_src}},
            {"action": "done", "report": "median() appended after agent1's mean"},
        ])
        agent3 = ScriptedSeat([
            {"action": "post", "message": {"type": "STATUS", "text": "taking mode()"}},
            {"action": "work", "note": "appended mode()",
             "files": {"stats.py": mean_src + median_src + mode_src}},
            {"action": "done", "report": "mode() appended; suite should be green"},
        ])

        room = self.room({"agent1": agent1, "agent2": agent2, "agent3": agent3},
                         max_rounds=5)
        episode = room.run()
        episode["verdict"] = drive_task_room.grade(self.harness, self.task,
                                                   self.ws, episode)
        room.close()
        path = room.save_episode(episode)

        # The board carried the coordination and later seats saw it.
        self.assertEqual(episode["board_messages"], 3)
        self.assertTrue(agent3.saw("agent2 median, agent3 mode"))
        # Direct edits and host-applied edits landed in ONE workspace.
        final = (ws / "stats.py").read_text(encoding="utf-8")
        for fn in ("def mean", "def median", "def mode"):
            self.assertIn(fn, final)
        # The host's graders, not the seats, say it passed.
        self.assertTrue(episode["verdict"]["passed"], episode["verdict"])
        self.assertEqual(episode["termination_reason"], "all_done")
        self.assertEqual(episode["rounds_used"], 3)
        for seat in ("agent1", "agent2", "agent3"):
            self.assertEqual(episode["seats"][seat]["turns"], 3)
            self.assertEqual(episode["seats"][seat]["violations"], 0)
        # The episode record on disk is complete and CLOSED is up.
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["board"]), 3)
        self.assertTrue((room.root / "CLOSED").exists())


if __name__ == "__main__":
    unittest.main()
