"""Tests for the human/assistant-driven provider.

This is the provider that let the workbench face a real model for the first
time. The mechanism it depends on is replay: each pass re-runs the answers
already given, so a run advances one turn per invocation. If replay is not
faithful, every session run is quietly wrong.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentwb.providers.base import build_provider
from agentwb.providers.session import NeedsAnswer, SessionProvider
from agentwb.types import Message


class SessionCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "session"

    def tearDown(self):
        self._tmp.cleanup()

    def provider(self, answers=None) -> SessionProvider:
        p = SessionProvider(root=str(self.root))
        if answers is not None:
            (self.root / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
        return p

    def ask(self, provider):
        return provider.generate("system", [Message(role="user", content="task")], [])


class TestAnswerReplay(SessionCase):
    def test_an_unanswered_turn_stops_the_run(self):
        with self.assertRaises(NeedsAnswer) as cm:
            self.ask(self.provider([]))
        self.assertEqual(cm.exception.index, 0)
        self.assertTrue(cm.exception.pending_path.is_file())

    def test_the_pending_file_records_the_real_prompt(self):
        """The decision must be made on the actual input, not a summary."""
        p = self.provider([])
        with self.assertRaises(NeedsAnswer):
            p.generate("SYSTEM TEXT", [Message(role="user", content="TASK TEXT")], [])
        payload = json.loads((self.root / "pending.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["system"], "SYSTEM TEXT")
        self.assertEqual(payload["messages"][0]["content"], "TASK TEXT")
        self.assertEqual(payload["turn"], 0)

    def test_answers_replay_in_order(self):
        p = self.provider([
            {"tool": "run_tests", "arguments": {}},
            {"tool": "read_file_region", "arguments": {"path": "a.py"}},
        ])
        self.assertEqual(self.ask(p).tool_calls[0].name, "run_tests")
        self.assertEqual(self.ask(p).tool_calls[0].name, "read_file_region")

    def test_the_run_stops_at_the_first_unanswered_turn(self):
        p = self.provider([{"tool": "run_tests", "arguments": {}}])
        self.ask(p)
        with self.assertRaises(NeedsAnswer) as cm:
            self.ask(p)
        self.assertEqual(cm.exception.index, 1)

    def test_reset_rewinds_for_the_next_pass(self):
        """Replay depends on this: each pass must start from turn 0."""
        p = self.provider([{"tool": "run_tests", "arguments": {}}])
        self.ask(p)
        p.reset()
        self.assertEqual(self.ask(p).tool_calls[0].name, "run_tests")

    def test_text_answers_finish_the_run(self):
        response = self.ask(self.provider([{"text": "done, and here is the evidence"}]))
        self.assertEqual(response.tool_calls, [])
        self.assertIn("evidence", response.text)

    def test_arguments_survive_replay_intact(self):
        p = self.provider([{"tool": "edit_file", "arguments": {
            "path": "calculator.py", "old": "    if b == 0:\n        return 0",
            "new": "    if b == 0:\n        raise ValueError(\"x\")"}}])
        args = self.ask(p).tool_calls[0].arguments
        self.assertIn("\n", args["old"])
        self.assertEqual(args["path"], "calculator.py")


class TestMalformedAnswers(SessionCase):
    def test_an_answer_with_neither_tool_nor_text_is_rejected(self):
        """A dropped turn would look like the agent choosing to stop."""
        with self.assertRaises(ValueError) as cm:
            self.ask(self.provider([{"thinking": "hmm"}]))
        self.assertIn("tool", str(cm.exception))

    def test_a_non_object_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ask(self.provider(["just run the tests"]))

    def test_invalid_json_is_reported_clearly(self):
        p = self.provider()
        (self.root / "answers.json").write_text("[{oops}]", encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            self.ask(p)
        self.assertIn("not valid JSON", str(cm.exception))


class TestRegistration(SessionCase):
    def test_it_is_reachable_as_a_provider_key(self):
        self.assertEqual(build_provider("session", root=str(self.root)).name, "session")


if __name__ == "__main__":
    unittest.main()
