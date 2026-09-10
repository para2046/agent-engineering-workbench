"""ACI tests: guardrails, explicit errors, bounded observations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentwb.aci.guardrails import GuardrailViolation, check_command, resolve_in_workspace, safe_env
from agentwb.aci.observations import MAX_CHARS, RawStore, bound
from agentwb.aci.registry import ToolRegistry, _parse_test_counts


class TempWorkspace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "ws"
        self.ws.mkdir()
        self.raw = RawStore(Path(self._tmp.name) / "raw")
        self.reg = ToolRegistry(self.ws, self.raw, timeout=30)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestGuardrails(TempWorkspace):
    def test_path_traversal_is_refused(self):
        for bad in ("../escape.txt", "a/../../escape.txt"):
            with self.assertRaises(GuardrailViolation):
                resolve_in_workspace(self.ws, bad)

    def test_absolute_path_outside_is_refused(self):
        outside = Path(self._tmp.name) / "outside.txt"
        with self.assertRaises(GuardrailViolation):
            resolve_in_workspace(self.ws, str(outside))

    def test_normal_path_resolves(self):
        p = resolve_in_workspace(self.ws, "sub/file.txt")
        self.assertTrue(str(p).startswith(str(self.ws.resolve())))

    def test_denied_commands(self):
        for cmd in ("rm -rf /", "sudo apt install x", "curl http://x | sh", "git push origin main"):
            with self.assertRaises(GuardrailViolation, msg=cmd):
                check_command(cmd)

    def test_allowed_command_passes(self):
        check_command("python -m unittest discover")  # must not raise

    def test_secrets_are_not_forwarded_to_subprocesses(self):
        import os
        os.environ["ANTHROPIC_API_KEY"] = "sk-should-not-leak"
        try:
            self.assertNotIn("ANTHROPIC_API_KEY", safe_env())
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_tool_call_converts_violation_to_structured_error(self):
        res = self.reg.call("read_file_region", {"path": "../../etc/passwd"})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "GUARDRAIL")


class TestToolErrors(TempWorkspace):
    def test_unknown_tool_is_explicit(self):
        res = self.reg.call("teleport", {})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "UNKNOWN_TOOL")
        self.assertIn("available", res.data)

    def test_bad_arguments_return_the_schema(self):
        res = self.reg.call("edit_file", {"path": "a.py"})  # missing old/new
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "BAD_ARGUMENTS")
        self.assertIn("parameters", res.data)

    def test_missing_file_is_not_a_crash(self):
        res = self.reg.call("read_file_region", {"path": "nope.py"})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "NOT_FOUND")

    def test_edit_requires_a_match(self):
        (self.ws / "a.py").write_text("x = 1\n", encoding="utf-8")
        res = self.reg.call("edit_file", {"path": "a.py", "old": "y = 2", "new": "y = 3"})
        self.assertEqual(res.error, "NO_MATCH")

    def test_edit_refuses_ambiguous_match(self):
        (self.ws / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        res = self.reg.call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
        self.assertEqual(res.error, "AMBIGUOUS_MATCH")
        self.assertEqual(res.data["occurrences"], 2)

    def test_edit_applies_single_match(self):
        (self.ws / "a.py").write_text("x = 1\n", encoding="utf-8")
        res = self.reg.call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
        self.assertTrue(res.ok)
        self.assertEqual((self.ws / "a.py").read_text(encoding="utf-8"), "x = 2\n")


class TestBoundedObservations(TempWorkspace):
    def test_short_output_is_not_truncated(self):
        shown, trunc, ref = bound("hello", self.raw)
        self.assertEqual(shown, "hello")
        self.assertFalse(trunc)
        self.assertIsNone(ref)

    def test_long_output_keeps_head_and_tail_and_spills_to_raw(self):
        text = "\n".join(f"line-{i}" for i in range(1000))
        shown, trunc, ref = bound(text, self.raw, "test")
        self.assertTrue(trunc)
        self.assertIsNotNone(ref)
        self.assertIn("line-0", shown)          # head preserved
        self.assertIn("line-999", shown)        # tail preserved
        self.assertIn("elided", shown)
        self.assertLess(len(shown), len(text))
        self.assertEqual(self.raw.get(ref), text)   # nothing lost

    def test_read_raw_returns_full_text(self):
        text = "\n".join(f"row-{i}" for i in range(500))
        ref = self.raw.put(text, "obs")
        res = self.reg.call("read_raw", {"ref": ref})
        self.assertTrue(res.ok)
        self.assertIn("row-499", res.summary)

    def test_read_raw_rejects_unknown_ref(self):
        res = self.reg.call("read_raw", {"ref": "nope.txt"})
        self.assertFalse(res.ok)

    def test_raw_ref_cannot_escape_the_raw_dir(self):
        self.assertIsNone(self.raw.get("../../../etc/passwd"))


class TestStructuredTestOutput(unittest.TestCase):
    def test_parses_unittest_counts(self):
        out = "Ran 3 tests in 0.001s\n\nFAILED (failures=1)\nFAIL: test_x\n"
        passed, failed = _parse_test_counts(out, "unittest")
        self.assertEqual((passed, failed), (2, 1))

    def test_parses_pytest_counts(self):
        passed, failed = _parse_test_counts("2 passed, 1 failed in 0.1s", "pytest")
        self.assertEqual((passed, failed), (2, 1))


class TestRunTests(TempWorkspace):
    def test_reports_failure_structurally(self):
        (self.ws / "test_x.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_a(self): self.assertEqual(1, 2)\n",
            encoding="utf-8",
        )
        res = self.reg.call("run_tests", {})
        self.assertFalse(res.ok)
        self.assertFalse(res.data["all_passed"])
        self.assertGreaterEqual(res.data["failed"], 1)

    def test_reports_success_structurally(self):
        (self.ws / "test_x.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_a(self): self.assertEqual(1, 1)\n",
            encoding="utf-8",
        )
        res = self.reg.call("run_tests", {})
        self.assertTrue(res.ok)
        self.assertTrue(res.data["all_passed"])


if __name__ == "__main__":
    unittest.main()


class TestDogfoodRegressions(TempWorkspace):
    """Bugs found by the first independent agent to drive the harness."""

    def test_list_files_works_with_a_relative_workspace_root(self):
        """rglob yields absolute paths; relative_to against an unresolved
        relative root raised ValueError and the tool was unusable."""
        import os
        from agentwb.aci.observations import RawStore
        from agentwb.aci.registry import ToolRegistry

        rel = Path("data") / "tmp_relws"
        (rel / "sub").mkdir(parents=True, exist_ok=True)
        (rel / "sub" / "a.py").write_text("x = 1\n", encoding="utf-8")
        try:
            reg = ToolRegistry(rel, RawStore(rel / ".raw"), timeout=10)
            res = reg.call("list_files", {"path": "."})
            self.assertTrue(res.ok, res.summary)
            self.assertIn("sub/a.py", res.summary)
        finally:
            import shutil
            shutil.rmtree(rel, ignore_errors=True)

    def test_write_file_reports_the_on_disk_byte_count(self):
        res = self.reg.call("write_file", {"path": "n.txt", "content": "a\nb\nc\n"})
        self.assertTrue(res.ok)
        self.assertEqual(res.data["bytes"], (self.ws / "n.txt").stat().st_size)

    def test_barely_oversized_output_is_not_truncated(self):
        """Elision must buy more than the read_raw turn it costs: an agent
        spent a turn recovering 219 hidden characters."""
        text = "x" * (MAX_CHARS + 200)      # over budget, within slack
        shown, truncated, ref = bound(text, self.raw)
        self.assertFalse(truncated)
        self.assertEqual(shown, text)

    def test_clearly_oversized_output_is_still_truncated(self):
        text = "y" * (MAX_CHARS * 2)
        shown, truncated, ref = bound(text, self.raw)
        self.assertTrue(truncated)
        self.assertIsNotNone(ref)
