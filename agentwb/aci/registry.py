"""Tool registry and the V0 tool set.

Each tool has a clear purpose, a small parameter surface, structured input and
output, bounded observations and explicit errors -- the ACI checklist from
spec section 4. Tools are plain functions bound to a workspace context; the
registry hands the runtime both the schemas (for the provider) and the
callables (for execution).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..types import ToolResult, ToolSchema
from .guardrails import GuardrailViolation, check_command, resolve_in_workspace, safe_env
from .observations import RawStore, bound

ToolFn = Callable[..., ToolResult]


@dataclass
class Tool:
    schema: ToolSchema
    fn: ToolFn


class ToolRegistry:
    """Workspace-bound collection of tools."""

    def __init__(self, workspace: Path, raw: RawStore, timeout: int = 60):
        self.workspace = Path(workspace)
        self.raw = raw
        self.timeout = timeout
        self._tools: dict[str, Tool] = {}
        self._install_defaults()

    # -- registry plumbing ------------------------------------------------
    def register(self, schema: ToolSchema, fn: ToolFn) -> None:
        self._tools[schema.name] = Tool(schema, fn)

    def schemas(self, allow: Optional[list[str]] = None) -> list[ToolSchema]:
        names = allow if allow else list(self._tools)
        return [self._tools[n].schema for n in names if n in self._tools]

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                summary=f"no such tool: {name}",
                error="UNKNOWN_TOOL",
                data={"available": sorted(self._tools)},
            )
        try:
            return tool.fn(**arguments)
        except GuardrailViolation as exc:
            return ToolResult(ok=False, summary=str(exc), error="GUARDRAIL")
        except TypeError as exc:
            # wrong/missing arguments -- report the schema back so the agent can retry
            return ToolResult(
                ok=False,
                summary=f"bad arguments for {name}: {exc}",
                error="BAD_ARGUMENTS",
                data={"parameters": tool.schema.parameters},
            )
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            return ToolResult(ok=False, summary=f"{type(exc).__name__}: {exc}", error="TOOL_ERROR")

    # -- default tool set -------------------------------------------------
    def _install_defaults(self) -> None:
        self.register(
            ToolSchema(
                name="list_files",
                description="List files in a workspace directory. Use before guessing a path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory relative to the workspace root. Default '.'"},
                    },
                    "required": [],
                },
            ),
            self._list_files,
        )
        self.register(
            ToolSchema(
                name="read_file_region",
                description="Read a numbered line range from a file. Omitting start/end reads from the top; end defaults to start+120. Prefer ranges over whole files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start": {"type": "integer", "description": "1-indexed first line. Default 1."},
                        "end": {"type": "integer", "description": "Inclusive last line. Default start+120."},
                    },
                    "required": ["path"],
                },
            ),
            self._read_file_region,
        )
        self.register(
            ToolSchema(
                name="write_file",
                description="Write a file, creating parent directories. Overwrites existing content.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            ),
            self._write_file,
        )
        self.register(
            ToolSchema(
                name="edit_file",
                description="Replace one exact occurrence of `old` with `new` in a file. Fails if `old` is absent or ambiguous.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                },
            ),
            self._edit_file,
        )
        self.register(
            ToolSchema(
                name="search_code",
                description="Search workspace files for a literal string. Returns path:line matches, capped.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "glob": {"type": "string", "description": "File glob, default '**/*.py'"},
                    },
                    "required": ["query"],
                },
            ),
            self._search_code,
        )
        self.register(
            ToolSchema(
                name="run_tests",
                description="Run the workspace test suite. Returns pass/fail counts and failure detail.",
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Optional test file or directory."},
                    },
                    "required": [],
                },
            ),
            self._run_tests,
        )
        self.register(
            ToolSchema(
                name="shell",
                description="Run a shell command in the workspace. Use a purpose-built tool when one exists.",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            ),
            self._shell,
        )
        self.register(
            ToolSchema(
                name="read_raw",
                description="Retrieve the full text of a truncated earlier observation by its ref.",
                parameters={
                    "type": "object",
                    "properties": {"ref": {"type": "string"}},
                    "required": ["ref"],
                },
            ),
            self._read_raw,
        )

    # -- implementations --------------------------------------------------
    def _list_files(self, path: str = ".") -> ToolResult:
        target = resolve_in_workspace(self.workspace, path)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"not a directory: {path}", error="NOT_A_DIRECTORY")
        # rglob yields absolute paths (target is resolved), so the base for
        # relative display must be resolved too. With a relative workspace root
        # this raised ValueError and the tool was unusable -- found by the
        # first independent agent to drive the harness, invisible to the test
        # suite because every test constructed the registry with an absolute
        # tmpdir.
        base = self.workspace.resolve()
        entries = []
        for p in sorted(target.rglob("*")):
            if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in p.parts):
                continue
            rel = p.relative_to(base).as_posix()
            entries.append(f"{rel}/" if p.is_dir() else f"{rel} ({p.stat().st_size}b)")
            if len(entries) >= 200:
                entries.append("... [listing capped at 200 entries]")
                break
        body = "\n".join(entries) or "(empty)"
        shown, trunc, ref = bound(body, self.raw, "listing")
        return ToolResult(ok=True, summary=shown, data={"count": len(entries)}, truncated=trunc, raw_ref=ref)

    def _read_file_region(self, path: str, start: int = 1, end: Optional[int] = None) -> ToolResult:
        target = resolve_in_workspace(self.workspace, path)
        if not target.is_file():
            return ToolResult(ok=False, summary=f"no such file: {path}", error="NOT_FOUND")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start))
        end = int(end) if end is not None else start + 120
        end = min(end, len(lines))
        if start > len(lines):
            return ToolResult(
                ok=False,
                summary=f"{path} has {len(lines)} lines; start={start} is past the end",
                error="OUT_OF_RANGE",
            )
        numbered = [f"{i:>4}  {lines[i - 1]}" for i in range(start, end + 1)]
        body = "\n".join(numbered)
        shown, trunc, ref = bound(body, self.raw, "file")
        return ToolResult(
            ok=True,
            summary=shown,
            data={"path": path, "start": start, "end": end, "total_lines": len(lines)},
            truncated=trunc,
            raw_ref=ref,
        )

    def _write_file(self, path: str, content: str) -> ToolResult:
        target = resolve_in_workspace(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.is_file()
        target.write_text(content, encoding="utf-8")
        # Report the on-disk size, not len(content): text-mode writes on
        # Windows translate newlines, and a byte count that disagrees with
        # the filesystem reads as corruption to a careful agent.
        on_disk = target.stat().st_size
        return ToolResult(
            ok=True,
            summary=f"{'overwrote' if existed else 'created'} {path} ({on_disk} bytes on disk)",
            data={"path": path, "bytes": on_disk, "chars": len(content), "existed": existed},
        )

    def _edit_file(self, path: str, old: str, new: str) -> ToolResult:
        target = resolve_in_workspace(self.workspace, path)
        if not target.is_file():
            return ToolResult(ok=False, summary=f"no such file: {path}", error="NOT_FOUND")
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(
                ok=False,
                summary=f"`old` not found in {path} -- read the region first and match it exactly",
                error="NO_MATCH",
            )
        if count > 1:
            return ToolResult(
                ok=False,
                summary=f"`old` appears {count} times in {path}; include more context to make it unique",
                error="AMBIGUOUS_MATCH",
                data={"occurrences": count},
            )
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult(ok=True, summary=f"edited {path} (1 replacement)", data={"path": path})

    def _search_code(self, query: str, glob: str = "**/*.py") -> ToolResult:
        hits: list[str] = []
        for p in sorted(self.workspace.glob(glob)):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if query in line:
                        rel = p.relative_to(self.workspace).as_posix()
                        hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 100:
                            break
            except OSError:
                continue
            if len(hits) >= 100:
                hits.append("... [capped at 100 matches]")
                break
        body = "\n".join(hits) or f"no matches for {query!r} in {glob}"
        shown, trunc, ref = bound(body, self.raw, "search")
        return ToolResult(ok=True, summary=shown, data={"matches": len(hits), "query": query},
                          truncated=trunc, raw_ref=ref)

    def _run_tests(self, target: Optional[str] = None) -> ToolResult:
        """Run the suite with pytest when available, else unittest discovery.

        Returns a *structured* outcome -- passed/failed counts plus the failure
        text -- because "did tests pass" is a question graders answer in code,
        never by asking a model to read console noise.
        """
        if shutil.which("pytest") or self._module_available("pytest"):
            cmd = [sys.executable, "-m", "pytest", "-q"]
            if target:
                cmd.append(target)
            runner = "pytest"
        else:
            cmd = [sys.executable, "-m", "unittest", "discover", "-v"]
            if target:
                cmd = [sys.executable, "-m", "unittest", target, "-v"]
            runner = "unittest"

        proc = self._run(cmd)
        out = (proc.stdout or "") + (proc.stderr or "")
        passed, failed = _parse_test_counts(out, runner)
        ok = proc.returncode == 0
        shown, trunc, ref = bound(out, self.raw, "tests")
        headline = f"[{runner}] {'PASS' if ok else 'FAIL'} exit={proc.returncode} passed={passed} failed={failed}"
        return ToolResult(
            ok=ok,
            summary=f"{headline}\n{shown}",
            data={
                "runner": runner,
                "exit_code": proc.returncode,
                "passed": passed,
                "failed": failed,
                "all_passed": ok,
            },
            truncated=trunc,
            raw_ref=ref,
        )

    def _shell(self, command: str) -> ToolResult:
        check_command(command)
        proc = self._run(command, shell=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        shown, trunc, ref = bound(out, self.raw, "shell")
        return ToolResult(
            ok=proc.returncode == 0,
            summary=f"exit={proc.returncode}\n{shown}",
            data={"exit_code": proc.returncode, "command": command},
            error=None if proc.returncode == 0 else "NONZERO_EXIT",
            truncated=trunc,
            raw_ref=ref,
        )

    def _read_raw(self, ref: str) -> ToolResult:
        text = self.raw.get(ref)
        if text is None:
            return ToolResult(ok=False, summary=f"no raw observation named {ref!r}", error="NOT_FOUND")
        shown, trunc, _ = bound(text, None, "raw", max_chars=12000, max_lines=400)
        return ToolResult(ok=True, summary=shown, data={"ref": ref}, truncated=trunc)

    # -- helpers ----------------------------------------------------------
    def _run(self, cmd, shell: bool = False) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                cwd=str(self.workspace),
                shell=shell,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=safe_env(),
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {self.timeout}s")

    @staticmethod
    def _module_available(name: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(name) is not None


def _parse_test_counts(output: str, runner: str) -> tuple[int, int]:
    """Best-effort structured counts from test output."""
    import re

    if runner == "pytest":
        passed = sum(int(m) for m in re.findall(r"(\d+) passed", output))
        failed = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", output))
        return passed, failed

    m = re.search(r"^Ran (\d+) test", output, re.M)
    total = int(m.group(1)) if m else 0
    fails = len(re.findall(r"^(?:FAIL|ERROR):", output, re.M))
    return max(total - fails, 0), fails
