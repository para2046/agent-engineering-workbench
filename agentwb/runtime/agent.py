"""The single-agent ReAct loop.

state -> decision -> action -> observation -> next state, recorded as it goes.

The loop is deliberately dull. All the leverage in this system lives in the
tools the agent is given, the trajectory it leaves behind, and the graders that
judge the environment afterwards -- not in clever control flow here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from ..aci.observations import RawStore
from ..costs import PriceBook
from ..aci.registry import ToolRegistry
from ..providers.base import ModelProvider, ProviderError, reset_provider
from ..trajectories.recorder import TrajectoryRecorder
from ..types import (
    Message,
    Step,
    Task,
    TerminationReason,
    Trajectory,
    Usage,
    new_run_id,
    utcnow,
)
from .termination import TerminationMonitor

SYSTEM_PROMPT_VERSION = "single_agent:v1"

SYSTEM_PROMPT = """\
You are an engineering agent working inside a sandboxed workspace.

How to work:
- Inspect before you act. Read the files you intend to change.
- Prefer the purpose-built tools (run_tests, read_file_region, edit_file) over raw shell.
- Verify with the environment, not with your own judgement. If a test can prove
  it, run the test. Never report success you have not observed.
- If an observation was truncated, use read_raw to retrieve the full text
  rather than guessing what it said.
- When you are finished, reply with plain text (no tool call) stating what you
  changed and what evidence shows it worked.

You do not decide whether the task passed. Graders inspect the environment
afterwards. Report what you observed, accurately, including anything you could
not verify.
"""


class AgentRunner:
    """Executes one task, in one workspace, producing one trajectory."""

    def __init__(
        self,
        provider: ModelProvider,
        workspace: Path,
        recorder: TrajectoryRecorder,
        raw: RawStore,
        tool_timeout: int = 60,
        system_prompt: str = SYSTEM_PROMPT,
        prompt_version: str = SYSTEM_PROMPT_VERSION,
        retriever=None,
        prices: PriceBook = None,
    ):
        self.provider = provider
        self.workspace = Path(workspace)
        self.recorder = recorder
        self.registry = ToolRegistry(self.workspace, raw, timeout=tool_timeout)
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        self.retriever = retriever
        self.prices = prices

    def run(self, task: Task, trial: int = 0, run_id: Optional[str] = None) -> Trajectory:
        reset_provider(self.provider)
        traj = Trajectory(
            trajectory_id=run_id or new_run_id(),
            task_id=task.id,
            model=getattr(self.provider, "model", ""),
            provider=getattr(self.provider, "name", ""),
            prompt_version=self.prompt_version,
            workspace=str(self.workspace),
            trial=trial,
            agent_config={
                "max_steps": task.max_steps,
                "tools": task.tools,
                "system_prompt_version": self.prompt_version,
            },
        )
        self.recorder.open(traj)

        tools = self.registry.schemas(task.tools)

        # Retrieval happens once, before the first turn, and what it injected is
        # recorded on the trajectory. A run that was shown past experience and
        # does not say so is not reproducible.
        context = ""
        if self.retriever is not None:
            retrieved = self.retriever.retrieve(task)
            traj.retrieval = retrieved.provenance
            context = retrieved.as_context()

        messages: list[Message] = [
            Message(role="user", content=self._task_message(task, context))
        ]
        monitor = TerminationMonitor(max_steps=task.max_steps)
        total = Usage()
        step_no = 0

        while True:
            stop = monitor.check(step_no)
            if stop is not None:
                traj.termination_reason = stop.value
                break

            step_no += 1
            started = time.time()
            try:
                resp = self.provider.generate(self.system_prompt, messages, tools)
            except ProviderError as exc:
                traj.termination_reason = TerminationReason.ENVIRONMENT_ERROR.value
                traj.final_output = {"error": str(exc)}
                break

            total = total + resp.usage
            latency = int((time.time() - started) * 1000)

            # no tool call -> the agent considers itself done
            if not resp.tool_calls:
                step = Step(
                    step=step_no,
                    state={"messages": len(messages)},
                    action={"type": "final_answer"},
                    observation={},
                    rationale=resp.text[:2000],
                    latency_ms=latency,
                    token_usage=resp.usage.to_dict(),
                    started_at=utcnow(),
                )
                traj.steps.append(step)
                self.recorder.step(traj, step)
                traj.final_output = {"text": resp.text}
                traj.termination_reason = TerminationReason.AGENT_FINISHED.value
                break

            messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls))

            for call in resp.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                monitor.observe_action(call.name, call.arguments, result.ok)
                step = Step(
                    step=step_no,
                    state={"messages": len(messages)},
                    action={"type": "tool_call", "tool": call.name, "arguments": call.arguments},
                    observation={"summary": result.summary, "truncated": result.truncated},
                    tool_result=result.to_dict(),
                    rationale=resp.text[:2000],
                    latency_ms=latency,
                    token_usage=resp.usage.to_dict(),
                    started_at=utcnow(),
                )
                traj.steps.append(step)
                self.recorder.step(traj, step)
                messages.append(
                    Message(role="tool", content=result.summary, tool_call_id=call.id, name=call.name)
                )
                latency = 0  # attribute provider latency to the first result only

        traj.ended_at = utcnow()
        traj.metrics = self._metrics(traj, total, monitor)
        # None when no price is configured -- never silently zero.
        traj.metrics["estimated_cost_usd"] = (
            self.prices.estimate(traj.model, total.input_tokens, total.output_tokens)
            if self.prices else None
        )
        return traj

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _task_message(task: Task, context: str = "") -> str:
        parts = [f"TASK ({task.id}):", task.prompt]
        if task.success_criteria:
            parts += ["", "SUCCESS CRITERIA:", task.success_criteria]
        if context:
            parts += ["", context]
        return "\n".join(parts)

    @staticmethod
    def _metrics(traj: Trajectory, usage: Usage, monitor: "TerminationMonitor") -> dict:
        tool_calls = [s for s in traj.steps if s.action.get("type") == "tool_call"]
        failures = [s for s in tool_calls if not (s.tool_result or {}).get("ok", True)]
        return {
            "steps": len(traj.steps),
            "tool_calls": len(tool_calls),
            "tool_failures": len(failures),
            "repeated_actions": monitor.repeat_count,
            "latency_ms": sum(s.latency_ms for s in traj.steps),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
