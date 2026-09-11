"""DSPy adapter (spec section 11).

DSPy is used here as the **proposal engine** of the optimization loop: its
typed signature/prediction machinery, running on a real LM, drafts candidate
policies from failure evidence. Everything after that is deliberately NOT
DSPy's job -- candidates go through the same invariant guard before scoring
and the same promotion gate before adoption as a hand-written prompt. A
candidate from a published optimizer earns no extra trust.

Why not `dspy.compile` with a metric over a trainset? Our metric is "run the
whole agent on a task and grade the environment" -- minutes and real money per
call. Handing that to an inner optimization loop would spend the budget
inside DSPy where the gate cannot see it. So DSPy proposes (cheap, one call
per candidate) and the harness evaluates (expensive, but owned by the gate).

``ProviderLM`` bridges any workbench provider into DSPy, so the same
``--judge-provider claude-cli`` that powers judges also powers DSPy -- no
separate API-key plumbing.
"""

from __future__ import annotations

from typing import Any, Optional

from ..prompts import VersionedPrompt
from ..providers.base import ModelProvider, ProviderError
from ..types import Message
from .optimizer import PolicyCandidate, guard


def _require_dspy():
    try:
        import dspy  # noqa: WPS433
        return dspy
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ProviderError(
            "DSPy is not installed -- run `pip install dspy-ai`, or use "
            "`--optimizer reflective`"
        ) from exc


class ProviderLM:
    """A dspy.BaseLM that answers through a workbench ModelProvider.

    Built lazily via :func:`provider_lm` because the base class only exists
    once dspy is importable.
    """


def provider_lm(provider: ModelProvider):
    dspy = _require_dspy()

    class _ProviderLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model=f"workbench/{getattr(provider, 'name', 'provider')}")
            self._provider = provider

        def forward(self, prompt=None, messages=None, **kwargs):
            system_parts: list[str] = []
            user_parts: list[str] = []
            for m in (messages or ([] if prompt is None else [{"role": "user", "content": prompt}])):
                content = m.get("content", "")
                if isinstance(content, list):        # multimodal shape; text only
                    content = " ".join(str(c.get("text", "")) for c in content
                                       if isinstance(c, dict))
                (system_parts if m.get("role") == "system" else user_parts).append(str(content))

            response = self._provider.generate(
                "\n\n".join(system_parts),
                [Message(role="user", content="\n\n".join(user_parts))],
                [],
            )

            # OpenAI-response shape, which is what BaseLM postprocessing expects.
            from types import SimpleNamespace
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=response.text, tool_calls=None),
                    finish_reason="stop")],
                usage={"prompt_tokens": response.usage.input_tokens,
                       "completion_tokens": response.usage.output_tokens,
                       "total_tokens": response.usage.input_tokens
                       + response.usage.output_tokens},
                model=self.model,
            )

    return _ProviderLM()


class DSPyOptimizer:
    name = "dspy"

    def __init__(self, provider: Optional[ModelProvider] = None,
                 metric: Optional[Any] = None, **_):
        self._dspy = _require_dspy()
        if provider is None:
            raise ValueError(
                "DSPyOptimizer needs a provider to run its LM on -- pass the judge "
                "provider (e.g. --judge-provider claude-cli)"
            )
        self.provider = provider
        self.metric = metric      # reserved for a future compile-based mode

    def propose(self, baseline: VersionedPrompt, feedback: str,
                n: int = 2) -> list[PolicyCandidate]:
        dspy = self._dspy

        class ReviseAgentPolicy(dspy.Signature):
            """Revise an AI agent's system prompt using evidence of how it failed.

            Keep every commitment about verifying against the environment and
            about graders deciding the outcome -- revisions that drop them are
            rejected before scoring, so removing them cannot help. Address the
            specific failures in the evidence; do not pad the prompt with
            generic best-practice advice."""

            current_prompt: str = dspy.InputField(desc="the policy as it stands")
            failure_evidence: str = dspy.InputField(
                desc="deterministic findings from recent failed runs")
            angle: str = dspy.InputField(
                desc="the specific angle this revision should take")
            revised_prompt: str = dspy.OutputField(
                desc="the complete revised system prompt, ready to use verbatim")
            rationale: str = dspy.OutputField(
                desc="one sentence: which failure this addresses and how")

        # Distinct angles rather than n identical samples -- diversity in the
        # candidates is what gives the gate something to choose between.
        angles = [
            "make the verification requirement concrete and procedural",
            "tighten tool-use guidance so the agent reads before it edits",
            "add an explicit self-check step before the final report",
            "shorten and sharpen: remove anything the failures show is ignored",
        ]

        predict = dspy.Predict(ReviseAgentPolicy)
        out: list[PolicyCandidate] = []
        with dspy.context(lm=provider_lm(self.provider)):
            for i in range(max(1, n)):
                try:
                    result = predict(current_prompt=baseline.text,
                                     failure_evidence=feedback,
                                     angle=angles[i % len(angles)])
                except Exception as exc:  # noqa: BLE001 - a failed proposal is skipped, never faked
                    out.append(guard(PolicyCandidate(
                        name=baseline.name, version=f"dspy{i + 1}", text="",
                        parent=baseline.id, optimizer=self.name,
                        rationale=f"proposal failed: {exc}")))
                    continue
                text = str(getattr(result, "revised_prompt", "") or "").strip()
                rationale = str(getattr(result, "rationale", "") or "").strip()
                out.append(guard(PolicyCandidate(
                    name=baseline.name, version=f"dspy{i + 1}", text=text,
                    parent=baseline.id, optimizer=self.name,
                    rationale=rationale[:600] or f"dspy revision, angle: {angles[i % len(angles)]}",
                )))
        return out
