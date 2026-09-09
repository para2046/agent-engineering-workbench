"""Cost estimation (spec section 27).

Optimization must weigh quality against spend --

    utility = task_success - l1*cost - l2*latency - l3*tool_failures - l4*extra_rounds

-- and a promotion gate cannot enforce a cost ceiling it cannot measure. So
this exists to feed those two, not to produce an invoice.

**There is no built-in price table, on purpose.** Model prices change, vary by
tier and region, and a stale hard-coded number would silently produce confident
wrong figures in every report downstream. That is precisely the failure this
codebase refuses everywhere else, so cost is UNKNOWN (``None``) until you tell
it what your tokens cost:

    // agentwb.json
    "model_prices": {
      "claude-sonnet-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
      "gpt-4o":          {"input_per_mtok": 2.50, "output_per_mtok": 10.00}
    }

``None`` propagates honestly: a promotion gate with a cost ceiling refuses to
pass a candidate whose cost it could not measure, rather than assuming zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

MTOK = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok
                + output_tokens * self.output_per_mtok) / MTOK


class PriceBook:
    """Model prices, supplied by configuration.

    Lookup is exact first, then longest-prefix, so ``claude-sonnet-5`` also
    prices ``claude-sonnet-5-20260101`` without needing an entry per snapshot.
    """

    def __init__(self, prices: Optional[dict[str, Any]] = None):
        self._prices: dict[str, ModelPrice] = {}
        for model, spec in (prices or {}).items():
            try:
                self._prices[model] = ModelPrice(
                    input_per_mtok=float(spec["input_per_mtok"]),
                    output_per_mtok=float(spec["output_per_mtok"]),
                )
            except (KeyError, TypeError, ValueError):
                # A malformed entry is skipped rather than defaulted -- a wrong
                # price is worse than a missing one.
                continue

    def __bool__(self) -> bool:
        return bool(self._prices)

    @property
    def models(self) -> list[str]:
        return sorted(self._prices)

    def price_for(self, model: str) -> Optional[ModelPrice]:
        if not model:
            return None
        if model in self._prices:
            return self._prices[model]
        matches = [k for k in self._prices if model.startswith(k)]
        return self._prices[max(matches, key=len)] if matches else None

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Estimated USD, or None when the model has no configured price.

        None means "not measured". It never means zero.
        """
        price = self.price_for(model)
        if price is None:
            return None
        return round(price.estimate(input_tokens or 0, output_tokens or 0), 6)


def utility(
    success: float,
    cost: Optional[float] = None,
    latency_ms: int = 0,
    tool_failures: int = 0,
    extra_rounds: int = 0,
    weights: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """The spec's objective (section 27).

    Returns None when cost was requested as a term but is unmeasured -- a
    utility number computed with an unknown treated as zero would rank an
    unpriced model as free, which is exactly backwards.
    """
    w = {"cost": 1.0, "latency": 0.0, "tool_failures": 0.0, "rounds": 0.0, **(weights or {})}
    if w["cost"] and cost is None:
        return None
    return round(
        success
        - w["cost"] * (cost or 0.0)
        - w["latency"] * (latency_ms / 1000.0)
        - w["tool_failures"] * tool_failures
        - w["rounds"] * extra_rounds,
        6,
    )
