"""Evaluation metrics: accuracy, cost-weighted confusion, escalation threshold.

All costs are in units of "one low-stakes misroute" (1.0). The numbers in
``CostConfig`` are business inputs, not ML outputs: a PM sets them with support
and finance, and the threshold/model choice follows from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ticket_router.labels import INTENT_TO_TEAM, INTENTS, MISROUTE_SEVERITY
from ticket_router.urgency import URGENCY_COST_MULTIPLIER

# Misrouting within the same team (e.g. billing <-> refund) is cheap: the
# right people see it, they just re-tag it.
SAME_TEAM_DISCOUNT = 0.25
# Cost of a human triaging one escalated ticket.
HUMAN_TRIAGE_COST = 1.0


@dataclass
class CostConfig:
    severity: dict[str, float] = field(default_factory=lambda: dict(MISROUTE_SEVERITY))
    same_team_discount: float = SAME_TEAM_DISCOUNT
    urgency_multiplier: dict[str, float] = field(default_factory=lambda: dict(URGENCY_COST_MULTIPLIER))
    human_cost: float = HUMAN_TRIAGE_COST


DEFAULT_COSTS = CostConfig()


def base_cost_matrix(cost: CostConfig = DEFAULT_COSTS) -> np.ndarray:
    """C[i, j] = cost of predicting intent j when the truth is intent i."""
    n = len(INTENTS)
    matrix = np.zeros((n, n))
    for i, true in enumerate(INTENTS):
        for j, pred in enumerate(INTENTS):
            if i == j:
                continue
            sev = cost.severity[true]
            same_team = INTENT_TO_TEAM[true] == INTENT_TO_TEAM[pred]
            matrix[i, j] = sev * cost.same_team_discount if same_team else sev
    return matrix


def per_ticket_cost(
    y_true: np.ndarray, y_pred: np.ndarray, urgency: np.ndarray, cost: CostConfig = DEFAULT_COSTS
) -> np.ndarray:
    """Misroute cost per ticket, scaled by that ticket's (reference) urgency."""
    mult = np.array([cost.urgency_multiplier[u] for u in urgency])
    return base_cost_matrix(cost)[y_true, y_pred] * mult


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Confusion matrix (rows = true, cols = pred), optionally summing ``weights``."""
    n = len(INTENTS)
    m = np.zeros((n, n))
    np.add.at(m, (y_true, y_pred), 1.0 if weights is None else weights)
    return m


def cost_weighted_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, urgency: np.ndarray, cost: CostConfig = DEFAULT_COSTS
) -> np.ndarray:
    """Total misroute cost attributed to each (true, pred) cell."""
    return confusion(y_true, y_pred, per_ticket_cost(y_true, y_pred, urgency, cost))


@dataclass
class ThresholdPoint:
    threshold: float
    coverage: float  # fraction auto-routed
    auto_accuracy: float  # accuracy on auto-routed tickets
    misroute_cost: float  # summed over auto-routed tickets
    escalation_cost: float  # human_cost * n_escalated
    total_cost: float
    n: int  # tickets evaluated

    @property
    def cost_per_ticket(self) -> float:
        return self.total_cost / max(self.n, 1)


DEFAULT_THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def sweep_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    urgency: np.ndarray,
    cost: CostConfig = DEFAULT_COSTS,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
) -> list[ThresholdPoint]:
    """Escalate tickets with confidence < t to a human; tally total cost per t."""
    costs = per_ticket_cost(y_true, y_pred, urgency, cost)
    correct = y_true == y_pred
    n = len(y_true)
    points = []
    for t in thresholds:
        auto = confidence >= t
        n_auto = int(auto.sum())
        misroute = float(costs[auto].sum())
        escalation = cost.human_cost * (n - n_auto)
        points.append(
            ThresholdPoint(
                threshold=float(t),
                coverage=n_auto / n,
                auto_accuracy=float(correct[auto].mean()) if n_auto else float("nan"),
                misroute_cost=misroute,
                escalation_cost=escalation,
                total_cost=misroute + escalation,
                n=n,
            )
        )
    return points


def best_threshold(points: list[ThresholdPoint], min_auto_accuracy: float | None = None) -> ThresholdPoint:
    """Lowest-total-cost threshold, optionally subject to an accuracy floor."""
    candidates = points
    if min_auto_accuracy is not None:
        candidates = [p for p in points if p.auto_accuracy >= min_auto_accuracy] or points[-1:]
    return min(candidates, key=lambda p: (p.total_cost, p.threshold))


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    urgency: np.ndarray,
    threshold: float,
    cost: CostConfig = DEFAULT_COSTS,
) -> ThresholdPoint:
    return sweep_thresholds(y_true, y_pred, confidence, urgency, cost, np.array([threshold]))[0]


# --- Calibration: does "90% confident" mean right 90% of the time? ----------
# Escalation only works if confidence separates right from wrong answers, so a
# badly calibrated model can't be rescued by a threshold.


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def calibration_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[CalibrationBin]:
    """Equal-width confidence bins over [0, 1]; empty bins are omitted."""
    idx = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            bins.append(
                CalibrationBin(
                    lo=b / n_bins,
                    hi=(b + 1) / n_bins,
                    count=int(mask.sum()),
                    mean_confidence=float(confidence[mask].mean()),
                    accuracy=float(correct[mask].mean()),
                )
            )
    return bins


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by bin size (0 = perfect)."""
    n = len(confidence)
    return float(sum(b.count / n * abs(b.accuracy - b.mean_confidence) for b in calibration_bins(confidence, correct, n_bins)))


def confident_error_share(confidence: np.ndarray, correct: np.ndarray, level: float = 0.9) -> float:
    """Share of mistakes made at >= ``level`` confidence: errors no threshold below ``level`` can catch."""
    wrong = ~correct
    return float((confidence[wrong] >= level).mean()) if wrong.any() else float("nan")
