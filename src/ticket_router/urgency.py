"""Rule-based urgency scoring.

Bitext has no urgency labels, so urgency is a transparent heuristic: an intent
prior plus keyword signals for anger, time pressure, and money/security risk.
It is used both to prioritise routed tickets and to weight misroute costs in
evaluation (an urgent misroute costs more than a calm one).
"""

from __future__ import annotations

import re

URGENCY_LEVELS = ["low", "normal", "high"]
URGENCY_COST_MULTIPLIER = {"low": 0.5, "normal": 1.0, "high": 2.0}

_SIGNALS: dict[str, list[str]] = {
    "anger": [
        r"damn\w*", r"\bf+u+c+k\w*", r"\bshit\w*", r"\bcrap\w*", r"\bhell\b", r"\bbloody\b",
        r"\bfreaking\b", r"\bfrickin\w*",
        r"\bunacceptable\b", r"\bridiculous\b", r"\bworst\b", r"\bterrible\b",
        r"\bfurious\b", r"\bangry\b", r"\bfed up\b", r"\bsick of\b", r"\bscam\w*",
        r"!{2,}",
    ],
    "time_pressure": [
        r"\burgent\w*", r"\basap\b", r"\bimmediately\b", r"\bright now\b",
        r"\bright away\b", r"\btoday\b", r"\bstill (?:waiting|haven'?t|not)\b",
    ],
    "risk": [
        r"\bfraud\w*", r"\bhack\w*", r"\bstolen\b", r"\bunauthori[sz]ed\b",
        r"\bcharged twice\b", r"\bdouble[- ]charged\b", r"\bchargeback\b",
        r"\blocked out\b", r"\bcan'?t (?:log ?in|access|sign ?in)\b",
        r"\blawyer\b", r"\blegal action\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in _SIGNALS.items()}

# Points per signal group; a risk signal (fraud, double charge, lockout) is
# urgent on its own.
_SIGNAL_WEIGHTS = {"anger": 1, "time_pressure": 1, "risk": 2}
# Intents that are churn- or money-sensitive.
_ELEVATED_INTENTS = {"cancellation", "complaint", "refund"}
# Intents that default to low urgency when no other signal fires.
_LOW_STAKES_INTENTS = {"general", "order_fulfillment"}


def urgency_signals(text: str) -> list[str]:
    """Return the names of the signal groups that fire on this text."""
    return [name for name, pats in _COMPILED.items() if any(p.search(text) for p in pats)]


def score_urgency(text: str, intent: str) -> str:
    """Map (ticket text, intent) to one of ``URGENCY_LEVELS``."""
    score = sum(_SIGNAL_WEIGHTS[s] for s in urgency_signals(text)) + (1 if intent in _ELEVATED_INTENTS else 0)
    if score >= 2:
        return "high"
    if score == 1:
        return "normal"
    return "low" if intent in _LOW_STAKES_INTENTS else "normal"
