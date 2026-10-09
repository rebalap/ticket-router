"""End-to-end router: ticket text -> intent, urgency, team (or human escalation)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from ticket_router.labels import HUMAN_TRIAGE_TEAM, INTENT_TO_TEAM, INTENTS
from ticket_router.models import MODEL_REGISTRY, IntentClassifier, load_model
from ticket_router.urgency import score_urgency, urgency_signals


@dataclass
class RoutingDecision:
    text: str
    intent: str
    confidence: float
    urgency: str
    urgency_signals: list[str]
    team: str
    escalated: bool


class TicketRouter:
    def __init__(self, classifier: IntentClassifier, threshold: float):
        self.classifier = classifier
        self.threshold = threshold

    def route_many(self, texts: list[str], threshold: float | None = None) -> list[RoutingDecision]:
        threshold = self.threshold if threshold is None else threshold
        probs = self.classifier.predict_proba(texts)
        decisions = []
        for text, p in zip(texts, probs):
            k = int(p.argmax())
            intent, conf = INTENTS[k], float(p[k])
            escalated = conf < threshold
            decisions.append(
                RoutingDecision(
                    text=text,
                    intent=intent,
                    confidence=round(conf, 4),
                    urgency=score_urgency(text, intent),
                    urgency_signals=urgency_signals(text),
                    team=HUMAN_TRIAGE_TEAM if escalated else INTENT_TO_TEAM[intent],
                    escalated=escalated,
                )
            )
        return decisions

    def route(self, text: str, threshold: float | None = None) -> RoutingDecision:
        return self.route_many([text], threshold)[0]

    def intent_probabilities(self, text: str) -> dict[str, float]:
        probs = self.classifier.predict_proba([text])[0]
        return {intent: float(p) for intent, p in zip(INTENTS, probs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("texts", nargs="+", help="Ticket text(s) to route")
    ap.add_argument("--model", default="distilbert", choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--threshold", type=float, default=0.5, help="Use the value `ticket-eval` picked")
    args = ap.parse_args()

    router = TicketRouter(load_model(args.model), args.threshold)
    for decision in router.route_many(args.texts):
        print(json.dumps(asdict(decision), indent=2))


if __name__ == "__main__":
    main()
