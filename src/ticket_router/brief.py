"""Generate a PM-facing routing decision brief (markdown) from eval results."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from ticket_router import metrics
from ticket_router.labels import INTENT_TO_TEAM, INTENTS
from ticket_router.results import OUTPUT_DIR, ModelSummary, summarize_all, summary_table


def recommend(summaries: list[ModelSummary]) -> ModelSummary:
    """Lowest expected total cost per ticket (misroutes + human triage)."""
    return min(summaries, key=lambda s: s.total_cost_per_ticket)


def _robustness_section(summaries: list[ModelSummary], rec: ModelSummary, all_human: float) -> str:
    with_hard = [s for s in summaries if s.hard]
    if not with_hard:
        return ""
    rows = "\n".join(
        f"| {s.label} | {s.accuracy:.1%} | {s.hard.accuracy:.1%} | {s.hard.accuracy_high_urgency:.0%} "
        f"| {s.hard.cost_per_ticket:.2f} | {s.hard.confident_error_share:.0%} |"
        for s in with_hard
    )
    warnings = []
    if rec.hard:
        best_hard = min(with_hard, key=lambda s: s.hard.cost_per_ticket)
        if best_hard.model != rec.model:
            warnings.append(
                f"On realistic tickets the cheapest model is **{best_hard.label}** "
                f"({best_hard.hard.cost_per_ticket:.2f}/ticket vs {rec.hard.cost_per_ticket:.2f}), "
                "not the benchmark winner."
            )
        if rec.hard.cost_per_ticket > all_human:
            warnings.append(
                f"On realistic tickets the recommended model costs **more than all-human triage** "
                f"({rec.hard.cost_per_ticket:.2f} vs {all_human:.2f})."
            )
        if rec.hard.confident_error_share >= 0.5:
            warnings.append(
                f"{rec.hard.confident_error_share:.0%} of its hard-set mistakes are made at ≥90% confidence, "
                "so a confidence threshold can't catch them."
            )
    flag = (
        "\n> ⚠️ **Robustness check fails.** " + " ".join(warnings) + " Validate on real tickets before launch.\n"
        if warnings
        else ""
    )
    return f"""
## Robustness check: {with_hard[0].hard.n} hand-written realistic tickets
| Model | Benchmark accuracy | Hard-set accuracy | Hard-set accuracy (high urgency) | Hard-set cost / ticket | Mistakes at ≥90% confidence |
|---|---:|---:|---:|---:|---:|
{rows}
{flag}"""


def render_brief(
    summaries: list[ModelSummary],
    cost: metrics.CostConfig = metrics.DEFAULT_COSTS,
    chosen: ModelSummary | None = None,
) -> str:
    rec = chosen or recommend(summaries)
    all_human = cost.human_cost
    saving = 1 - rec.total_cost_per_ticket / all_human if all_human else float("nan")
    escalated_per_1k = round((1 - rec.coverage) * 1000)

    comparison = summary_table(summaries).T.to_markdown(floatfmt=".3f")
    cost_rows = "\n".join(
        f"| {i} | {INTENT_TO_TEAM[i]} | {cost.severity[i]:g} |" for i in INTENTS
    )
    urgency_rows = ", ".join(f"{k} ×{v:g}" for k, v in cost.urgency_multiplier.items())
    risks = "\n".join(
        f"- **{c.true} → {c.pred}**: {c.count} test ticket{'s' if c.count != 1 else ''}, {c.cost:.1f} cost units"
        for c in rec.worst_confusions(5)
    ) or "- No misroutes on the test sample."

    robustness = _robustness_section(summaries, rec, all_human)

    return f"""# Ticket routing: decision brief

_Generated {dt.date.today().isoformat()} from {rec.n_test} held-out test tickets (Bitext, remapped to 8 SaaS intents)._

## Recommendation
Auto-route with **{rec.label}** at a confidence threshold of **{rec.threshold:.2f}**; tickets below it go to human triage.

| Outcome (test set) | Value |
|---|---|
| Auto-routed share | {rec.coverage:.1%} |
| Accuracy on auto-routed tickets | {rec.auto_accuracy:.2%} |
| Escalated to humans | ~{escalated_per_1k} per 1,000 tickets |
| Expected cost / ticket | {rec.total_cost_per_ticket:.3f} vs {all_human:.3f} if humans triage everything ({saving:.0%} lower) |
| Latency | {f"{rec.latency_ms:.1f} ms/ticket" if rec.latency_ms is not None else "n/a"} |

The threshold was chosen on the validation split and is reported on the test split.
{robustness}
## Model comparison
{comparison}

## Cost assumptions (business inputs, so revisit these with Support and Finance)
Units: 1.0 = one low-stakes misroute. Human triage = {cost.human_cost:g} per ticket.
Same-team misroutes are discounted to ×{cost.same_team_discount:g}. Urgency multipliers: {urgency_rows}.

| True intent | Owning team | Misroute severity |
|---|---|---|
{cost_rows}

## Top risks: most expensive confusions for the recommended model
{risks}

## Proposed success metrics and guardrails
- **North star:** misroute rate on high-urgency tickets (currently {1 - rec.accuracy_high_urgency:.2%} before escalation).
- **Auto-route coverage** ≥ {rec.coverage:.0%}, with **auto-routed accuracy** ≥ {rec.auto_accuracy:.1%}.
- **Escalation queue volume** stays within human triage capacity (~{escalated_per_1k}/1k tickets).
- **Guardrail:** re-tune the threshold whenever cost assumptions change; monitor weekly for drift in the confidence distribution.

## Known limitations
- Bitext is synthetic and e-commerce flavoured. Validate on a sample of real tickets before launch.
- Urgency is a keyword heuristic, not a learned label.
- "Bug" is approximated by sign-up/registration problems; there are no real bug reports in the data.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-cost", type=float, default=metrics.HUMAN_TRIAGE_COST)
    ap.add_argument("--min-auto-accuracy", type=float, default=None)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "decision-brief.md")
    args = ap.parse_args()

    cost = metrics.CostConfig(human_cost=args.human_cost)
    summaries = summarize_all(cost=cost, min_auto_accuracy=args.min_auto_accuracy)
    if not summaries:
        raise SystemExit("No predictions found — run `make baseline` or `make predict` first.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_brief(summaries, cost))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
