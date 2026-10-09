"""Compare models: accuracy, cost-weighted confusion, and escalation threshold.

The escalation threshold is chosen on the val split and reported on test, so the
headline numbers aren't tuned on the data they're reported on. Models are
compared on the intersection of ticket ids they were each run on (zero-shot is
usually run on a sample).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ticket_router import metrics
from ticket_router.labels import INTENTS
from ticket_router.data import load_split
from ticket_router.plots import cost_confusion_figure, reliability_figure, threshold_sweep_figure
from ticket_router.results import ALL_MODELS, OUTPUT_DIR, ModelSummary, summarize_all, summary_table
from ticket_router.urgency import score_urgency


def _matrix_md(m: np.ndarray, fmt: str) -> str:
    short = [i[:10] for i in INTENTS]
    lines = ["| true \\ pred | " + " | ".join(short) + " |", "|---" * (len(INTENTS) + 1) + "|"]
    for i, name in enumerate(INTENTS):
        lines.append(f"| **{name}** | " + " | ".join(format(v, fmt) if v else "·" for v in m[i]) + " |")
    return "\n".join(lines)


def _hard_set_section(summaries: list[ModelSummary], out: Path) -> str:
    with_hard = [s for s in summaries if s.hard]
    if not with_hard:
        return "## Hard set\n\nNo hard-set predictions. Run `uv run ticket-predict --model <name> --splits hard`.\n"
    reliability_figure({s.label: s.hard.calibration for s in with_hard}, "Calibration: hard set").savefig(
        out / "calibration_hard.png", dpi=120
    )
    rows = ["| Model | Bitext test accuracy | **Hard set accuracy** | Drop | ECE (hard) | Mistakes at ≥90% confidence |",
            "|---|---:|---:|---:|---:|---:|"]
    for s in with_hard:
        rows.append(
            f"| {s.label} | {s.accuracy:.1%} | **{s.hard.accuracy:.1%}** | {s.hard.accuracy - s.accuracy:+.1%} "
            f"| {s.hard.ece:.3f} | {s.hard.confident_error_share:.0%} |"
        )
    tags = sorted({t for s in with_hard for t in s.hard.by_tag})
    tag_rows = ["| Tag | n | " + " | ".join(s.label for s in with_hard) + " |", "|---|---:|" + "---:|" * len(with_hard)]
    for t in tags:
        n = with_hard[0].hard.by_tag[t][0]
        tag_rows.append(f"| {t} | {n} | " + " | ".join(f"{s.hard.by_tag[t][1]:.0%}" for s in with_hard) + " |")
    misses = []
    for s in with_hard:
        top = s.hard.misses.head(3)
        misses.append(f"**{s.label}**\n" + "\n".join(
            f"- \"{r.text}\": {r.intent} → **{r.pred}** ({r.confidence:.0%} confident)" for r in top.itertuples()
        ))
    hard = load_split("hard")
    rule = [score_urgency(t, i) for t, i in zip(hard["text"], hard["intent"])]
    agree = sum(r == h for r, h in zip(rule, hard["urgency"])) / len(hard)
    rule_high = sum(r == "high" and h == "high" for r, h in zip(rule, hard["urgency"])) / max((hard["urgency"] == "high").sum(), 1)
    return (
        f"## Hard set: {with_hard[0].hard.n} hand-written tickets\n\n"
        "Realistic SaaS tickets with typos, sarcasm, indirect requests, and product bugs Bitext never shows "
        "(see `evals/README.md`). Same thresholds as above (chosen on Bitext val).\n\n"
        + "\n".join(rows) + "\n\n**Accuracy by difficulty tag:**\n\n" + "\n".join(tag_rows)
        + "\n\n**Most expensive hard-set mistakes:**\n\n" + "\n\n".join(misses)
        + f"\n\n**Keyword urgency rule vs hand labels:** agrees on {agree:.0%} of tickets, "
        f"and catches {rule_high:.0%} of the hand-labelled high-urgency ones.\n\n"
        "![calibration on hard set](calibration_hard.png)\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--human-cost", type=float, default=metrics.HUMAN_TRIAGE_COST,
                    help="Cost of one human triage, in misroute units")
    ap.add_argument("--min-auto-accuracy", type=float, default=None, help="Optional accuracy floor for auto-routed")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    cost = metrics.CostConfig(human_cost=args.human_cost)
    summaries = summarize_all(args.models, cost, args.min_auto_accuracy)
    if not summaries:
        raise SystemExit("No predictions found — run `make baseline` or `uv run ticket-predict --model <name>`.")
    skipped = sorted(set(args.models) - {s.model for s in summaries})
    if skipped:
        print(f"Skipping models without val+test predictions: {skipped}")

    table = summary_table(summaries)
    print(table.T.to_string(float_format=lambda v: f"{v:.4f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    sections = []
    for s in summaries:
        cost_confusion_figure(s.cost_confusion, f"{s.label}: cost-weighted confusion (test)").savefig(
            args.out / f"cost_confusion_{s.model}.png", dpi=120
        )
        threshold_sweep_figure(s.val_sweep, s.threshold, f"{s.label}: threshold sweep (val)").savefig(
            args.out / f"threshold_{s.model}.png", dpi=120
        )
        worst = "\n".join(f"- **{c.true} → {c.pred}**: {c.count} tickets, cost {c.cost:.1f}" for c in s.worst_confusions())
        sections.append(
            f"## {s.label}\n\n"
            f"![cost confusion](cost_confusion_{s.model}.png) ![threshold](threshold_{s.model}.png)\n\n"
            f"**Most expensive confusions (test):**\n{worst or '- none'}\n\n"
            f"**Cost-weighted confusion (test, summed cost):**\n\n{_matrix_md(s.cost_confusion, '.1f')}\n\n"
            f"**Raw confusion counts (test):**\n\n{_matrix_md(s.count_confusion, '.0f')}\n"
        )

    reliability_figure({s.label: s.calibration for s in summaries}, "Calibration: Bitext test").savefig(
        args.out / "calibration_test.png", dpi=120
    )
    calibration = (
        "## Calibration: does the confidence score mean what it says?\n\n"
        "Escalating low-confidence tickets only helps if confidence separates right answers from wrong ones. "
        "ECE (expected calibration error) is the average gap between confidence and actual accuracy; 0 is perfect.\n\n"
        "| Model | ECE (Bitext test) | Mistakes at ≥90% confidence (test) |\n|---|---:|---:|\n"
        + "\n".join(f"| {s.label} | {s.ece:.3f} | {s.confident_error_share:.0%} |" for s in summaries)
        + "\n\n![calibration on test](calibration_test.png)\n"
    )

    report = args.out / "report.md"
    report.write_text(
        "# Ticket router evaluation\n\n"
        f"Human triage cost = {cost.human_cost:g} (1.0 = one low-stakes misroute). "
        f"Threshold chosen on val, reported on test. Compared on {summaries[0].n_test} common test tickets.\n\n"
        + table.T.to_markdown(floatfmt=".4f")
        + "\n\n"
        + _hard_set_section(summaries, args.out)
        + "\n"
        + calibration
        + "\n"
        + "\n".join(sections)
    )
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
