---
description: Write a PM decision brief on whether and how to automate ticket routing
argument-hint: "[--human-cost N] [--min-auto-accuracy X]"
---
1. Run `uv run ticket-brief $ARGUMENTS` (falls back to committed `demo_assets/` if you have no runs of your own).
2. Read `outputs/decision-brief.md`.
3. Give the user a summary of 5 bullets or fewer: the recommendation, the coverage and cost trade-off, the top risk, and one question to settle with Support or Finance (usually: what does a human triage really cost relative to a misroute?).
4. Point out the dataset caveat: Bitext is synthetic, so validate on real tickets before committing to launch numbers.
