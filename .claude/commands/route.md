---
description: Route one or more tickets and explain the decision
argument-hint: "\"ticket text\" [\"another ticket\"]"
---
Route these tickets: $ARGUMENTS

1. Pick the best available model: `distilbert` if `models/distilbert-ticket/config.json` exists, else `tfidf` (run `make baseline` if needed).
2. Use the cost-optimal threshold for that model from the latest `outputs/report.md` if it exists, otherwise 0.5.
3. Run `uv run ticket-route <tickets> --model <model> --threshold <t>`.
4. For each ticket, explain in plain language: intent, confidence, urgency (and which signals fired), the team, and, if escalated, which team it would otherwise have gone to.
