---
description: Regenerate predictions for available models, run the eval, and summarize the results
argument-hint: "[--human-cost N] [--min-auto-accuracy X]"
---
1. Run `make baseline` if `outputs/preds_tfidf_test.csv` is missing. Run `uv run ticket-predict --model distilbert` if `models/distilbert-ticket/config.json` exists. Don't run zero-shot unless the user asks (1.6 GB download).
2. Run `uv run ticket-eval $ARGUMENTS`.
3. Read `outputs/report.md` and summarize: the accuracy table, the threshold chosen per model, the 3 most expensive confusions for the best model, and anything surprising. Example: a threshold of 0.0 means escalation never pays at this human cost.
4. Remind the user that thresholds come from val and numbers are reported on test.
