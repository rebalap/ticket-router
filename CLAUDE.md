# Ticket Router

Support-ticket router: labels **intent** (8 SaaS intents) and **urgency** (low/normal/high), routes to a team, and escalates low-confidence tickets to human triage. It compares a TF-IDF baseline, zero-shot `facebook/bart-large-mnli`, and fine-tuned `distilbert-base-uncased` on the Bitext customer-support dataset. A Streamlit app turns the evals into a PM decision: which model, what threshold, what it costs.

**Scope:** a standalone portfolio/learning project, not part of the Autism Therapy Platform. The PM OS stage gates from the global CLAUDE.md do not apply here.

## Commands
```bash
make demo          # prepare data + baseline if missing, then open the app (port 8501 or next free)
make test          # pytest (metrics, urgency, brief, Streamlit AppTest smoke tests)
make baseline      # data split + TF-IDF model + its predictions (~1 min)
make train         # fine-tune DistilBERT (~10-15 min on Apple MPS)
make predict       # DistilBERT (full) + zero-shot (1,000-ticket sample) predictions, incl. the hard set
make eval          # outputs/report.md + plots
make brief         # outputs/decision-brief.md
make demo-assets   # refresh committed demo_assets/ from outputs/
uv run ticket-route "ticket text" --model distilbert --threshold 0.6
```
Use `uv` for everything. Don't call pip or bare python. Run commands from the project root, because paths like `data/`, `models/`, and `outputs/` are relative to it.

## Architecture
- `src/ticket_router/labels.py`: **source of truth** for the taxonomy (Bitext's 27 intents → 8), intent → team, misroute severities, zero-shot descriptions.
- `urgency.py`: keyword heuristic (anger / time pressure / risk) plus an intent prior. Bitext has no urgency labels.
- `data.py`: download Bitext, clean `{{placeholders}}`, write stratified 80/10/10 splits to `data/`.
- `models.py`: `TfidfClassifier`, `ZeroShotClassifier`, `DistilBertClassifier`; all expose `predict_proba(texts) -> (n, 8)` in `INTENTS` order. Add new models to `MODEL_REGISTRY`.
- `train.py` (`--model tfidf|distilbert`), `predict.py` (writes `outputs/preds_<model>_<split>.csv` and `latency.json`).
- `metrics.py`: `CostConfig`, cost matrix, cost-weighted confusion, threshold sweep. Pure numpy, no I/O.
- `results.py`: loads predictions (`outputs/` first, then `demo_assets/`), aligns models on shared ticket ids, `summarize()` per model. Shared by the eval CLI, the app, and the brief.
- `evals/hard_tickets.csv`: 80 hand-written realistic tickets with hand-labelled urgency and difficulty tags (`load_split("hard")`). Committed; see `evals/README.md` for the labelling rules.
- Calibration lives in `metrics.py` (`calibration_bins`, `expected_calibration_error`, `confident_error_share`). `results.evaluate_hard_set` scores the hard set at the val-chosen threshold.
- `brief.py`: markdown decision brief, with an automatic robustness warning when the benchmark winner fails the hard set. `evaluate.py`: report + PNGs. `plots.py`: figures. `router.py`: `TicketRouter`. `demo.py`: `make demo`.
- `app/`: Streamlit UI only (`Home.py`, `pages/1-4`, shared helpers in `app/ui.py`). Keep logic in `src/` and call it from the app.

## Conventions
- **Choose the threshold on val and report on test.** Never tune on test.
- The **hard set is a test set.** Never train on it or tune on it. If you change its labels, record why in `evals/README.md`.
- Compare models on the **same ticket ids** (`results.align_predictions`), because zero-shot runs on a sample.
- Costs are business inputs, in units of "one low-stakes misroute = 1.0". Change defaults in `labels.MISROUTE_SEVERITY` and `metrics.py`, not in the app.
- Any change to `metrics.py`, `urgency.py`, or `labels.py` needs a test in `tests/`. Run `make test` before you say a task is finished.
- After changing the taxonomy or urgency rules: `make baseline && make train && make predict && make demo-assets`, then commit `demo_assets/`.
- `data/`, `models/`, `outputs/` are gitignored. `demo_assets/` is committed (~1.4 MB).

## Gotchas
- Bitext is synthetic, templated, and e-commerce flavoured. Supervised models hit ~99.5%+, so don't present these numbers as production expectations.
- Bitext's `CANCEL` category is only "check cancellation fee". Real cancellations come from `cancel_order` and `delete_account` (see `labels.BITEXT_TO_INTENT`).
- "Bug" has no real equivalent; `technical_issue` is a proxy built from `registration_problems`.
- High urgency is rare (~1.3%), so high-urgency accuracy is measured on only a few dozen test tickets.
- The `flags` column name clashes with `DataFrame.flags`. Use `df["flags"]`, not `df.flags`.
- transformers is 5.x: `TrainingArguments` has no `warmup_ratio` (we compute `warmup_steps`) and `Trainer` takes `processing_class`, not `tokenizer`.
- The zero-shot model downloads 1.6 GB on first use. Don't trigger it in tests.
