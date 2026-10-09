# Find uv even when ~/.local/bin isn't on make's PATH (the default uv install location).
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: setup demo baseline train predict eval brief demo-assets test help

setup:      ## Install dependencies
	$(UV) sync

demo:       ## Prepare data + baseline if missing, then open the app (~1 min cold)
	$(UV) run ticket-demo

baseline:   ## Data split + TF-IDF baseline + its predictions (~1 min)
	$(UV) run ticket-prepare
	$(UV) run ticket-train --model tfidf
	$(UV) run ticket-predict --model tfidf

train:      ## Fine-tune DistilBERT (~10-15 min on Apple MPS)
	$(UV) run ticket-train --model distilbert --epochs 2

predict:    ## Predictions for DistilBERT (full) and zero-shot (1,000-ticket sample)
	$(UV) run ticket-predict --model distilbert
	$(UV) run ticket-predict --model zero_shot --limit 1000

eval:       ## Write outputs/report.md and plots
	$(UV) run ticket-eval

brief:      ## Write outputs/decision-brief.md
	$(UV) run ticket-brief

demo-assets: ## Refresh committed sample predictions from outputs/
	$(UV) run python -c "from ticket_router.results import export_demo_assets; export_demo_assets()"

test:       ## Run the test suite
	$(UV) run pytest -q

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
