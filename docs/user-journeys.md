# User journeys

## 1. Product manager (primary user)
**Question:** *Should we auto-route support tickets? If so, with which model, at what confidence threshold, and what will it cost?*

| Step | Where | What they do | What they get |
|---|---|---|---|
| 1 | `make demo` → Home | Read the 4-step overview and model status | Orientation in under 1 minute |
| 2 | Route a ticket | Click the samples, then paste tickets from their own queue | Intent, confidence across all intents, urgency + signals, team, escalation |
| 3 | Compare models | Read the side-by-side table and the build-vs-buy trade-offs | Which approach wins, and what each needs (labels, infra, time) |
| 4 | Cost explorer | Set severities, same-team discount, urgency multipliers, human cost **with Support and Finance** | Cost-optimal threshold, auto-route %, escalations per 1k, cost per ticket vs all-human |
| 5 | Error review | Open the most expensive confusions and the escalation queue | Where the taxonomy is ambiguous, and how much escalation catches |
| 6 | Cost explorer → Download brief | Export the markdown decision brief | Recommendation, risks, success metrics, and limitations, ready for a PRD |

**Decisions this supports**
- *Automate or not:* compare cost per ticket against the all-human baseline.
- *Build vs buy:* zero-shot needs no labelled data and is weaker. Fine-tuning needs labels and is strong.
- *Threshold policy:* the threshold follows from what a human triage costs relative to a misroute. If staffing changes, re-run it.
- *Success metrics for the PRD:* misroute rate on high-urgency tickets, auto-route coverage, accuracy when auto-routed, escalation queue volume against triage capacity.

**Worked insight:** with zero-shot and human triage at 1.0, escalation never pays (threshold 0.0). Drop human triage to 0.2 and the optimal policy escalates ~78% of tickets, which lifts accuracy on what's still auto-routed from 65% to 88%. The cost assumptions *are* the product decision.

## 2. Support operations lead
**Question:** *Will this send the right tickets to my teams, and what lands in my triage queue?*
- **Home → How routing works:** check the intent → team table.
- **Cost explorer:** argue for the severities ("a missed cancellation costs us a customer").
- **Error review → Escalation queue:** see what humans would get, and how often the model's guess was already right.

## 3. ML engineer
**Question:** *How do I improve or extend the model?*
```bash
make baseline                         # data + TF-IDF in ~1 min
make train && make predict && make eval
uv run ticket-route "ticket text" --model distilbert --threshold 0.6
```
- Extend the taxonomy in `src/ticket_router/labels.py` (`BITEXT_TO_INTENT`, `INTENT_TO_TEAM`, `MISROUTE_SEVERITY`, `ZERO_SHOT_DESCRIPTIONS`).
- Add a model: implement `predict_proba(texts) -> (n, 8)` and register it in `models.MODEL_REGISTRY`.
- After changes, run `make demo-assets` and commit `demo_assets/`, so the demo shows current numbers.

## The ticket's own path
```
ticket text
  → classifier → probabilities over 8 intents → confidence = max probability
  → confidence < threshold ?  ── yes → human_triage queue
                              └─ no  → team that owns the intent
  → urgency (intent prior + anger / time-pressure / risk keywords) → queue priority
```
