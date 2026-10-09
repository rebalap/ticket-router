# Model card: Ticket Router intent classifiers

## Intended use
Route English SaaS support tickets to one of 5 teams (billing, retention, tech_support, operations, frontline) by intent, and escalate low-confidence tickets to human triage. This is a portfolio/learning project and has not been validated on production traffic.

## Data
- **Source:** Bitext customer-support LLM chatbot training dataset (26,872 rows, 27 intents). After de-duplicating the text, 24,554 tickets remain.
- **Splits:** stratified 80/10/10 by intent (seed 42): 19,643 train, 2,455 val, 2,456 test.
- **Cleaning:** template slots like `{{Order Number}}` are replaced with lowercase words.
- **Taxonomy:** 27 Bitext intents regrouped into 8 (see `labels.BITEXT_TO_INTENT`):

| Intent | Bitext intents | Team |
|---|---|---|
| billing | check_invoice, get_invoice, check_payment_methods, payment_issue | billing |
| refund | check_refund_policy, get_refund, track_refund | billing |
| cancellation | cancel_order, delete_account, check_cancellation_fee | retention |
| complaint | complaint | retention |
| account | create_account, edit_account, switch_account, recover_password | tech_support |
| technical_issue | registration_problems | tech_support |
| order_fulfillment | place/change/track_order, delivery_options/period, shipping address ×2 | operations |
| general | contact_customer_service, contact_human_agent, review, newsletter_subscription | frontline |

## Models
| Model | Training | Notes |
|---|---|---|
| TF-IDF + LogReg | 1–2-grams, sublinear TF, C=10 | Trains in seconds; strong because Bitext is templated |
| Zero-shot BART-large-MNLI | none | Hypothesis: "This customer support ticket is about {description}." Run on a 1,000-ticket stratified sample |
| DistilBERT (uncased) | 2 epochs, lr 5e-5, batch 32, max_len 64, best epoch picked on val | ~10–15 min on Apple MPS |

## Metrics (threshold chosen on val)
| | TF-IDF | Zero-shot | DistilBERT |
|---|---:|---:|---:|
| **Bitext test (1,001 shared tickets)** | | | |
| Accuracy | 99.6% | 64.5% | 99.9% |
| High-urgency accuracy | 100% | 93.3% | 100% |
| Total cost / ticket (all-human = 1.0) | 0.013 | 0.487 | 0.006 |
| Calibration error (ECE, 10 bins) | 0.014 | 0.136 | 0.003 |
| **Hard set (80 hand-written tickets, `evals/`)** | | | |
| Accuracy | 66.2% | 65.0% | 61.3% |
| High-urgency accuracy | 48% | 70% | 39% |
| Total cost / ticket (all-human = 1.0) | 1.63 | 0.70 | 1.69 |
| Calibration error (ECE, 5 bins) | 0.101 | 0.170 | 0.333 |
| Mistakes made at ≥90% confidence | 11% | 0% | 68% |

On realistic tickets, the supervised models lose about 35 points of accuracy and become overconfident. DistilBERT in particular makes most of its mistakes at ≥90% confidence, so threshold-based escalation can't catch them.

## Urgency
Bitext has no urgency labels, so urgency is a **heuristic** (`urgency.py`):
- Signal points: anger keywords (1), time pressure (1), risk such as fraud, double charge, or lockout (2).
- +1 for churn- or money-sensitive intents (cancellation, complaint, refund).
- A score of 2 or more is **high**; 1 is **normal**; 0 is **low** for general/order intents, otherwise normal.
- Distribution: 1.3% high, 59% normal, 40% low. High-urgency metrics rest on only a few dozen test tickets.
- **Checked against hand labels on the hard set:** it agrees on 66% of tickets but catches only 26% of the high-urgency ones. It misses sarcasm, implied anger, and SaaS-specific risks such as outages.

## Limitations
- **Synthetic, templated data.** Real tickets are longer, messier, and often have several intents. Expect much lower accuracy on real tickets.
- **E-commerce flavour.** Orders and shipping dominate; SaaS concepts (plans, seats, integrations, outages) are missing.
- **"Bug" is a proxy.** `technical_issue` comes only from sign-up problems.
- **Single label.** A ticket like "charged twice, cancel my account" really has two intents.
- **Calibration.** Zero-shot is under-confident on the benchmark. DistilBERT is very well calibrated on the benchmark but badly overconfident off-distribution.
- **The hard set is small and author-written** (n=80; 3–20 tickets per tag). Treat it as a stress test, not a replacement for real labelled tickets.

## Before relying on it
Label 200–500 real tickets, re-run the eval on them, and set the cost inputs with Support and Finance.
