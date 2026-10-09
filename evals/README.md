# Hard test set (`hard_tickets.csv`)

80 hand-written tickets (10 per intent), written to resemble real SaaS support traffic rather than Bitext's clean templates. Bitext saturates: supervised models score 99%+ on it. This set measures how the models hold up on realistic input.

| Column | Meaning |
|---|---|
| `id` | 900000+ so it never collides with Bitext ids |
| `intent` | The intent that should decide routing |
| `urgency` | **Hand-labelled** (low / normal / high). It weights misroute cost here, and is used to check the keyword urgency rule |
| `tags` | What makes the ticket hard (`;`-separated, see below) |
| `secondary_intent` | Second intent for multi-intent tickets |

**Tags:** `typo` (misspellings, slang), `terse` (a few words), `indirect` (intent implied, not stated), `sarcasm`, `saas_terms` (seats, SSO, prorated, workspace…), `out_of_distribution` (real product bugs such as 500 errors and API timeouts, which Bitext never shows; its `technical_issue` is only sign-up problems), `multi_intent`.

**Labelling rules**
- Multi-intent: label the **higher-stakes** intent, the one whose misroute costs more. Example: "charged twice… cancel everything" is `cancellation`, with `billing` secondary.
- Urgency is **high** for anger or sarcasm aimed at the company, an explicit deadline, security or fraud risk, a production outage, or a repeated unresolved contact.
- Urgency is **low** for informational questions with no problem attached.

This set was written by the project author. It's small (n=80), so treat per-tag numbers as directional.
