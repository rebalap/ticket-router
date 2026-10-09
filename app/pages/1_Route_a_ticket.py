"""Route a single ticket live and explain the decision."""

import pandas as pd
import streamlit as st
from ui import (
    SAMPLE_TICKETS,
    cached_router,
    is_live,
    model_label,
    page_setup,
    summaries,
)

from ticket_router.labels import INTENT_TO_TEAM

page_setup("Route a ticket")
st.title("📨 Route a ticket")

# Best locally-available model first; zero-shot is always offered (downloads on first use).
live = [m for m in ("distilbert", "tfidf") if is_live(m)] + ["zero_shot"]
chosen_thresholds = {s.model: s.threshold for s in summaries()}

left, right = st.columns([2, 1])
with right:
    model = st.selectbox("Model", live, format_func=model_label)
    if model == "zero_shot":
        st.caption("First use downloads ~1.6 GB and is slow on CPU.")
    default_t = chosen_thresholds.get(model, 0.5)
    threshold = st.slider(
        "Escalate below confidence",
        0.0,
        1.0,
        float(default_t),
        0.01,
        help="Default is the cost-optimal threshold picked on val (see the Cost explorer).",
    )
    st.caption(f"Cost-optimal threshold for this model: **{default_t:.2f}**" if model in chosen_thresholds else "")

with left:
    st.write("Try a sample:")
    cols = st.columns(3)
    for k, (name, text) in enumerate(SAMPLE_TICKETS.items()):
        if cols[k % 3].button(name, use_container_width=True):
            st.session_state["ticket_text"] = text
    text = st.text_area("Ticket text", key="ticket_text", height=110, placeholder="Paste a support ticket…")

if not text.strip():
    st.info("Type a ticket or pick a sample above.")
    st.stop()

router = cached_router(model)
decision = router.route(text, threshold)
probs = router.intent_probabilities(text)

st.divider()
a, b, c, d = st.columns(4)
a.metric("Intent", decision.intent)
b.metric("Confidence", f"{decision.confidence:.1%}")
c.metric("Urgency", decision.urgency.upper())
d.metric("Routed to", decision.team)

if decision.escalated:
    st.warning(
        f"**Escalated to human triage.** Confidence {decision.confidence:.1%} is below the "
        f"{threshold:.2f} threshold. Without escalation it would have gone to "
        f"**{INTENT_TO_TEAM[decision.intent]}**."
    )
else:
    st.success(f"Auto-routed to **{decision.team}** (owns `{decision.intent}`).")

why = (
    f"Signals: {', '.join(decision.urgency_signals)}" if decision.urgency_signals else "No anger, time-pressure, or risk signals"
)
st.caption(f"Urgency **{decision.urgency}**. {why}; intent `{decision.intent}`.")

st.subheader("Score for every intent")
chart = pd.DataFrame({"probability": probs}).sort_values("probability", ascending=False)
st.bar_chart(chart, horizontal=True, height=280)
