"""Ticket Router: landing page."""

import streamlit as st
from ui import MODEL_REQUIREMENTS, is_live, model_label, page_setup, prediction_source

from ticket_router.labels import INTENT_TO_TEAM, INTENTS

page_setup("Home")

st.title("🎫 Support Ticket Router")
st.markdown(
    "Reads an incoming support ticket, labels its **intent** and **urgency**, and routes it to a team. "
    "Low-confidence tickets are **escalated to a human**. The tool also compares three ways of building "
    "the classifier and answers the product question behind it: *should we automate routing, with which "
    "model, at what threshold, and what will it cost?*"
)

st.subheader("Where to start")
c1, c2, c3, c4 = st.columns(4)
c1.page_link("pages/1_Route_a_ticket.py", label="**1 · Route a ticket**", icon="📨")
c1.caption("Paste a ticket, see intent, urgency, team, and whether it escalates.")
c2.page_link("pages/2_Compare_models.py", label="**2 · Compare models**", icon="⚖️")
c2.caption("Baseline vs zero-shot vs fine-tuned: accuracy, cost, coverage, latency.")
c3.page_link("pages/3_Cost_explorer.py", label="**3 · Cost & threshold explorer**", icon="🎚️")
c3.caption("Set business costs, watch the threshold move, and export a decision brief.")
c4.page_link("pages/4_Error_review.py", label="**4 · Error review**", icon="🔍")
c4.caption("Read the actual tickets behind the most expensive mistakes.")

st.subheader("Model status")
rows = []
for name, (_, how, note) in MODEL_REQUIREMENTS.items():
    rows.append(
        {
            "model": model_label(name),
            "live routing": "✅ ready" if is_live(name) else f"⏳ run `{how}`",
            "eval results": prediction_source(name) or "—",
            "cost to enable": note,
        }
    )
st.dataframe(rows, hide_index=True, use_container_width=True)

with st.expander("How routing works"):
    st.markdown(
        "1. The classifier scores all 8 intents → **confidence** is the top probability.\n"
        "2. If confidence < **threshold**, the ticket goes to **human triage**.\n"
        "3. Otherwise it goes to the team that owns the intent.\n"
        "4. **Urgency** (low / normal / high) comes from intent plus keyword signals for anger, "
        "time pressure, and risk. It sets queue priority and weights misroute cost."
    )
    st.dataframe(
        [{"intent": i, "team": INTENT_TO_TEAM[i]} for i in INTENTS], hide_index=True, use_container_width=False
    )

with st.expander("About the data and its limits"):
    st.markdown(
        "- **Bitext customer-support dataset**: about 27k synthetic tickets across 27 fine-grained intents, "
        "regrouped into 8 SaaS-style intents.\n"
        "- It's templated and clean, so accuracy is far higher than you'd get on real tickets. "
        "Use it to compare approaches and to rehearse the decision, not to promise a launch number.\n"
        "- Urgency is a keyword heuristic (Bitext has no urgency labels).\n"
        "- \"Bug\" is approximated by sign-up/registration problems."
    )
