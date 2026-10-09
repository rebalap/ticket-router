"""Read the tickets behind the most expensive misroutes and the escalation queue."""

import streamlit as st
from ui import get_cost, no_predictions_message, page_setup, summaries

from ticket_router import metrics
from ticket_router.labels import INTENT_TO_ID
from ticket_router.results import load_aligned

page_setup("Error review")
st.title("🔍 Error review")
st.markdown(
    "Accuracy says how often a model is wrong. This page shows **which** mistakes cost the most and the ticket text "
    "behind them, which is where taxonomy problems and missing training data show up."
)

results = summaries()
if not results:
    no_predictions_message()
    st.stop()

by_name = {s.label: s for s in results}
s = by_name[st.radio("Model", list(by_name), horizontal=True)]
test = load_aligned([r.model for r in results], "test")[s.model].copy()
test["misroute_cost"] = metrics.per_ticket_cost(
    test["intent"].map(INTENT_TO_ID).to_numpy(), test["pred"].map(INTENT_TO_ID).to_numpy(), test["urgency"].to_numpy(), get_cost()
)
cols = ["text", "intent", "pred", "confidence", "urgency", "misroute_cost"]

tab_err, tab_esc = st.tabs(["Most expensive mistakes", f"Escalation queue (threshold {s.threshold:.2f})"])

with tab_err:
    worst = s.worst_confusions(10)
    if not worst:
        st.success("No misroutes on this test sample.")
    else:
        st.dataframe(
            [{"true → predicted": f"{c.true} → {c.pred}", "tickets": c.count, "total cost": round(c.cost, 1)} for c in worst],
            hide_index=True,
        )
        cell = st.selectbox("Inspect", worst, format_func=lambda c: f"{c.true} → {c.pred} ({c.count} tickets)")
        rows = test[(test["intent"] == cell.true) & (test["pred"] == cell.pred)]
        st.dataframe(rows[cols].sort_values("misroute_cost", ascending=False), hide_index=True, use_container_width=True)
        wrong = test[test["intent"] != test["pred"]]
        caught = (wrong["confidence"] < s.threshold).mean() if len(wrong) else 0.0
        st.caption(f"{caught:.0%} of this model's mistakes fall below the threshold, so they'd be caught by escalation.")

with tab_esc:
    queue = test[test["confidence"] < s.threshold].sort_values("confidence")
    st.write(
        f"**{len(queue)}** of {len(test)} test tickets would go to human triage. "
        f"The model's guess was right for {(queue['intent'] == queue['pred']).mean():.0%} of them."
        if len(queue)
        else "Nothing would be escalated at this threshold."
    )
    if len(queue):
        st.dataframe(queue[cols], hide_index=True, use_container_width=True)
