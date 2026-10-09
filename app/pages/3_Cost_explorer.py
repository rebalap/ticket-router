"""PM page: set business costs, see the threshold respond, export a brief."""

import numpy as np
import pandas as pd
import streamlit as st
from ui import COST_KEY, FLOOR_KEY, get_cost, get_floor, no_predictions_message, page_setup, summaries

from ticket_router import metrics
from ticket_router.brief import recommend, render_brief
from ticket_router.labels import INTENT_TO_TEAM, INTENTS
from ticket_router.plots import cost_confusion_figure, threshold_sweep_figure
from ticket_router.results import load_aligned, prediction_arrays

page_setup("Cost explorer")
st.title("🎚️ Cost & threshold explorer")
st.markdown(
    "Misroutes don't all cost the same. An angry cancellation sent to billing is far worse than a newsletter "
    "question sent to ops. Set what each mistake costs (in units of **one low-stakes misroute = 1.0**), and "
    "the tool picks the escalation threshold that minimises total cost."
)

cost = get_cost()

with st.sidebar:
    st.header("Cost assumptions")
    human = st.slider("Human triage, per ticket", 0.1, 10.0, float(cost.human_cost), 0.1)
    discount = st.slider(
        "Same-team misroute discount", 0.0, 1.0, float(cost.same_team_discount), 0.05,
        help="e.g. billing ↔ refund: same team, just re-tag it",
    )
    st.caption("Urgency multipliers")
    mult = {
        u: st.number_input(u, 0.0, 10.0, float(cost.urgency_multiplier[u]), 0.25, key=f"mult_{u}")
        for u in ("low", "normal", "high")
    }
    use_floor = st.checkbox("Require a minimum auto-routed accuracy", value=get_floor() is not None)
    floor = st.slider("Minimum auto-routed accuracy", 0.80, 1.0, get_floor() or 0.99, 0.005) if use_floor else None
    if st.button("Reset to defaults"):
        st.session_state.pop(COST_KEY, None)
        st.session_state.pop(FLOOR_KEY, None)
        st.rerun()

st.subheader("Misroute severity by true intent")
sev_df = pd.DataFrame(
    {"intent": INTENTS, "owning team": [INTENT_TO_TEAM[i] for i in INTENTS], "severity": [cost.severity[i] for i in INTENTS]}
)
edited = st.data_editor(
    sev_df,
    disabled=["intent", "owning team"],
    hide_index=True,
    column_config={"severity": st.column_config.NumberColumn(min_value=0.0, max_value=50.0, step=0.5)},
    use_container_width=False,
)

st.session_state[COST_KEY] = metrics.CostConfig(
    severity=dict(zip(edited["intent"], edited["severity"].astype(float))),
    same_team_discount=discount,
    urgency_multiplier=mult,
    human_cost=human,
)
st.session_state[FLOOR_KEY] = floor
cost = get_cost()

results = summaries()
if not results:
    no_predictions_message()
    st.stop()

by_name = {s.label: s for s in results}
default = recommend(results).label
pick = st.radio("Model", list(by_name), index=list(by_name).index(default), horizontal=True)
s = by_name[pick]

override = st.toggle("Try my own threshold instead of the cost-optimal one")
threshold = st.slider("Threshold", 0.0, 1.0, s.threshold, 0.01) if override else s.threshold

yt, pt, ct, ut = prediction_arrays(load_aligned([r.model for r in results], "test")[s.model])
point = metrics.evaluate_at_threshold(yt, pt, ct, ut, threshold, cost)
saving = 1 - point.cost_per_ticket / cost.human_cost

m1, m2, m3, m4 = st.columns(4)
m1.metric("Auto-routed", f"{point.coverage:.1%}")
m2.metric("Accuracy when auto-routed", f"{point.auto_accuracy:.2%}" if not np.isnan(point.auto_accuracy) else "—")
m3.metric("Escalated per 1,000 tickets", f"{round((1 - point.coverage) * 1000)}")
m4.metric("Cost per ticket", f"{point.cost_per_ticket:.3f}", f"{saving:.0%} vs all-human", delta_color="normal")
st.caption(
    f"Threshold **{threshold:.2f}** "
    + ("(your override)" if override else "(cost-optimal on the validation split)")
    + f". Results are on {point.n} test tickets. All-human triage would cost {cost.human_cost:.3f} per ticket."
)

c1, c2 = st.columns(2)
with c1:
    st.pyplot(threshold_sweep_figure(s.val_sweep, threshold, f"{s.label}: cost vs threshold (val)"))
with c2:
    st.pyplot(cost_confusion_figure(s.cost_confusion, f"{s.label}: where the cost goes (test)"))

st.subheader("Take it to your stakeholders")
brief = render_brief(results, cost, chosen=s)
st.download_button("⬇️ Download decision brief (.md)", brief, file_name="routing-decision-brief.md", mime="text/markdown")
with st.expander("Preview brief"):
    st.markdown(brief)
