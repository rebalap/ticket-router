"""Side-by-side model comparison on the same held-out tickets."""

import pandas as pd
import streamlit as st
from ui import get_cost, no_predictions_message, page_setup, prediction_source, summaries

from ticket_router.brief import recommend
from ticket_router.plots import reliability_figure
from ticket_router.results import summary_table

page_setup("Compare models")
st.title("⚖️ Compare models")

results = summaries()
if not results:
    no_predictions_message()
    st.stop()

rec = recommend(results)
st.markdown(
    f"All models are scored on the **same {results[0].n_test} test tickets**. The escalation threshold is picked "
    f"on the validation split to minimise total cost, using the cost settings from the Cost explorer "
    f"(human triage = {get_cost().human_cost:g})."
)
st.success(
    f"Lowest expected cost: **{rec.label}**. It auto-routes {rec.coverage:.1%} of tickets at "
    f"{rec.auto_accuracy:.2%} accuracy, for {rec.total_cost_per_ticket:.3f} cost per ticket."
)

table = summary_table(results)
fmt = {c: "{:.3f}" for c in table.columns}
fmt.update({c: "{:.1%}" for c in ["accuracy", "accuracy (high urgency)", "auto-routed share", "auto-routed accuracy"]})
fmt.update({"test tickets": "{:.0f}", "threshold (chosen on val)": "{:.2f}", "latency ms / ticket": "{:.2f}"})
st.dataframe(table.style.format(fmt, na_rep="—"), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Accuracy")
    st.bar_chart(table[["accuracy", "auto-routed accuracy"]], stack=False, height=260)
with c2:
    st.subheader("Cost per ticket (lower is better)")
    st.bar_chart(
        table[["misroute cost / ticket (no escalation)", "total cost / ticket (with escalation)"]],
        stack=False,
        height=260,
    )

with_hard = [s for s in results if s.hard]
if with_hard:
    st.header("Stress test: hand-written realistic tickets")
    st.markdown(
        f"Bitext is clean and templated, so the benchmark is nearly saturated. These **{with_hard[0].hard.n} "
        "hand-written tickets** add typos, sarcasm, indirect requests, SaaS jargon, and real product bugs "
        "(`evals/hard_tickets.csv`)."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": s.label,
                    "benchmark accuracy": s.accuracy,
                    "hard-set accuracy": s.hard.accuracy,
                    "hard-set accuracy (high urgency)": s.hard.accuracy_high_urgency,
                    "hard-set cost / ticket": s.hard.cost_per_ticket,
                    "mistakes at ≥90% confidence": s.hard.confident_error_share,
                }
                for s in with_hard
            ]
        )
        .set_index("model")
        .style.format("{:.1%}")
        .format("{:.2f}", subset=["hard-set cost / ticket"]),
        use_container_width=True,
    )
    best_hard = min(with_hard, key=lambda s: s.hard.cost_per_ticket)
    if best_hard.model != rec.model:
        st.warning(
            f"**The ranking flips.** On realistic tickets the cheapest model is **{best_hard.label}** "
            f"({best_hard.hard.cost_per_ticket:.2f}/ticket), not the benchmark winner "
            f"({rec.hard.cost_per_ticket:.2f}/ticket). All-human triage costs {get_cost().human_cost:.2f}."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy by difficulty")
        tags = sorted(with_hard[0].hard.by_tag)
        st.dataframe(
            pd.DataFrame(
                {s.label: [s.hard.by_tag[t][1] for t in tags] for s in with_hard},
                index=[f"{t} (n={with_hard[0].hard.by_tag[t][0]})" for t in tags],
            ).style.format("{:.0%}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
        )
    with c2:
        st.subheader("Calibration: does confidence mean anything?")
        st.pyplot(reliability_figure({s.label: s.hard.calibration for s in with_hard}, "Hard set"))
        st.caption(
            "Points on the diagonal mean that '80% confident' really is right about 80% of the time. "
            "Points far below it are confidently wrong, and a confidence threshold can't catch those errors."
        )

st.subheader("Build vs buy trade-offs")
st.dataframe(
    pd.DataFrame(
        [
            {
                "approach": "TF-IDF + LogReg",
                "labelled data needed": "yes (thousands)",
                "time to first result": "seconds",
                "infra": "CPU, KB-sized model",
                "adapts to new intents": "retrain",
            },
            {
                "approach": "Zero-shot BART-MNLI",
                "labelled data needed": "no; only intent descriptions",
                "time to first result": "minutes (1.6 GB download)",
                "infra": "GPU preferred, slowest per ticket",
                "adapts to new intents": "edit a sentence",
            },
            {
                "approach": "Fine-tuned DistilBERT",
                "labelled data needed": "yes (thousands)",
                "time to first result": "~15 min training",
                "infra": "CPU fine, GPU faster",
                "adapts to new intents": "relabel and retrain",
            },
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

with st.expander("Where these numbers come from"):
    for s in results:
        st.write(f"- {s.label}: {prediction_source(s.model)}")
    st.caption(
        "Bitext is synthetic and templated, so every supervised model scores very high here. "
        "Treat the gaps between models as directional, and check them on real tickets before launch."
    )
