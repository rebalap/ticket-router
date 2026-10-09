"""Shared Streamlit helpers: cost settings in session state, cached models, summaries."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ticket_router import metrics
from ticket_router.models import DISTILBERT_DIR, TFIDF_PATH, load_model
from ticket_router.results import (
    ALL_MODELS,
    DEMO_ASSETS_DIR,
    MODEL_LABELS,
    OUTPUT_DIR,
    ModelSummary,
    find_predictions,
    summarize_all,
)
from ticket_router.router import TicketRouter

COST_KEY = "cost_config"
FLOOR_KEY = "min_auto_accuracy"

# Which local files make a model usable for live routing, and how to get them.
MODEL_REQUIREMENTS = {
    "tfidf": (TFIDF_PATH, "make baseline", "trains in ~5 s"),
    "zero_shot": (None, "nothing; downloads on first use", "1.6 GB download, ~65 ms/ticket on Apple MPS"),
    "distilbert": (DISTILBERT_DIR / "config.json", "make train", "~10–15 min on Apple MPS"),
}

SAMPLE_TICKETS = {
    "Angry cancellation": "This is the worst service ever. Cancel my damn subscription right now and delete my account.",
    "Double charge": "I was charged twice for my invoice this month, I need this fixed ASAP",
    "Refund status": "I returned the item two weeks ago and I'm still waiting for my refund",
    "Newsletter": "how do I sign up for your newsletter?",
    "Locked out": "I can't log in to my account, the password reset email never arrives",
    "Ambiguous": "I'm not sure who to talk to about my plan and the charge on it",
}


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"Ticket Router · {title}", page_icon="🎫", layout="wide")


def get_cost() -> metrics.CostConfig:
    if COST_KEY not in st.session_state:
        st.session_state[COST_KEY] = metrics.CostConfig()
    return st.session_state[COST_KEY]


def get_floor() -> float | None:
    return st.session_state.get(FLOOR_KEY)


def summaries() -> list[ModelSummary]:
    return summarize_all(ALL_MODELS, get_cost(), get_floor())


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def is_live(name: str) -> bool:
    path = MODEL_REQUIREMENTS[name][0]
    return path is None or Path(path).exists()


def prediction_source(name: str) -> str | None:
    path = find_predictions(name, "test")
    if path is None:
        return None
    return "your run (outputs/)" if path.parent == OUTPUT_DIR else "committed sample (demo_assets/)"


@st.cache_resource(show_spinner="Loading model…")
def cached_router(name: str) -> TicketRouter:
    # Threshold is applied per request, so build with 0 and override at call time.
    return TicketRouter(load_model(name), threshold=0.0)


def no_predictions_message() -> None:
    st.warning(
        "No saved predictions yet. Run `make baseline` (≈1 min) for live results, "
        f"or make sure `{DEMO_ASSETS_DIR}/` is present for the committed sample."
    )


def pct(x: float) -> str:
    return f"{x:.1%}"
