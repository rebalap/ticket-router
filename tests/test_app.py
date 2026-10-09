"""Smoke-test every Streamlit page renders without exceptions."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py", *sorted(p.relative_to(APP).as_posix() for p in (APP / "pages").glob("*.py"))]


@pytest.fixture(autouse=True)
def _app_on_path(monkeypatch):
    monkeypatch.chdir(APP.parent)
    monkeypatch.syspath_prepend(str(APP))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception, at.exception


def test_route_page_routes_a_sample_ticket():
    from ticket_router.models import TFIDF_PATH

    if not TFIDF_PATH.exists():
        pytest.skip("TF-IDF baseline not trained (make baseline)")
    at = AppTest.from_file(str(APP / "pages/1_Route_a_ticket.py"), default_timeout=60)
    at.run()
    at.selectbox[0].set_value("tfidf")
    at.text_area[0].set_value("I was charged twice for my invoice this month, I need this fixed ASAP")
    at.run()
    assert not at.exception, at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Intent"] == "billing"
    assert values["Urgency"] == "HIGH"


def test_cost_explorer_threshold_responds_to_human_cost():
    from ticket_router.results import available_models

    if "zero_shot" not in available_models():
        pytest.skip("No zero-shot predictions in outputs/ or demo_assets/")
    at = AppTest.from_file(str(APP / "pages/3_Cost_explorer.py"), default_timeout=60).run()
    at.radio[0].set_value("Zero-shot BART-large-MNLI").run()
    pricey = {m.label: m.value for m in at.metric}
    at.sidebar.slider[0].set_value(0.2).run()  # human triage gets cheap
    cheap = {m.label: m.value for m in at.metric}
    assert not at.exception, at.exception
    assert int(cheap["Escalated per 1,000 tickets"]) > int(pricey["Escalated per 1,000 tickets"])
    assert float(cheap["Cost per ticket"]) < float(pricey["Cost per ticket"])
