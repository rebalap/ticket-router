import numpy as np
import pandas as pd

from ticket_router.brief import recommend, render_brief
from ticket_router.labels import INTENTS
from ticket_router.results import align_predictions, summarize


def _preds(ids, true, pred, conf, urgency="normal"):
    return pd.DataFrame(
        {"id": ids, "text": "t", "intent": true, "pred": pred, "confidence": conf, "urgency": urgency}
    )


def test_align_predictions_uses_common_ids():
    a = _preds([1, 2, 3], ["billing"] * 3, ["billing"] * 3, [0.9] * 3)
    b = _preds([2, 3, 4], ["billing"] * 3, ["billing"] * 3, [0.9] * 3)
    aligned = align_predictions({"a": a, "b": b})
    assert list(aligned["a"]["id"]) == list(aligned["b"]["id"]) == [2, 3]


def test_brief_recommends_lowest_cost_model():
    rng = np.random.default_rng(0)
    true = list(rng.choice(INTENTS, 200))
    good = _preds(range(200), true, true, [0.99] * 200)
    bad_pred = ["billing" if t == "cancellation" else t for t in true]
    bad = _preds(range(200), true, bad_pred, [0.99] * 200, urgency="high")
    s_good, s_bad = summarize("distilbert", good, good), summarize("zero_shot", bad, bad)
    assert recommend([s_bad, s_good]).model == "distilbert"
    md = render_brief([s_bad, s_good])
    assert "Fine-tuned DistilBERT" in md and "## Top risks" in md
    assert s_bad.worst_confusions(1)[0].true == "cancellation"


def test_hard_set_loads_and_is_balanced():
    from ticket_router.data import load_split

    hard = load_split("hard")
    assert len(hard) == 80
    assert set(hard["intent"]) == set(INTENTS)
    assert set(hard["urgency"]) <= {"low", "normal", "high"}
    assert hard["id"].min() >= 900000  # never collides with Bitext ids


def test_brief_flags_benchmark_winner_that_fails_hard_set():
    rng = np.random.default_rng(1)
    true = list(rng.choice(INTENTS, 100))
    perfect = _preds(range(100), true, true, [0.99] * 100)
    # On the hard set, the benchmark winner is confidently wrong; the other model is right.
    wrong = [INTENTS[(INTENTS.index(t) + 1) % len(INTENTS)] for t in true]
    hard_bad = _preds(range(100), true, wrong, [0.99] * 100, urgency="high")
    hard_ok = _preds(range(100), true, true, [0.6] * 100)
    winner = summarize("distilbert", perfect, perfect, hard=hard_bad)
    other = summarize("zero_shot", perfect.assign(confidence=0.9), perfect.assign(confidence=0.9), hard=hard_ok)
    assert winner.hard.confident_error_share == 1.0
    md = render_brief([winner, other], chosen=winner)
    assert "Robustness check fails" in md and "Zero-shot BART-large-MNLI" in md
