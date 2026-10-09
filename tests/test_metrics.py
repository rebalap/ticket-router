import numpy as np
import pytest

from ticket_router import metrics
from ticket_router.labels import INTENT_TO_ID as ID


def test_cost_matrix_diagonal_zero_and_asymmetric():
    c = metrics.base_cost_matrix()
    assert np.all(np.diag(c) == 0)
    # Angry-cancellation-to-billing is worse than newsletter-to-billing.
    assert c[ID["cancellation"], ID["billing"]] > c[ID["general"], ID["billing"]]


def test_same_team_confusion_is_discounted():
    c = metrics.base_cost_matrix()
    assert c[ID["refund"], ID["billing"]] == pytest.approx(3.0 * metrics.SAME_TEAM_DISCOUNT)
    assert c[ID["refund"], ID["account"]] == pytest.approx(3.0)


def test_urgency_scales_cost():
    y = np.array([ID["cancellation"]] * 2)
    p = np.array([ID["billing"]] * 2)
    costs = metrics.per_ticket_cost(y, p, np.array(["normal", "high"]))
    assert costs[1] == pytest.approx(2 * costs[0])


def test_cost_weighted_confusion_sums_to_total_cost():
    y = np.array([ID["cancellation"], ID["general"], ID["billing"]])
    p = np.array([ID["billing"], ID["general"], ID["refund"]])
    u = np.array(["high", "low", "normal"])
    cwc = metrics.cost_weighted_confusion(y, p, u)
    assert cwc.sum() == pytest.approx(metrics.per_ticket_cost(y, p, u).sum())
    assert cwc[ID["cancellation"], ID["billing"]] == pytest.approx(10.0)


def test_threshold_sweep_tradeoff():
    # Two confident correct, one unconfident expensive mistake.
    y = np.array([ID["billing"], ID["refund"], ID["cancellation"]])
    p = np.array([ID["billing"], ID["refund"], ID["billing"]])
    conf = np.array([0.95, 0.9, 0.4])
    u = np.array(["normal", "normal", "high"])
    pts = metrics.sweep_thresholds(y, p, conf, u, thresholds=np.array([0.0, 0.5, 0.99]))
    assert [pt.coverage for pt in pts] == pytest.approx([1.0, 2 / 3, 0.0])
    assert pts[0].total_cost == pytest.approx(10.0)  # misroute the angry cancellation
    assert pts[1].total_cost == pytest.approx(1.0)  # escalate it instead
    assert pts[2].total_cost == pytest.approx(3.0)  # escalate everything
    assert metrics.best_threshold(pts).threshold == 0.5


def test_cost_config_override_changes_the_best_threshold():
    y = np.array([ID["billing"], ID["refund"], ID["cancellation"]])
    p = np.array([ID["billing"], ID["refund"], ID["billing"]])
    conf = np.array([0.95, 0.9, 0.4])
    u = np.array(["normal", "normal", "high"])
    grid = np.array([0.0, 0.5])
    # Humans are so expensive that eating the misroute is cheaper.
    pricey = metrics.CostConfig(human_cost=20.0)
    assert metrics.best_threshold(metrics.sweep_thresholds(y, p, conf, u, pricey, grid)).threshold == 0.0
    # Doubling cancellation severity doubles that cell.
    severe = metrics.CostConfig(severity={**metrics.DEFAULT_COSTS.severity, "cancellation": 10.0})
    assert metrics.per_ticket_cost(y, p, u, severe)[2] == pytest.approx(20.0)


def test_perfectly_calibrated_model_has_zero_ece():
    # 80%-confident predictions that are right 80% of the time.
    conf = np.full(10, 0.8)
    correct = np.array([True] * 8 + [False] * 2)
    assert metrics.expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_overconfident_model_has_large_ece_and_confident_errors():
    conf = np.full(10, 0.99)
    correct = np.array([True] * 5 + [False] * 5)
    assert metrics.expected_calibration_error(conf, correct) == pytest.approx(0.49)
    assert metrics.confident_error_share(conf, correct) == 1.0


def test_calibration_bins_cover_all_predictions():
    conf = np.array([0.05, 0.15, 0.95, 1.0, 1.0])
    correct = np.array([False, True, True, True, False])
    bins = metrics.calibration_bins(conf, correct)
    assert sum(b.count for b in bins) == 5
    assert bins[-1].lo == 0.9 and bins[-1].count == 3  # 1.0 falls in the top bin
