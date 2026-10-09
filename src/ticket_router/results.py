"""Load saved predictions and summarise them per model.

Shared by the eval CLI, the Streamlit app, and the decision brief. Predictions
are read from ``outputs/`` (your own runs) and fall back to ``demo_assets/``
(committed samples), so the app works on a fresh clone with nothing trained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ticket_router import metrics
from ticket_router.labels import INTENT_TO_ID, INTENTS

OUTPUT_DIR = Path("outputs")
DEMO_ASSETS_DIR = Path("demo_assets")
SEARCH_DIRS = (OUTPUT_DIR, DEMO_ASSETS_DIR)
ALL_MODELS = ["tfidf", "zero_shot", "distilbert"]
MODEL_LABELS = {
    "tfidf": "TF-IDF + LogReg (baseline)",
    "zero_shot": "Zero-shot BART-large-MNLI",
    "distilbert": "Fine-tuned DistilBERT",
}
LATENCY_FILE = "latency.json"


def predictions_path(model: str, split: str, directory: Path = OUTPUT_DIR) -> Path:
    return directory / f"preds_{model}_{split}.csv"


def find_predictions(model: str, split: str, dirs: tuple[Path, ...] = SEARCH_DIRS) -> Path | None:
    for d in dirs:
        path = predictions_path(model, split, d)
        if path.exists():
            return path
    return None


def available_models(models: list[str] = ALL_MODELS, dirs: tuple[Path, ...] = SEARCH_DIRS) -> list[str]:
    """Models with both val and test predictions somewhere on disk."""
    return [m for m in models if all(find_predictions(m, s, dirs) for s in ("val", "test"))]


def align_predictions(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every model's predictions to the ticket ids they all share."""
    ids = set.intersection(*(set(f["id"]) for f in frames.values()))
    return {m: f[f["id"].isin(ids)].sort_values("id").reset_index(drop=True) for m, f in frames.items()}


def load_aligned(models: list[str], split: str, dirs: tuple[Path, ...] = SEARCH_DIRS) -> dict[str, pd.DataFrame]:
    return align_predictions({m: pd.read_csv(find_predictions(m, split, dirs)) for m in models})


def prediction_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, confidence, urgency) as numpy arrays."""
    return (
        df["intent"].map(INTENT_TO_ID).to_numpy(),
        df["pred"].map(INTENT_TO_ID).to_numpy(),
        df["confidence"].to_numpy(),
        df["urgency"].to_numpy(),
    )


def load_latency(dirs: tuple[Path, ...] = SEARCH_DIRS) -> dict[str, float]:
    """ms/ticket per model; earlier dirs (your own runs) win."""
    merged: dict[str, float] = {}
    for d in reversed(dirs):
        path = d / LATENCY_FILE
        if path.exists():
            merged.update(json.loads(path.read_text()))
    return merged


def record_latency(model: str, ms_per_ticket: float, directory: Path = OUTPUT_DIR) -> None:
    path = directory / LATENCY_FILE
    data = json.loads(path.read_text()) if path.exists() else {}
    data[model] = round(ms_per_ticket, 2)
    path.write_text(json.dumps(data, indent=2) + "\n")


def export_demo_assets(models: list[str] = ALL_MODELS, out: Path = DEMO_ASSETS_DIR) -> None:
    """Copy your predictions, restricted to the ids every model shares, into ``demo_assets/``.

    Commit the result so a fresh clone shows real numbers for all models without training.
    """
    models = available_models(models, (OUTPUT_DIR,))
    out.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        for model, df in load_aligned(models, split, (OUTPUT_DIR,)).items():
            df.to_csv(predictions_path(model, split, out), index=False)
    for model in models:
        if (src := find_predictions(model, "hard", (OUTPUT_DIR,))) is not None:
            pd.read_csv(src, keep_default_na=False).to_csv(predictions_path(model, "hard", out), index=False)
    latency = {m: v for m, v in load_latency((OUTPUT_DIR,)).items() if m in models}
    (out / LATENCY_FILE).write_text(json.dumps(latency, indent=2) + "\n")
    print(f"Exported {models} to {out}/")


@dataclass
class Confusion:
    true: str
    pred: str
    count: int
    cost: float


# The hard set is small (n=80), so use coarser calibration bins there.
HARD_SET_CALIBRATION_BINS = 5


@dataclass
class HardSetEval:
    """One model on the hand-written hard set, at the threshold chosen on val."""

    n: int
    accuracy: float
    accuracy_high_urgency: float
    coverage: float
    auto_accuracy: float
    cost_per_ticket: float
    ece: float
    confident_error_share: float
    calibration: list[metrics.CalibrationBin]
    by_tag: dict[str, tuple[int, float]]  # tag -> (n tickets, accuracy)
    misses: pd.DataFrame  # wrong predictions, most expensive first


@dataclass
class ModelSummary:
    model: str
    n_test: int
    accuracy: float
    accuracy_high_urgency: float
    misroute_cost_per_ticket: float  # if everything were auto-routed
    threshold: float  # chosen on val
    coverage: float  # on test, at threshold
    auto_accuracy: float
    total_cost_per_ticket: float  # misroutes + human triage, at threshold
    latency_ms: float | None
    val_sweep: list[metrics.ThresholdPoint]
    cost_confusion: np.ndarray  # test, summed cost per (true, pred)
    count_confusion: np.ndarray  # test, ticket counts
    ece: float  # test calibration error
    confident_error_share: float  # test: share of mistakes made at >= 90% confidence
    calibration: list[metrics.CalibrationBin]  # test
    hard: HardSetEval | None = None

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)

    def worst_confusions(self, k: int = 5) -> list[Confusion]:
        cells = [
            Confusion(INTENTS[i], INTENTS[j], int(self.count_confusion[i, j]), float(self.cost_confusion[i, j]))
            for i in range(len(INTENTS))
            for j in range(len(INTENTS))
            if i != j and self.cost_confusion[i, j] > 0
        ]
        return sorted(cells, key=lambda c: c.cost, reverse=True)[:k]


def evaluate_hard_set(
    hard: pd.DataFrame, threshold: float, cost: metrics.CostConfig = metrics.DEFAULT_COSTS
) -> HardSetEval:
    y, p, c, u = prediction_arrays(hard)
    correct = y == p
    at = metrics.evaluate_at_threshold(y, p, c, u, threshold, cost)
    high = u == "high"
    tags = hard.get("tags", pd.Series([""] * len(hard))).fillna("").str.split(";")
    by_tag = {}
    for tag in sorted({t for ts in tags for t in ts if t}):
        mask = tags.map(lambda ts, t=tag: t in ts).to_numpy()
        by_tag[tag] = (int(mask.sum()), float(correct[mask].mean()))
    misses = hard.assign(misroute_cost=metrics.per_ticket_cost(y, p, u, cost))[~correct]
    return HardSetEval(
        n=len(y),
        accuracy=metrics.accuracy(y, p),
        accuracy_high_urgency=metrics.accuracy(y[high], p[high]),
        coverage=at.coverage,
        auto_accuracy=at.auto_accuracy,
        cost_per_ticket=at.cost_per_ticket,
        ece=metrics.expected_calibration_error(c, correct, HARD_SET_CALIBRATION_BINS),
        confident_error_share=metrics.confident_error_share(c, correct),
        calibration=metrics.calibration_bins(c, correct, HARD_SET_CALIBRATION_BINS),
        by_tag=by_tag,
        misses=misses.sort_values(["misroute_cost", "confidence"], ascending=False).reset_index(drop=True),
    )


def summarize(
    model: str,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cost: metrics.CostConfig = metrics.DEFAULT_COSTS,
    min_auto_accuracy: float | None = None,
    latency_ms: float | None = None,
    hard: pd.DataFrame | None = None,
) -> ModelSummary:
    """Pick the escalation threshold on val, then report on test (and the hard set, if given)."""
    yv, pv, cv, uv = prediction_arrays(val)
    yt, pt, ct, ut = prediction_arrays(test)
    val_sweep = metrics.sweep_thresholds(yv, pv, cv, uv, cost)
    chosen = metrics.best_threshold(val_sweep, min_auto_accuracy)
    no_escalation = metrics.evaluate_at_threshold(yt, pt, ct, ut, 0.0, cost)
    at_chosen = metrics.evaluate_at_threshold(yt, pt, ct, ut, chosen.threshold, cost)
    high = ut == "high"
    correct = yt == pt
    return ModelSummary(
        model=model,
        n_test=len(yt),
        accuracy=metrics.accuracy(yt, pt),
        accuracy_high_urgency=metrics.accuracy(yt[high], pt[high]),
        misroute_cost_per_ticket=no_escalation.cost_per_ticket,
        threshold=chosen.threshold,
        coverage=at_chosen.coverage,
        auto_accuracy=at_chosen.auto_accuracy,
        total_cost_per_ticket=at_chosen.cost_per_ticket,
        latency_ms=latency_ms,
        val_sweep=val_sweep,
        cost_confusion=metrics.cost_weighted_confusion(yt, pt, ut, cost),
        count_confusion=metrics.confusion(yt, pt),
        ece=metrics.expected_calibration_error(ct, correct),
        confident_error_share=metrics.confident_error_share(ct, correct),
        calibration=metrics.calibration_bins(ct, correct),
        hard=evaluate_hard_set(hard, chosen.threshold, cost) if hard is not None else None,
    )


def summarize_all(
    models: list[str] | None = None,
    cost: metrics.CostConfig = metrics.DEFAULT_COSTS,
    min_auto_accuracy: float | None = None,
    dirs: tuple[Path, ...] = SEARCH_DIRS,
) -> list[ModelSummary]:
    models = available_models(models or ALL_MODELS, dirs)
    if not models:
        return []
    val, test = load_aligned(models, "val", dirs), load_aligned(models, "test", dirs)
    latency = load_latency(dirs)
    hard = {m: pd.read_csv(p, keep_default_na=False) for m in models if (p := find_predictions(m, "hard", dirs))}
    return [
        summarize(m, val[m], test[m], cost, min_auto_accuracy, latency.get(m), hard.get(m)) for m in models
    ]


def summary_table(summaries: list[ModelSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": s.label,
                "test tickets": s.n_test,
                "accuracy": s.accuracy,
                "accuracy (high urgency)": s.accuracy_high_urgency,
                "misroute cost / ticket (no escalation)": s.misroute_cost_per_ticket,
                "threshold (chosen on val)": s.threshold,
                "auto-routed share": s.coverage,
                "auto-routed accuracy": s.auto_accuracy,
                "total cost / ticket (with escalation)": s.total_cost_per_ticket,
                "latency ms / ticket": s.latency_ms,
                "calibration error (ECE, test)": s.ece,
                "hard set accuracy": s.hard.accuracy if s.hard else None,
                "hard set accuracy (high urgency)": s.hard.accuracy_high_urgency if s.hard else None,
                "hard set cost / ticket (with escalation)": s.hard.cost_per_ticket if s.hard else None,
                "calibration error (ECE, hard set)": s.hard.ece if s.hard else None,
                "hard set mistakes made at ≥90% confidence": s.hard.confident_error_share if s.hard else None,
            }
            for s in summaries
        ]
    ).set_index("model")
