"""Intent classifiers. Both expose ``predict_proba(texts) -> (n, len(INTENTS))``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ticket_router.labels import INTENTS, ZERO_SHOT_DESCRIPTIONS, ZERO_SHOT_TEMPLATE

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
DISTILBERT_BASE = "distilbert-base-uncased"
DISTILBERT_DIR = Path("models/distilbert-ticket")
TFIDF_PATH = Path("models/tfidf.joblib")


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class IntentClassifier(Protocol):
    name: str

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


class ZeroShotClassifier:
    """bart-large-mnli NLI zero-shot over natural-language intent descriptions."""

    name = "zero_shot_bart"

    def __init__(self, model: str = ZERO_SHOT_MODEL, batch_size: int = 16):
        from transformers import pipeline

        self.pipe = pipeline("zero-shot-classification", model=model, device=pick_device())
        self.batch_size = batch_size
        self._desc_to_idx = {ZERO_SHOT_DESCRIPTIONS[i]: k for k, i in enumerate(INTENTS)}
        self._candidates = [ZERO_SHOT_DESCRIPTIONS[i] for i in INTENTS]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        outputs = self.pipe(
            texts,
            candidate_labels=self._candidates,
            hypothesis_template=ZERO_SHOT_TEMPLATE,
            multi_label=False,
            batch_size=self.batch_size,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        probs = np.zeros((len(texts), len(INTENTS)))
        for row, out in enumerate(outputs):
            for label, score in zip(out["labels"], out["scores"]):
                probs[row, self._desc_to_idx[label]] = score
        return probs


class DistilBertClassifier:
    """Fine-tuned DistilBERT sequence classifier (see ``ticket_router.train``)."""

    name = "distilbert_ft"

    def __init__(self, model_dir: Path = DISTILBERT_DIR, batch_size: int = 64, max_length: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} not found — run `uv run ticket-train` first.")
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if labels != INTENTS:
            raise ValueError(f"Model labels {labels} don't match INTENTS {INTENTS}; retrain.")
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits
                chunks.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, len(INTENTS)))


class TfidfClassifier:
    """TF-IDF + logistic regression: the trains-in-seconds baseline."""

    name = "tfidf"

    def __init__(self, path: Path = TFIDF_PATH):
        import joblib

        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found — run `uv run ticket-train --model tfidf` first.")
        self.pipeline = joblib.load(path)
        if list(self.pipeline.classes_) != list(range(len(INTENTS))):
            raise ValueError("TF-IDF model classes don't match INTENTS; retrain.")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)


MODEL_REGISTRY = {
    "tfidf": TfidfClassifier,
    "zero_shot": ZeroShotClassifier,
    "distilbert": DistilBertClassifier,
}


def load_model(name: str) -> IntentClassifier:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from None
