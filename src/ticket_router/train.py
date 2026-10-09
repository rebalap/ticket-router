"""Train an intent model: the TF-IDF baseline (seconds) or DistilBERT (minutes).

DistilBERT is fine-tuned on the train split, with early stopping on val.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ticket_router.data import load_split
from ticket_router.labels import INTENT_TO_ID, INTENTS
from ticket_router.models import DISTILBERT_BASE, DISTILBERT_DIR, TFIDF_PATH


def train_tfidf(out: Path = TFIDF_PATH, seed: int = 42) -> float:
    """Fit TF-IDF + logistic regression; returns val accuracy."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train, val = load_split("train"), load_split("val")
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10.0, max_iter=2000, random_state=seed),
    )
    pipeline.fit(train["text"], train["intent"].map(INTENT_TO_ID))
    val_acc = float((pipeline.predict(val["text"]) == val["intent"].map(INTENT_TO_ID)).mean())
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    print(f"TF-IDF baseline: val accuracy {val_acc:.4f} -> {out}")
    return val_acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["tfidf", "distilbert"], default="distilbert")
    ap.add_argument("--base", default=DISTILBERT_BASE)
    ap.add_argument("--out", type=Path, default=None, help="Defaults to models/<model>")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--train-limit", type=int, default=None, help="Subsample train for quick runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.model == "tfidf":
        train_tfidf(args.out or TFIDF_PATH, args.seed)
        return
    args.out = args.out or DISTILBERT_DIR

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    def to_dataset(name: str, limit: int | None = None) -> Dataset:
        df = load_split(name)
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=args.seed)
        ds = Dataset.from_dict({"text": df["text"].tolist(), "label": df["intent"].map(INTENT_TO_ID).tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length), batched=True)

    train_ds, val_ds = to_dataset("train", args.train_limit), to_dataset("val")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(INTENTS),
        id2label=dict(enumerate(INTENTS)),
        label2id=INTENT_TO_ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_steps=int(0.06 * math.ceil(len(train_ds) / args.batch_size) * args.epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    print("val:", trainer.evaluate())
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
