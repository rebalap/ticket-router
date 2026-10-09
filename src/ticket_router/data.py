"""Download Bitext, map to our intent taxonomy, and write stratified splits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ticket_router.labels import BITEXT_TO_INTENT
from ticket_router.urgency import score_urgency

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATA_DIR = Path("data")
SPLITS = ("train", "val", "test")
# Hand-written, committed robustness set (see evals/README.md).
HARD_SET_PATH = Path("evals/hard_tickets.csv")

_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def clean_text(text: str) -> str:
    """Turn Bitext template slots like ``{{Order Number}}`` into plain words."""
    text = _PLACEHOLDER.sub(lambda m: m.group(1).lower(), text)
    return re.sub(r"\s+", " ", text).strip()


def load_bitext() -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset(DATASET_ID, split="train").to_pandas()
    df = df.rename(columns={"instruction": "text", "intent": "bitext_intent"})
    df["text"] = df["text"].map(clean_text)
    df["intent"] = df["bitext_intent"].map(BITEXT_TO_INTENT)
    missing = df["intent"].isna()
    if missing.any():
        raise ValueError(f"Unmapped Bitext intents: {sorted(df.loc[missing, 'bitext_intent'].unique())}")
    df["urgency"] = [score_urgency(t, i) for t, i in zip(df["text"], df["intent"])]
    df = df.drop_duplicates("text").reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df[["id", "text", "intent", "urgency", "bitext_intent", "category", "flags"]]


def split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> dict[str, pd.DataFrame]:
    train, rest = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df["intent"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=test_frac / (val_frac + test_frac), stratify=rest["intent"], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def load_split(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load ``train``/``val``/``test`` from ``data_dir``, or ``hard`` from ``evals/``."""
    if name == "hard":
        return pd.read_csv(HARD_SET_PATH, keep_default_na=False)
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `uv run ticket-prepare` first.")
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_bitext()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in split(df, args.val_frac, args.test_frac, args.seed).items():
        part.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name:>5}: {len(part):>6} rows -> {args.out / f'{name}.csv'}")
    print("\nIntent distribution:\n" + df["intent"].value_counts().to_string())
    print("\nUrgency distribution:\n" + df["urgency"].value_counts().to_string())


if __name__ == "__main__":
    main()
