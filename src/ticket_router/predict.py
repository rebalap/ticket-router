"""Run a model over a split and save per-ticket probabilities for evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ticket_router.data import load_split
from ticket_router.labels import INTENTS
from ticket_router.models import MODEL_REGISTRY, load_model
from ticket_router.results import OUTPUT_DIR, predictions_path, record_latency


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODEL_REGISTRY), required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test", "hard"])
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stratified sample per split (zero-shot is slow; e.g. --limit 1000)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    model = load_model(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        df = load_split(split)
        if args.limit and args.limit < len(df) and split != "hard":
            frac = args.limit / len(df)
            df = df.groupby("intent", group_keys=False).sample(frac=frac, random_state=args.seed)
        start = time.perf_counter()
        probs = model.predict_proba(df["text"].tolist())
        elapsed = time.perf_counter() - start

        out = df[[c for c in ("id", "text", "intent", "urgency", "tags") if c in df]].copy()
        out["pred"] = [INTENTS[i] for i in probs.argmax(axis=1)]
        out["confidence"] = probs.max(axis=1)
        for k, name in enumerate(INTENTS):
            out[f"p_{name}"] = probs[:, k]
        path = predictions_path(args.model, split, args.out)
        out.to_csv(path, index=False)
        acc = float(np.mean(out["pred"] == out["intent"]))
        ms_per_ticket = 1000 * elapsed / max(len(out), 1)
        if split == "test":
            record_latency(args.model, ms_per_ticket, args.out)
        print(f"[{args.model}/{split}] n={len(out)} acc={acc:.4f} {ms_per_ticket:.2f} ms/ticket -> {path}")


if __name__ == "__main__":
    main()
