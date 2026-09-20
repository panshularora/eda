"""Measure the Round 2 model on this new domain, against labels we did not write.

The rulebook requires the Round 2 NLP model to be applied to Round 3 data. It
does not require anyone to check whether that model still works once the domain
changes from 2015-era tweets to 2026 app-store reviews. It plainly might not:
the vocabulary, the length, the register and the purpose of the text are all
different.

We can check, because Google Play hands us the answer key. Every review carries
a 1-5 star rating chosen by the same person who wrote the text, at the same
moment, about the same experience. That is an independent sentiment label on
roughly 120,000 rows - an order of magnitude more than the Round 2 held-out
set, and from a completely different distribution.

So this module grades Round 2 on its transfer:

* stars are mapped to the Round 2 label space (1-2 Negative, 3 Neutral, 4-5
  Positive) and the disagreement is reported as a confusion matrix, macro-F1,
  Cohen's kappa and per-class recall;
* the same is computed *per delay domain*, because a model that reads food
  delivery well and airlines badly should not be trusted uniformly;
* the confidence the model attaches is checked against whether it was right,
  so the analysis downstream can weight or gate on it honestly.

The number that comes out of this is the licence to believe - or to discount -
every sentiment curve in the report.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)

from config import PROCESSED, REPORTS

LABELS = ["Negative", "Neutral", "Positive"]


def stars_to_label(rating: float) -> str | None:
    """1-2 stars Negative, 3 Neutral, 4-5 Positive.

    The 3-star boundary is the debatable one: a 3-star review is often a
    complaint with a redeeming feature. We keep it as Neutral because that is
    the plain reading of a mid-scale rating, and we report the 3-star row of
    the confusion matrix separately so the reader can judge the choice.
    """
    if pd.isna(rating):
        return None
    r = float(rating)
    if r <= 2:
        return "Negative"
    if r == 3:
        return "Neutral"
    return "Positive"


def evaluate(df: pd.DataFrame) -> dict:
    rated = df[df["rating"].notna() & df["r2_sentiment"].notna()].copy()
    rated["star_label"] = rated["rating"].map(stars_to_label)
    rated = rated[rated["star_label"].notna()]
    if rated.empty:
        return {"n": 0, "note": "no rated rows available"}

    y_true = rated["star_label"].to_numpy()
    y_pred = rated["r2_sentiment"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    out = {
        "n": int(len(rated)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0),
        "star_distribution": {str(int(k)): int(v) for k, v in
                              rated["rating"].value_counts().sort_index().items()},
        "predicted_distribution": {k: int(v) for k, v in
                                   pd.Series(y_pred).value_counts().items()},
    }

    # polarity-only view: collapse Neutral away, because the operational
    # question is "is this person unhappy", and Neutral is where a tweet-trained
    # model is least at home
    mask = (y_true != "Neutral") & (y_pred != "Neutral")
    if mask.sum():
        out["polarity_only"] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "cohen_kappa": float(cohen_kappa_score(y_true[mask], y_pred[mask])),
        }

    # per domain - trust is not uniform
    per_domain = {}
    for dom, g in rated.groupby("delay_domain"):
        if len(g) < 200 or not dom:
            continue
        per_domain[dom] = {
            "n": int(len(g)),
            "accuracy": float(accuracy_score(g["star_label"], g["r2_sentiment"])),
            "macro_f1": float(f1_score(g["star_label"], g["r2_sentiment"],
                                       average="macro", zero_division=0)),
            "negative_recall": float(
                ((g["r2_sentiment"] == "Negative") & (g["star_label"] == "Negative")).sum()
                / max((g["star_label"] == "Negative").sum(), 1)),
        }
    out["per_domain"] = dict(sorted(per_domain.items(),
                                    key=lambda kv: -kv[1]["macro_f1"]))

    # does the model's confidence mean anything here?
    correct = (y_true == y_pred).astype(float)
    conf = rated["r2_sentiment_confidence"].to_numpy()
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    out["confidence_bands"] = [
        {"band": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
         "accuracy": float(correct[m].mean())}
        for lo, hi in bins
        if (m := (conf >= lo) & (conf < hi)).sum() >= 100
    ]

    # how much does agreement improve if we only trust confident predictions?
    for thr in (0.6, 0.7, 0.8):
        m = conf >= thr
        if m.sum() >= 500:
            out[f"accuracy_at_conf_ge_{thr}"] = {
                "n": int(m.sum()), "coverage": float(m.mean()),
                "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
            }
    return out


def main() -> dict:
    src = PROCESSED / "reactions_labelled.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_labelled.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    res = evaluate(df)
    (REPORTS / "round2_transfer.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")

    if res.get("n"):
        print(f"  Round 2 model vs {res['n']:,} independent star labels")
        print(f"    accuracy {res['accuracy']:.4f} | macro-F1 {res['macro_f1']:.4f} "
              f"| kappa {res['cohen_kappa']:.4f}")
        if "polarity_only" in res:
            po = res["polarity_only"]
            print(f"    polarity only (Neutral dropped): accuracy {po['accuracy']:.4f} "
                  f"on {po['n']:,} rows, kappa {po['cohen_kappa']:.4f}")
        print("    best domains:", list(res["per_domain"])[:3])
        print("    worst domains:", list(res["per_domain"])[-3:])
    return res


if __name__ == "__main__":
    main()
