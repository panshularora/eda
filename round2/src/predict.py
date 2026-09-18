"""Inference entry point for the trained Social Engine bundle.

    python predict.py "the app keeps crashing after the update"
    python predict.py --file posts.txt
    python predict.py --csv new_posts.csv --column post_text --out scored.csv

The bundle stores two calibrated pipelines, so every prediction comes back
with a probability. For the topic task the recovered substring rule is
reported alongside the learned model - when they disagree, the rule is the
one that matches how the labels in Dataset 2 were produced, and the
disagreement is worth reading (see the technical report).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audit_labels import apply_rule
from config import MODEL_BUNDLE


def load_bundle(path=MODEL_BUNDLE):
    if not Path(path).exists():
        sys.exit(f"model bundle not found at {path} - run train.py first")
    return joblib.load(path)


def score(texts: list[str], bundle=None) -> pd.DataFrame:
    """Predict both targets for a list of posts."""
    bundle = bundle or load_bundle()
    out = pd.DataFrame({"post_text": texts})
    for task, model in bundle["models"].items():
        classes = np.asarray(model.classes_)
        proba = model.predict_proba(texts)
        out[f"{task}_pred"] = classes[proba.argmax(1)]
        out[f"{task}_confidence"] = proba.max(1).round(4)
    out["topic_rule"] = apply_rule(out["post_text"]).to_numpy()
    out["topic_model_agrees_with_rule"] = out["topic_pred"] == out["topic_rule"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score posts with the round-2 models.")
    ap.add_argument("text", nargs="*", help="one or more posts to score")
    ap.add_argument("--file", help="text file, one post per line")
    ap.add_argument("--csv", help="CSV of posts to score")
    ap.add_argument("--column", default="post_text", help="text column in --csv")
    ap.add_argument("--out", help="write results to this CSV instead of stdout")
    args = ap.parse_args(argv)

    if args.csv:
        texts = pd.read_csv(args.csv)[args.column].astype(str).tolist()
    elif args.file:
        texts = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        texts = args.text
    else:
        ap.error("give text, --file or --csv")

    result = score(texts)
    if args.out:
        result.to_csv(args.out, index=False, encoding="utf-8")
        print(f"wrote {len(result)} rows to {args.out}")
    else:
        for _, r in result.iterrows():
            flag = "" if r.topic_model_agrees_with_rule else f"  (rule says {r.topic_rule})"
            print(f"\n{r.post_text[:100]}")
            print(f"  sentiment : {r.sentiment_pred:<9s} p={r.sentiment_confidence:.3f}")
            print(f"  topic     : {r.topic_pred:<22s} p={r.topic_confidence:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
