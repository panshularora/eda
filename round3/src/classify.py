"""Label every reaction: is it on topic, what kind of delay, what kind of reaction.

Three labelling passes, deliberately kept separate because they have very
different reliability and the report needs to say so:

1. **Relevance** - does this text describe a delay or service failure at all?
   Broad recall filter; precision is measured by hand audit, not assumed.
2. **Delay type and reaction type** - the two taxonomies from ``config``,
   applied as ordered first-match rules. These are transparent and auditable:
   anyone can read the pattern that fired and disagree with it. The alternative
   (an unsupervised clustering) would look more sophisticated and be far harder
   to defend to a judge asking "why is this row labelled that?".
3. **Sentiment and topic from the Round 2 model** - the rulebook requires the
   Round 2 NLP model to be applied here, so it is, unchanged and uncalibrated
   to this domain. That transfer is then *measured* rather than assumed, in
   ``validate.py``, against the star ratings the reviewers wrote themselves.

A rule that fires on a word is only as good as the word. Every assignment
therefore records which pattern matched, in ``delay_type_evidence`` and
``reaction_type_evidence``, so the labels are inspectable rather than magic.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (DELAY_RELEVANCE, DELAY_TYPES, PROCESSED, REACTION_TYPES,
                    ROUND2_MODEL, ROUND2_SRC)

_RELEVANCE = re.compile(DELAY_RELEVANCE, re.I)
_DELAY = [(name, re.compile(pat, re.I)) for name, pat in DELAY_TYPES]
_REACTION = [(name, re.compile(pat, re.I)) for name, pat in REACTION_TYPES]

# Time expressions are the strongest objective evidence of a *quantified* delay
_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|hours?|hrs?|days?|weeks?|months?)\b", re.I)
_UNIT_HOURS = {"min": 1 / 60, "minute": 1 / 60, "hour": 1.0, "hr": 1.0,
               "day": 24.0, "week": 168.0, "month": 720.0}


def is_relevant(text: str) -> bool:
    return bool(_RELEVANCE.search(text or ""))


def first_match(text: str, rules) -> tuple[str, str]:
    """Return (label, matched_span). Ordered rules, first one wins."""
    for name, rx in rules:
        m = rx.search(text or "")
        if m:
            return name, m.group(0)[:60]
    return "", ""


def extract_delay_hours(text: str) -> float | None:
    """Largest duration mentioned, in hours - a crude severity proxy."""
    best = None
    for value, unit in _DURATION.findall(text or ""):
        try:
            v = float(value)
        except ValueError:
            continue
        key = unit.lower().rstrip("s").replace("utes", "").replace("ute", "")
        key = {"hrs": "hr", "mins": "min"}.get(key, key)
        factor = _UNIT_HOURS.get(key)
        if factor is None:
            continue
        hours = v * factor
        if hours > 24 * 400:                 # "20 years of this" is rhetoric
            continue
        best = hours if best is None else max(best, hours)
    return best


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    text = df["full_text"].fillna("")
    df["is_delay_related"] = text.map(is_relevant)

    delay = text.map(lambda t: first_match(t, _DELAY))
    df["delay_type"] = [d[0] for d in delay]
    df["delay_type_evidence"] = [d[1] for d in delay]

    reaction = text.map(lambda t: first_match(t, _REACTION))
    df["reaction_type"] = [r[0] for r in reaction]
    df["reaction_type_evidence"] = [r[1] for r in reaction]

    df["stated_delay_hours"] = text.map(extract_delay_hours)

    # An unmatched but relevant row is "unspecified", not silently blank -
    # blanks and "we looked and found nothing" are different facts.
    df.loc[df["is_delay_related"] & (df["delay_type"] == ""), "delay_type"] = "unspecified_delay"
    df.loc[df["reaction_type"] == "", "reaction_type"] = "neutral_report"
    return df


# ---------------------------------------------------------------------------
def load_round2():
    """Load the Round 2 bundle, with its own modules importable."""
    # APPEND, never insert(0): the Round 2 package contains modules whose names
    # collide with ours (config.py, build_notebook.py). Putting it first on the
    # path silently shadows this round's modules - which is exactly what it did
    # the first time, and the failure surfaced three steps later in an unrelated
    # script. Appending keeps Round 2 importable for joblib without letting it
    # win a name contest.
    if str(ROUND2_SRC) not in sys.path:
        sys.path.append(str(ROUND2_SRC))
    if not Path(ROUND2_MODEL).exists():
        raise SystemExit(f"Round 2 model not found at {ROUND2_MODEL}")
    import joblib
    return joblib.load(ROUND2_MODEL)


def apply_round2(df: pd.DataFrame, batch: int = 20000) -> pd.DataFrame:
    """Score every reaction with the Round 2 sentiment and topic models."""
    bundle = load_round2()
    texts = df["full_text"].fillna("").tolist()

    for task in ("sentiment", "topic"):
        model = bundle["models"][task]
        classes = np.asarray(model.classes_)
        preds, confs = [], []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            proba = model.predict_proba(chunk)
            idx = proba.argmax(1)
            preds.extend(classes[idx])
            confs.extend(proba.max(1))
            print(f"      {task}: {min(i + batch, len(texts)):,}/{len(texts):,}",
                  end="\r", flush=True)
        df[f"r2_{task}"] = preds
        df[f"r2_{task}_confidence"] = np.round(confs, 4)
        print(f"      {task}: {len(texts):,} scored          ", flush=True)

    # A single signed number is what a time series needs.
    polarity = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    df["sentiment_score"] = df["r2_sentiment"].map(polarity).astype(float)
    # Confidence-weighted variant: a hesitant Negative should move the mean less
    # than a confident one.
    df["sentiment_score_weighted"] = (
        df["sentiment_score"] * df["r2_sentiment_confidence"]).round(4)
    return df


# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    src = PROCESSED / "reactions_normalised.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_normalised.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    print(f"  loaded {len(df):,} normalised reactions")

    df = apply_rules(df)
    rel = int(df["is_delay_related"].sum())
    print(f"  delay-related: {rel:,} ({rel / len(df):.1%})")
    print("  delay types:", df.loc[df.is_delay_related, "delay_type"]
          .value_counts().head(6).to_dict())
    print("  reactions  :", df["reaction_type"].value_counts().head(6).to_dict())

    print("  applying the Round 2 NLP model ...")
    df = apply_round2(df)
    print("  sentiment:", df["r2_sentiment"].value_counts().to_dict())

    out = PROCESSED / "reactions_labelled.parquet"
    try:
        df.to_parquet(out, index=False)
    except Exception:
        out = PROCESSED / "reactions_labelled.csv"
        df.to_csv(out, index=False, encoding="utf-8")
    print(f"  wrote {out.name}")
    return df


if __name__ == "__main__":
    main()
