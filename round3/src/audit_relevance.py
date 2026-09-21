"""Reproduce the topic-relevance audit reported in reports/relevance_audit.md.

Two of the three checks in that document are mechanical and are re-run here:

1. **Validation against an independent label.** The star ratings were never
   used to build the filter, so they are an outside instrument. A correct topic
   filter selects rows whose ratings are far worse than the corpus baseline,
   and tightening the filter should widen that gap. This script prints the gap
   for the shipped filter.

2. **The sample.** Draws the same 120 rows (seed 42) that were hand-adjudicated,
   and writes them to `reports/relevance_audit_sample.csv` so a reader can
   re-read exactly what was read and disagree with any verdict.

The adjudication itself is a human judgement and cannot be recomputed; the
verdicts and the reasoning behind every false positive are written out in the
markdown report.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DELAY_RELEVANCE, PROCESSED, REPORTS       # noqa: E402

SAMPLE_N = 120
SEED = 42


def main() -> None:
    src = PROCESSED / "reactions_labelled.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_normalised.parquet"
    if not src.exists():
        raise SystemExit("no processed corpus found - run the pipeline first")
    df = pd.read_parquet(src)

    rx = re.compile(DELAY_RELEVANCE, re.I)
    hit = df["full_text"].fillna("").map(lambda t: bool(rx.search(t)))

    sel, base = df[hit], df[~hit]
    print(f"corpus                    {len(df):>8,}")
    print(f"selected as delay-related {int(hit.sum()):>8,}  ({hit.mean():.1%})")
    print()
    print("validation against the star ratings (an independent label):")
    print(f"  selected  mean rating {sel['rating'].mean():.3f}   "
          f"five-star share {(sel['rating'] == 5).mean():.3f}")
    print(f"  baseline  mean rating {base['rating'].mean():.3f}")
    print(f"  contrast  {base['rating'].mean() - sel['rating'].mean():.3f} stars")
    print()

    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(sel)), min(SAMPLE_N, len(sel))))
    sample = sel.iloc[idx][["record_id", "source", "brand", "rating",
                            "delay_type", "delay_type_evidence", "full_text"]]
    out = REPORTS / "relevance_audit_sample.csv"
    sample.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote the {len(sample)}-row adjudication sample -> {out}")
    print("verdicts and false-positive analysis: reports/relevance_audit.md")

    # The three false-positive families the hand audit found, as assertions, so
    # a future edit to the pattern cannot silently reintroduce them.
    regressions = {
        "late night food is not good": False,
        "incurred late-payment or overdraft fees": False,
        "can't wait to see how the results will be": False,
        "chocolate and translate": False,
        "my parcel is 3 days late": True,
        "still waiting for my refund": True,
        "the app is down for hours": True,
    }
    print("\nregression checks on the families the audit found:")
    bad = 0
    for probe, expected in regressions.items():
        got = bool(rx.search(probe))
        flag = "ok " if got == expected else "FAIL"
        bad += got != expected
        print(f"  {flag} {str(got):5s} (expected {expected!s:5s})  {probe!r}")
    if bad:
        raise SystemExit(f"{bad} relevance regression(s)")


if __name__ == "__main__":
    main()
