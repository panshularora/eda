"""Assemble the published dataset, its data dictionary and its quality report.

Three tables are published, because the topic is genuinely relational and
flattening it would destroy the part that matters:

``reactions``  one row per public reaction, with its delay type, reaction type,
               engagement, and the Round 2 sentiment and topic labels;
``incidents``  one row per documented delay event, with a start time - the
               evidence behind every trigger claim in the analysis;
``attention``  daily Wikipedia pageviews per brand - the independent signal
               used to corroborate spikes.

A dataset nobody can interpret is not a deliverable, so the dictionary is
generated from the frame itself (never typed by hand, so it cannot drift) and
records for every column its type, completeness, cardinality and provenance -
where the value came from and whether we derived it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from config import PROCESSED, RAW, REPORTS, SUBMISSION, TOPIC, WINDOW_DAYS
from fetch import read_jsonl

SUBMISSION_BUDGET_MB = 9.5          # the form caps uploads; stay inside it

# column -> (provenance, meaning)
DICTIONARY: dict[str, tuple[str, str]] = {
    "record_id":            ("derived", "SHA-256 of source + native id + text prefix; stable across runs"),
    "source":               ("collected", "platform the reaction came from"),
    "source_id":            ("collected", "app id, subreddit, hashtag or query that produced the row"),
    "brand":                ("collected/derived", "operator the reaction is about; from the app roster, or matched in text for social sources"),
    "delay_domain":         ("collected/derived", "sector of the delay: food_delivery, quick_commerce, parcel_courier, ecommerce, ride_hailing, airline, telecom_isp"),
    "store_country":        ("collected", "storefront the review was read from (Play only)"),
    "is_thin_text":         ("derived", "text shorter than 15 characters; still valid for volume and rating, too thin to mine"),
    "created_utc":          ("collected", "when the reaction was published, in UTC"),
    "date":                 ("derived", "UTC calendar date of created_utc"),
    "hour_utc":             ("derived", "UTC hour bucket of created_utc"),
    "collected_utc":        ("derived", "when our collector retrieved the row"),
    "title":                ("collected", "headline or post title; empty for Play reviews"),
    "text":                 ("collected", "body text as published, whitespace-normalised, URLs stripped"),
    "full_text":            ("derived", "title + text, the field all NLP runs on"),
    "text_length":          ("derived", "characters in full_text"),
    "word_count":           ("derived", "whitespace tokens in full_text"),
    "author_pseudonym":     ("derived", "salted SHA-256 of the handle; the raw handle is never stored"),
    "publisher":            ("collected", "outlet that published the article (news rows only); never a delay brand"),
    "rating":               ("collected", "1-5 star rating chosen by the reviewer (Play only) - an independent sentiment label"),
    "engagement":           ("collected", "endorsement count; meaning varies by source, see engagement_kind"),
    "engagement_kind":      ("derived", "what engagement counts on this source: thumbs_up, fav_boost_reply, points_comments or none"),
    "app_version":          ("collected", "app version the reviewer was running (Play only)"),
    "company_replied":      ("collected", "whether the operator publicly replied"),
    "company_reply_utc":    ("collected", "when the operator replied"),
    "url":                  ("collected", "public link to the reaction or its source"),
    "is_delay_related":     ("derived", "text matches the delay-relevance pattern"),
    "delay_type":           ("derived", "first matching delay pattern; see delay_type_evidence"),
    "delay_type_evidence":  ("derived", "the literal span that triggered delay_type - makes the label auditable"),
    "reaction_type":        ("derived", "first matching reaction pattern; see reaction_type_evidence"),
    "reaction_type_evidence": ("derived", "the literal span that triggered reaction_type"),
    "stated_delay_hours":   ("derived", "largest duration mentioned in the text, in hours; crude severity proxy"),
    "r2_sentiment":         ("model", "Round 2 sentiment model prediction: Negative / Neutral / Positive"),
    "r2_sentiment_confidence": ("model", "calibrated probability of the predicted sentiment class"),
    "r2_topic":             ("model", "Round 2 topic model prediction"),
    "r2_topic_confidence":  ("model", "calibrated probability of the predicted topic class"),
    "sentiment_score":      ("derived", "sentiment mapped to -1 / 0 / +1 for time-series arithmetic"),
    "sentiment_score_weighted": ("derived", "sentiment_score multiplied by model confidence"),
}

ORDER = list(DICTIONARY)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quality_report(df: pd.DataFrame) -> dict:
    rep: dict = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "window_start_utc": str(df["created_utc"].min()),
        "window_end_utc": str(df["created_utc"].max()),
        "span_days": round((pd.to_datetime(df["created_utc"]).max()
                            - pd.to_datetime(df["created_utc"]).min()).total_seconds() / 86400, 2),
        "by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "by_delay_domain": {k: int(v) for k, v in df["delay_domain"].value_counts().items()},
        "by_delay_type": {k: int(v) for k, v in df["delay_type"].value_counts().items()},
        "by_reaction_type": {k: int(v) for k, v in df["reaction_type"].value_counts().items()},
        "by_sentiment": {k: int(v) for k, v in df["r2_sentiment"].value_counts().items()},
        "delay_related_share": float(df["is_delay_related"].mean()),
        "thin_text_share": float(df["is_thin_text"].mean()),
        "duplicate_record_ids": int(df["record_id"].duplicated().sum()),
        "distinct_authors": int(df["author_pseudonym"].nunique()),
        "distinct_brands": int(df["brand"].nunique()),
    }
    rep["completeness"] = {
        c: round(float(df[c].notna().mean()), 4) for c in df.columns
    }
    return rep


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        prov, meaning = DICTIONARY.get(col, ("derived", ""))
        s = df[col]
        example = ""
        nn = s.dropna()
        if len(nn):
            example = str(nn.iloc[0])[:60]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "provenance": prov,
            "non_null": int(s.notna().sum()),
            "completeness": round(float(s.notna().mean()), 4),
            "distinct": int(s.nunique(dropna=True)),
            "example": example,
            "meaning": meaning,
        })
    return pd.DataFrame(rows)


def _write_sized_csv(df: pd.DataFrame, path, budget_mb: float) -> tuple[int, bool, str]:
    """Fit the upload budget by shedding *bytes*, never rows.

    The first version of this dropped rows until the file fit, and threw away
    89% of the dataset to do it. That is the wrong trade: a structured dataset's
    value is its coverage, and a reader can always re-read a truncated review
    but cannot recover a row that was never shipped.

    So the ladder below removes redundancy first - ``full_text`` is exactly
    ``title`` + ``text`` and is reconstructible in one line - then caps the long
    tail of review prose, and only drops rows if a 200-character cap somehow
    still overflows. Whatever step was needed is returned and recorded, so the
    published file is never quietly different from what the reader assumes.
    """
    def size_of(frame) -> float:
        frame.to_csv(path, index=False, encoding="utf-8")
        return path.stat().st_size / 1e6

    if size_of(df) <= budget_mb:
        return len(df), False, "complete"

    # 1. drop the redundant concatenation
    slim = df.drop(columns=["full_text"], errors="ignore")
    if size_of(slim) <= budget_mb:
        return len(slim), False, "dropped full_text (= title + text)"

    # 2. cap the prose, keeping every row
    for cap in (800, 500, 350, 200):
        capped = slim.copy()
        capped["text"] = capped["text"].astype(str).str.slice(0, cap)
        capped["text_truncated_at"] = cap
        if size_of(capped) <= budget_mb:
            return len(capped), False, f"dropped full_text; text capped at {cap} chars"

    # 3. last resort
    capped = capped.sort_values(["is_delay_related", "engagement", "created_utc"],
                                ascending=[False, False, False])
    while size_of(capped) > budget_mb and len(capped) > 5000:
        capped = capped.head(int(len(capped) * 0.85))
    return len(capped), True, "rows dropped after byte reduction was exhausted"


def main() -> dict:
    src = PROCESSED / "reactions_labelled.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_labelled.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    cols = [c for c in ORDER if c in df.columns] + \
           [c for c in df.columns if c not in ORDER]
    df = df[cols]
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    # ---- full dataset, in the repo ---------------------------------------
    full_csv = PROCESSED / "round3_delay_reactions_full.csv"
    df.to_csv(full_csv, index=False, encoding="utf-8")
    full_gz = PROCESSED / "round3_delay_reactions_full.csv.gz"
    with open(full_csv, "rb") as fin, gzip.open(full_gz, "wb", compresslevel=9) as fout:
        fout.writelines(fin)
    full_json = PROCESSED / "round3_delay_reactions_full.json.gz"
    with gzip.open(full_json, "wt", encoding="utf-8", compresslevel=9) as fh:
        df.to_json(fh, orient="records", date_format="iso", lines=True)
    print(f"  full CSV   {full_csv.stat().st_size/1e6:6.1f} MB")
    print(f"  full CSV.gz{full_gz.stat().st_size/1e6:6.1f} MB")

    # ---- companion tables -------------------------------------------------
    inc = pd.DataFrame(read_jsonl(RAW / "incidents.jsonl"))
    if len(inc):
        inc.to_csv(PROCESSED / "round3_incidents.csv", index=False, encoding="utf-8")
    att = pd.DataFrame(read_jsonl(RAW / "attention.jsonl"))
    if len(att):
        att.to_csv(PROCESSED / "round3_attention.csv", index=False, encoding="utf-8")
    print(f"  incidents {len(inc):,} rows | attention {len(att):,} rows")

    # ---- submission copy, inside the upload budget ------------------------
    sub_csv = SUBMISSION / "Round3_Delay_Reactions_Dataset_Team_SE7EN.csv"
    n_kept, rows_dropped, how = _write_sized_csv(df.copy(), sub_csv, SUBMISSION_BUDGET_MB)
    print(f"  submission CSV {sub_csv.stat().st_size/1e6:.1f} MB, {n_kept:,} rows "
          f"({n_kept/len(df):.0%} of full)  [{how}]")

    # ---- dictionary + quality --------------------------------------------
    dd = data_dictionary(df)
    dd.to_csv(PROCESSED / "DATA_DICTIONARY.csv", index=False, encoding="utf-8")
    qual = quality_report(df)
    qual["files"] = {
        "full_csv": {"rows": int(len(df)),
                     "bytes": full_csv.stat().st_size,
                     "sha256": sha256_file(full_csv)},
        "submission_csv": {"rows": int(n_kept),
                           "row_coverage_vs_full": round(n_kept / len(df), 4),
                           "bytes": sub_csv.stat().st_size,
                           "sha256": sha256_file(sub_csv),
                           "size_reduction": how,
                           "rows_dropped": rows_dropped},
    }
    qual["topic"] = TOPIC
    qual["window_days_requested"] = WINDOW_DAYS
    qual["built_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (REPORTS / "data_quality.json").write_text(
        json.dumps(qual, indent=2, default=str), encoding="utf-8")

    # markdown dictionary next to the data
    lines = [
        f"# Round 3 dataset - {TOPIC}", "",
        f"**{len(df):,} reactions** collected from public sources, "
        f"{qual['window_start_utc'][:10]} to {qual['window_end_utc'][:10]} "
        f"({qual['span_days']} days).", "",
        "| column | type | provenance | complete | distinct | meaning |",
        "|---|---|---|---|---|---|",
    ]
    for r in dd.itertuples():
        lines.append(f"| `{r.column}` | {r.dtype} | {r.provenance} | "
                     f"{r.completeness:.0%} | {r.distinct:,} | {r.meaning} |")
    lines += ["", "## Companion tables", "",
              "| file | rows | contents |", "|---|---|---|",
              f"| `round3_incidents.csv` | {len(inc):,} | documented delay events with start times |",
              f"| `round3_attention.csv` | {len(att):,} | daily Wikipedia pageviews per brand |", ""]
    (PROCESSED / "DATA_DICTIONARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"  delay-related {qual['delay_related_share']:.1%} | "
          f"{qual['distinct_brands']} brands | "
          f"{qual['distinct_authors']:,} distinct authors")
    return qual


if __name__ == "__main__":
    main()
