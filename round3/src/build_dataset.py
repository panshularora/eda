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
    "r2_p_negative":        ("model", "Round 2 model P(Negative); kept because the max alone cannot support recalibration"),
    "r2_p_neutral":         ("model", "Round 2 model P(Neutral)"),
    "r2_p_positive":        ("model", "Round 2 model P(Positive)"),
    "r2_topic":             ("model", "Round 2 topic model prediction. NOT INTERPRETED: Round 2 established this label is a substring switch, and reports/analysis.json measures how closely the model still reproduces it on this corpus. Shipped because the rulebook requires the Round 2 model to be applied."),
    "r2_topic_confidence":  ("model", "calibrated probability of the predicted topic class"),
    "flag_churn_threat":    ("derived", "mentions uninstalling, switching or cancelling a subscription; independent of reaction_type"),
    "flag_refund_demand":   ("derived", "asks for money back or compensation; independent of reaction_type"),
    "flag_escalation":      ("derived", "threatens a complaint, consumer forum or legal action"),
    "flag_anger":           ("derived", "uses abusive or outraged vocabulary"),
    "flag_recovery":        ("derived", "mentions the issue being resolved, refunded or apologised for"),
    "flag_repeat_incident": ("derived", "says this has happened before"),
    "flag_money_lost":      ("derived", "says money was charged, deducted or lost"),
    "flag_staff_blamed":    ("derived", "names a driver, rider, courier or agent"),
    "competitor_named":     ("derived", "the operator the customer says they are switching to, if any"),
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


# Columns that are exactly reconstructible from others. Dropping them from the
# submission copy costs a reader one line of pandas and buys back a third of the
# file, so they go first whenever the upload budget binds.
DERIVABLE = ["full_text", "date", "hour_utc", "text_length", "word_count",
             "sentiment_score_weighted", "collected_utc"]


def _write_submission(df: pd.DataFrame, path, budget_mb: float) -> dict:
    """Build the submitted dataset under the upload cap, deliberately.

    76,000 rows carrying real review prose is ~50 MB of CSV. No amount of
    column pruning fits that into 9.5 MB while leaving the text readable, so
    something has to give and the choice should be made on what the dataset is
    *for* rather than by a loop that truncates until the number goes green.

    The assignment is "Reaction to a Major Delivery or Service Delay". The
    delay-related rows are that dataset; the remaining rows are a comparison
    baseline, valuable for measuring how delay reactions differ from ordinary
    ones but not themselves the subject. So the submitted file is **every
    delay-related row, with its text intact**, and the full corpus - baseline
    included - ships in the repository as .csv.gz and .json.gz.

    The ladder below still applies inside that choice, and whatever step was
    needed is recorded so the published file is never quietly different from
    what a reader assumes.
    """
    def write(frame) -> float:
        frame.to_csv(path, index=False, encoding="utf-8")
        return path.stat().st_size / 1e6

    steps: list[str] = []

    # 1. the on-topic dataset, complete
    topic = df[df["is_delay_related"]].copy()
    steps.append(f"delay-related rows only ({len(topic):,} of {len(df):,})")
    size = write(topic)
    if size <= budget_mb:
        return {"rows": len(topic), "size_mb": round(size, 2),
                "row_coverage_of_topic_subset": 1.0, "steps": steps,
                "text_intact": True}

    # 2. shed reconstructible columns
    slim = topic.drop(columns=[c for c in DERIVABLE if c in topic.columns])
    steps.append("dropped reconstructible columns: " + ", ".join(DERIVABLE))
    size = write(slim)
    if size <= budget_mb:
        return {"rows": len(slim), "size_mb": round(size, 2),
                "row_coverage_of_topic_subset": 1.0, "steps": steps,
                "text_intact": True}

    # 3. shed the permalink (provenance survives via source + source_id)
    if "url" in slim.columns:
        slim = slim.drop(columns=["url"])
        steps.append("dropped url (provenance preserved by source + source_id)")
        size = write(slim)
        if size <= budget_mb:
            return {"rows": len(slim), "size_mb": round(size, 2),
                    "row_coverage_of_topic_subset": 1.0, "steps": steps,
                    "text_intact": True}

    # 4. only now touch the text
    for cap in (600, 400, 280):
        capped = slim.copy()
        capped["text"] = capped["text"].astype(str).str.slice(0, cap)
        size = write(capped)
        if size <= budget_mb:
            steps.append(f"text capped at {cap} characters")
            return {"rows": len(capped), "size_mb": round(size, 2),
                    "row_coverage_of_topic_subset": 1.0, "steps": steps,
                    "text_intact": False, "text_cap": cap}

    # 5. last resort, and it should never be reached
    capped = capped.sort_values(["engagement", "created_utc"], ascending=False)
    while write(capped) > budget_mb and len(capped) > 5000:
        capped = capped.head(int(len(capped) * 0.85))
    steps.append("rows dropped after byte reduction was exhausted")
    return {"rows": len(capped), "size_mb": round(path.stat().st_size / 1e6, 2),
            "row_coverage_of_topic_subset": round(len(capped) / max(len(topic), 1), 4),
            "steps": steps, "text_intact": False}


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
    # Written in chunks: pandas' lines=True builds the entire JSON document as
    # one Python string before splitting it, which exhausts memory at this row
    # count. Streaming 5,000 rows at a time produces the identical file.
    full_json = PROCESSED / "round3_delay_reactions_full.json.gz"
    with gzip.open(full_json, "wt", encoding="utf-8", compresslevel=9) as fh:
        for start in range(0, len(df), 5000):
            chunk = df.iloc[start:start + 5000]
            if chunk.empty:
                continue
            block = chunk.to_json(orient="records", date_format="iso", lines=True)
            fh.write(block if block.endswith("\n") else block + "\n")
    print(f"  full CSV   {full_csv.stat().st_size/1e6:6.1f} MB")
    print(f"  full CSV.gz{full_gz.stat().st_size/1e6:6.1f} MB")

    # ---- companion tables -------------------------------------------------
    inc = pd.DataFrame(read_jsonl(RAW / "incidents.jsonl"))
    if len(inc):
        inc.to_csv(PROCESSED / "round3_incidents.csv", index=False, encoding="utf-8")
    att = pd.DataFrame(read_jsonl(RAW / "attention.jsonl"))
    if len(att):
        att.to_csv(PROCESSED / "round3_attention.csv", index=False, encoding="utf-8")
    # The census: one row per brand-day with the number of reviews that
    # actually existed, whether or not we kept any of them. It is what turns
    # the quota sample into a population estimate, so it ships beside the
    # dataset rather than living only in the repository.
    cen = pd.DataFrame(read_jsonl(RAW / "play_census.jsonl"))
    if len(cen):
        cen.to_csv(PROCESSED / "round3_play_census.csv", index=False, encoding="utf-8")
    print(f"  incidents {len(inc):,} rows | attention {len(att):,} rows "
          f"| census {len(cen):,} brand-days")

    # ---- submission copy, inside the upload budget ------------------------
    sub_csv = SUBMISSION / "Round3_Delay_Reactions_Dataset_Team_SE7EN.csv"
    sub = _write_submission(df.copy(), sub_csv, SUBMISSION_BUDGET_MB)
    print(f"  submission CSV {sub['size_mb']:.1f} MB, {sub['rows']:,} rows "
          f"= {sub['row_coverage_of_topic_subset']:.0%} of all delay-related rows, "
          f"text {'intact' if sub['text_intact'] else 'capped'}")
    for st in sub["steps"]:
        print(f"      - {st}")

    # ---- dictionary + quality --------------------------------------------
    dd = data_dictionary(df)
    dd.to_csv(PROCESSED / "DATA_DICTIONARY.csv", index=False, encoding="utf-8")
    qual = quality_report(df)
    qual["files"] = {
        "full_csv": {"rows": int(len(df)),
                     "bytes": full_csv.stat().st_size,
                     "sha256": sha256_file(full_csv)},
        "submission_csv": dict(sub, bytes=sub_csv.stat().st_size,
                               sha256=sha256_file(sub_csv),
                               scope="all delay-related reactions; the full corpus "
                                     "including the non-delay comparison baseline is "
                                     "published as round3_delay_reactions_full.csv.gz"),
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
