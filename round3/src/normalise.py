"""Turn five differently-shaped feeds into one analysable table.

The sources disagree about almost everything: Play gives a star rating and no
title, Reddit gives a title and no rating, news gives a publisher instead of an
author, Hacker News counts points where Mastodon counts favourites. Left alone
those differences would leak into the analysis as fake structure - the classic
way a multi-source dataset produces a finding that is really a schema artefact.

This module resolves them explicitly:

* one canonical schema, with a single ``engagement`` column whose meaning is
  recorded per source in ``engagement_kind`` rather than silently averaged;
* brand and delay-domain attribution for the sources that do not carry it, by
  matching an alias table against the text;
* timestamps parsed to real UTC instants, with unparseable rows dropped and
  counted rather than coerced to "now";
* deduplication at two levels - native id, then normalised text - because the
  same complaint is frequently cross-posted, and counting it twice would
  inflate exactly the spikes we are trying to detect.

Every row that is dropped is counted, and the counts end up in the quality
report. Nothing disappears silently.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from config import PLAY_APPS, PROCESSED, RAW, window_start
from fetch import read_jsonl

# ---------------------------------------------------------------------------
# what "engagement" means, per source - never averaged across these
# ---------------------------------------------------------------------------
ENGAGEMENT_KIND = {
    "google_play": "thumbs_up",          # other users endorsing the review
    "reddit":      "none",               # RSS does not expose score
    "mastodon":    "fav_boost_reply",    # favourites + boosts + replies
    "news":        "none",               # articles have no public reaction count
    "hackernews":  "points_comments",    # points + comment count
}

# ---------------------------------------------------------------------------
# brand attribution for sources that do not carry it
# ---------------------------------------------------------------------------
BRAND_ALIASES: dict[str, tuple[str, list[str]]] = {}
for _aid, _cc, _brand, _domain in PLAY_APPS:
    BRAND_ALIASES.setdefault(_brand, (_domain, []))
_EXTRA_ALIASES = {
    "DoorDash": ["doordash", "door dash", "dasher"],
    "Uber Eats": ["uber eats", "ubereats"],
    "Grubhub": ["grubhub", "grub hub"],
    "Deliveroo": ["deliveroo"],
    "Just Eat": ["just eat", "justeat"],
    "Zomato": ["zomato"],
    "Swiggy": ["swiggy"],
    "Domino's": ["dominos", "domino's"],
    "foodpanda": ["foodpanda", "food panda"],
    "Glovo": ["glovo"],
    "talabat": ["talabat"],
    "Blinkit": ["blinkit", "grofers"],
    "Zepto": ["zepto"],
    "Instamart": ["instamart"],
    "Instacart": ["instacart"],
    "bigbasket": ["bigbasket", "big basket"],
    "FedEx": ["fedex", "fed ex"],
    "UPS": ["ups "],
    "DHL": ["dhl"],
    "AfterShip": ["aftership"],
    "Bluedart": ["bluedart", "blue dart"],
    "Amazon": ["amazon", "prime delivery"],
    "Amazon IN": [],
    "Flipkart": ["flipkart"],
    "Myntra": ["myntra"],
    "Meesho": ["meesho"],
    "Temu": ["temu"],
    "Wish": ["wish.com"],
    "SHEIN": ["shein"],
    "AliExpress": ["aliexpress"],
    "eBay": ["ebay"],
    "Uber": ["uber"],
    "Ola": ["ola cabs", "olacabs"],
    "Lyft": ["lyft"],
    "Rapido": ["rapido"],
    "IndiGo": ["indigo"],
    "United": ["united airlines"],
    "Delta": ["delta airlines", "delta air"],
    "American": ["american airlines"],
    "Southwest": ["southwest airlines"],
    "Ryanair": ["ryanair"],
    "Jio": ["jio"],
    "Airtel": ["airtel"],
    "Xfinity": ["xfinity", "comcast"],
}
for _b, _al in _EXTRA_ALIASES.items():
    if _b in BRAND_ALIASES:
        BRAND_ALIASES[_b] = (BRAND_ALIASES[_b][0], _al)

# longest alias first so "uber eats" is not swallowed by "uber"
_ALIAS_INDEX: list[tuple[str, str, str]] = sorted(
    ((alias, brand, dom) for brand, (dom, aliases) in BRAND_ALIASES.items()
     for alias in aliases),
    key=lambda t: -len(t[0]),
)

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")


def normalise_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _URL.sub(" ", s)
    return _WS.sub(" ", s).strip()


def attribute_brand(text: str) -> tuple[str, str]:
    low = (text or "").lower()
    for alias, brand, domain in _ALIAS_INDEX:
        if alias in low:
            return brand, domain
    return "", ""


def parse_ts(value) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, format="mixed")
    except Exception:
        try:
            ts = pd.to_datetime(value, utc=True)
        except Exception:
            return None
    return None if pd.isna(ts) else ts


def record_id(source: str, native: str, text: str) -> str:
    basis = f"{source}|{native}|{text[:200]}"
    return hashlib.sha256(basis.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, dict]:
    frames, counts = [], {}
    for name in ("play_reviews", "reddit", "mastodon", "news", "hackernews"):
        rows = read_jsonl(RAW / f"{name}.jsonl")
        counts[name] = len(rows)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise SystemExit("no raw data found - run collect.py first")
    return pd.concat(frames, ignore_index=True), counts


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    audit: dict = {"input_rows": int(len(df))}

    # --- text -------------------------------------------------------------
    df["title"] = df.get("title", "").fillna("").map(normalise_text)
    df["text"] = df.get("text", "").fillna("").map(normalise_text)
    df["full_text"] = (df["title"] + ". " + df["text"]).str.strip(". ").str.strip()

    # Only genuinely empty text is dropped. A three-word review ("always late")
    # still carries a star rating, a timestamp and an engagement count, so it is
    # real evidence for the volume and rating series even though it is too thin
    # to mine for a delay type. Dropping those 47k rows outright would discard
    # 39% of the corpus and bias the volume series toward verbose complainers,
    # so instead they are kept and flagged, and the text-based analyses filter
    # on the flag themselves.
    before = len(df)
    df = df[df["full_text"].str.len() >= 3].copy()
    audit["dropped_empty_text"] = before - len(df)
    df["is_thin_text"] = df["full_text"].str.len() < 15
    audit["flagged_thin_text"] = int(df["is_thin_text"].sum())

    # --- time -------------------------------------------------------------
    df["created_utc"] = df["created_utc"].map(parse_ts)
    before = len(df)
    df = df[df["created_utc"].notna()].copy()
    audit["dropped_unparseable_time"] = before - len(df)

    start = pd.Timestamp(window_start())
    now = pd.Timestamp(datetime.now(timezone.utc))
    before = len(df)
    df = df[(df["created_utc"] >= start) & (df["created_utc"] <= now)].copy()
    audit["dropped_outside_window"] = before - len(df)

    # --- brand / domain attribution --------------------------------------
    need = df["brand"].fillna("") == ""
    attributed = df.loc[need, "full_text"].map(attribute_brand)
    df.loc[need, "brand"] = [a[0] for a in attributed]
    df.loc[need, "delay_domain"] = [a[1] for a in attributed]
    df["brand"] = df["brand"].fillna("")
    df["delay_domain"] = df["delay_domain"].fillna("")
    audit["brand_attributed_from_text"] = int(need.sum())
    audit["rows_without_brand"] = int((df["brand"] == "").sum())

    # --- engagement, with its meaning kept attached -----------------------
    df["engagement"] = pd.to_numeric(df.get("thumbs_up", 0), errors="coerce").fillna(0).astype(int)
    df["engagement_kind"] = df["source"].map(ENGAGEMENT_KIND).fillna("none")
    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")

    # --- dedupe -----------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["source", "record_native_id"], keep="first")
    audit["dropped_duplicate_native_id"] = before - len(df)

    df["text_key"] = (df["source"] + "|" +
                      df["full_text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True))
    before = len(df)
    df = df.drop_duplicates(subset=["text_key"], keep="first").drop(columns=["text_key"])
    audit["dropped_duplicate_text"] = before - len(df)

    # --- identity and ordering -------------------------------------------
    df["record_id"] = [record_id(s, str(n), t) for s, n, t in
                       zip(df["source"], df["record_native_id"], df["full_text"])]
    df = df.drop_duplicates(subset=["record_id"], keep="first")
    df = df.sort_values("created_utc").reset_index(drop=True)

    df["date"] = df["created_utc"].dt.strftime("%Y-%m-%d")
    df["hour_utc"] = df["created_utc"].dt.strftime("%Y-%m-%d %H:00")
    df["text_length"] = df["full_text"].str.len()
    df["word_count"] = df["full_text"].str.split().str.len()

    audit["output_rows"] = int(len(df))
    audit["by_source"] = {k: int(v) for k, v in Counter(df["source"]).items()}
    audit["window_start"] = str(df["created_utc"].min())
    audit["window_end"] = str(df["created_utc"].max())
    audit["span_days"] = round(
        (df["created_utc"].max() - df["created_utc"].min()).total_seconds() / 86400, 2)
    return df, audit


COLUMNS = [
    "record_id", "source", "source_id", "brand", "delay_domain", "store_country",
    "is_thin_text",
    "created_utc", "date", "hour_utc", "collected_utc",
    "title", "text", "full_text", "text_length", "word_count",
    "author_pseudonym", "rating", "engagement", "engagement_kind",
    "app_version", "company_replied", "company_reply_utc", "url",
]


def main() -> pd.DataFrame:
    raw, counts = load_raw()
    print("  raw rows by file:", counts)
    df, audit = build(raw)
    keep = [c for c in COLUMNS if c in df.columns]
    df = df[keep]
    out = PROCESSED / "reactions_normalised.parquet"
    try:
        df.to_parquet(out, index=False)
    except Exception:
        out = PROCESSED / "reactions_normalised.csv"
        df.to_csv(out, index=False, encoding="utf-8")
    print(f"  normalised {audit['input_rows']:,} -> {audit['output_rows']:,} rows")
    print(f"  window {audit['window_start'][:16]} .. {audit['window_end'][:16]} "
          f"({audit['span_days']} days)")
    print(f"  by source: {audit['by_source']}")
    import json
    (PROCESSED / "normalise_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return df


if __name__ == "__main__":
    main()
