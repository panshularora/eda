"""
Data Vortex · Round 1 · Phase 1 — Social Engine intake-pipeline restoration
Team SE7EN (Tanmay Singh, Panshul Arora)

Turns the two files recovered from Archive Node 07 into analysis- and SQL-ready tables.

    python src/clean.py

Inputs   data/raw/Social_Engine_Posts_Corrupted.csv   12,360 rows x 8 cols
         data/raw/Social_Engine_Users.csv                1,500 rows x 5 cols
Outputs  data/cleaned/posts_cleaned.csv                 one row per unique post
         data/cleaned/users_cleaned.csv                 one row per user
         reports/cleaning_audit.json                    every count quoted in the report
         reports/change_log.csv                         one line per changed value (lineage)

Design rules
  1. Read every cell as raw text (keep_default_na=False). Nothing is converted implicitly:
     each corruption is detected by an explicit rule, counted, fixed and logged.
  2. Never invent a value. A value is corrected only when the original can be recovered
     deterministically (sign flip, injected suffix, format change, exact duplicate).
     Unrecoverable values become NULL; they are never imputed into the cleaned table.
  3. Every rule is backed by evidence computed here and stored in the audit file,
     so the report and README quote numbers the code produced, not hand-typed ones.
  4. Deterministic: same inputs -> byte-identical outputs (checked by validate.py).
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "cleaned"
REPORTS = ROOT / "reports"

RAW_POSTS = RAW / "Social_Engine_Posts_Corrupted.csv"
RAW_USERS = RAW / "Social_Engine_Users.csv"

# SHA-256 of the files served by https://datavortex-social-engine.vercel.app (Archive Node 07).
EXPECTED_SHA256 = {
    RAW_POSTS.name: "49a460d6b6d8b7f5bbbc1605852e555611adffdc9d7248c2d33910e2dd5dce42",
    RAW_USERS.name: "d30efe470a4dd5612266c3b3ebcc5e84528ef12dc5d832698a6e2a64283f7fd8",
}

# Tokens that mean "no value". The raw file uses exactly two: '' and the string 'NULL'.
# The others are listed so the pipeline stays safe if a new extract uses them; the audit
# records how many of each were actually seen.
MISSING_TOKENS = ["", "NULL", "null", "None", "NaN", "nan", "N/A", "NA"]

PLATFORMS = ["Facebook", "Instagram", "Reddit", "Twitter", "YouTube"]

# Observation window of the dataset (every well-formed timestamp falls inside it).
WINDOW_START, WINDOW_END = pd.Timestamp("2024-05-01"), pd.Timestamp("2025-04-30 23:59:59")

# Artifacts appended to the END of text_content by the corruption process.
# Each was checked to occur only as a terminal suffix and never inside genuine text
# (see audit -> text.artifact_evidence), so removing it restores the original post.
TEXT_SUFFIX_ARTIFACTS = {
    "html_entity_amp": r"&amp;",   # HTML-escaped '&'; decoding would leave a stray '&'
    "html_tag_br": r"<br>",
    "html_tag_div": r"<div>",
    "mojibake_e_acute": r"Ã©",     # UTF-8 'é' mis-decoded as Latin-1
    "trailing_newlines": r"\n+",
}

LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "hi": "Hindi", "ja": "Japanese", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
}

POST_COLUMNS = [
    "post_id", "user_id", "platform", "text_content",
    "post_date", "post_datetime_utc", "timestamp_precision", "timestamp_source_format",
    "likes", "shares", "comments",
    "dq_duplicate_copies_removed", "dq_likes_sign_corrected", "dq_text_artifact_removed",
]
USER_COLUMNS = [
    "user_id", "location", "city", "country", "language", "language_name",
    "account_created", "follower_count",
]


# ───────────────────────────── helpers ─────────────────────────────

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChangeLog:
    """Collects one record per modified value so every transformation is traceable."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def add(self, post_id: pd.Series, column: str, rule: str, old: pd.Series, new: pd.Series) -> None:
        if len(post_id) == 0:
            return
        self.frames.append(pd.DataFrame({
            "post_id": post_id.values, "column": column, "rule": rule,
            "old_value": old.astype("string").values, "new_value": new.astype("string").values,
        }))

    def to_frame(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True)


def load_raw(path: Path) -> pd.DataFrame:
    """Read every cell as text; no implicit NA conversion, no type guessing."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def missing_mask(s: pd.Series) -> pd.Series:
    return s.isin(MISSING_TOKENS)


def token_counts(s: pd.Series) -> dict:
    vc = s[missing_mask(s)].value_counts()
    return {("<empty>" if k == "" else k): int(v) for k, v in vc.items()}


# ───────────────────────────── posts ─────────────────────────────

def profile_raw_file(posts: pd.DataFrame) -> dict:
    """Corruption counts on the raw file BEFORE de-duplication.

    Rules below report counts on the de-duplicated table (what actually gets fixed). Both are
    kept so every figure can be reconciled: raw count = fixed count + copies inside duplicates.
    """
    suffix_pattern = "(?:" + "|".join(TEXT_SUFFIX_ARTIFACTS.values()) + r")\Z"
    text_suffix = posts.text_content.str.contains(suffix_pattern, regex=True)
    text_without_suffix = posts.text_content.str.replace(suffix_pattern, "", regex=True)
    likes_num = pd.to_numeric(posts.likes.where(~missing_mask(posts.likes)), errors="coerce")
    return {
        "rows": int(len(posts)),
        "platform_missing": int(missing_mask(posts.platform).sum()),
        "likes_missing": int(missing_mask(posts.likes).sum()),
        "likes_negative": int((likes_num < 0).sum()),
        "text_missing_token": int(missing_mask(posts.text_content).sum()),
        "text_missing_including_hidden": int(missing_mask(text_without_suffix).sum()),
        "text_suffix_artifact": int(text_suffix.sum()),
        "timestamp_unix_epoch": int(posts.timestamp.str.fullmatch(r"\d{10}").sum()),
        "timestamp_dd_mm_yyyy": int(posts.timestamp.str.fullmatch(r"\d{2}-\d{2}-\d{4}").sum()),
        "timestamp_iso8601": int(posts.timestamp.str.contains("T").sum()),
    }


def deduplicate(posts: pd.DataFrame, log: ChangeLog, audit: dict) -> pd.DataFrame:
    """Remove repeated ingestions of the same post_id.

    Evidence first: a duplicate is only dropped if it is an exact copy of the kept row in
    every column. If copies disagreed we would have to merge them, so that case fails loudly.
    """
    posts = posts.copy()
    posts["_raw_row"] = np.arange(len(posts)) + 2  # +2 = 1-based line number incl. header
    dup_mask = posts.duplicated("post_id", keep=False)
    groups = posts[dup_mask].groupby("post_id")
    conflicting = [pid for pid, g in groups if len(g.drop(columns="_raw_row").drop_duplicates()) > 1]
    if conflicting:
        raise ValueError(f"{len(conflicting)} duplicate post_ids have conflicting values: {conflicting[:5]}")

    copies = posts.groupby("post_id")["post_id"].transform("size") - 1
    removed = posts[posts.duplicated("post_id", keep="first")]
    log.add(removed.post_id, "*row*", "drop_exact_duplicate",
            "raw line " + removed._raw_row.astype(str), pd.Series(["removed"] * len(removed)))

    posts["dq_duplicate_copies_removed"] = copies.astype(int)
    kept = posts.drop_duplicates("post_id", keep="first").drop(columns="_raw_row").reset_index(drop=True)

    sizes = groups.size().value_counts().sort_index()
    audit["duplicates"] = {
        "raw_rows": int(len(posts)),
        "duplicated_post_ids": int(groups.ngroups),
        "group_size_distribution": {f"{k}_copies": int(v) for k, v in sizes.items()},
        "rows_removed": int(len(removed)),
        "rows_after_dedup": int(len(kept)),
        "all_copies_identical_in_every_column": True,
        "exact_full_row_duplicates_in_raw": int(posts.drop(columns=["_raw_row", "dq_duplicate_copies_removed"]).duplicated().sum()),
    }
    return kept


def normalise_timestamps(posts: pd.DataFrame, log: ChangeLog, audit: dict) -> pd.DataFrame:
    """Parse three coexisting timestamp encodings with explicit formats.

    iso8601     2025-04-13T20:12:18   naive, assumed UTC (same clock as epoch values)
    unix_epoch  1722528840            seconds since 1970-01-01 UTC
    dd-mm-yyyy  25-09-2024            DATE ONLY: the time of day was lost in corruption

    Date-only rows are NOT given a fake 00:00:00 time. They keep post_date and get a NULL
    post_datetime_utc with timestamp_precision = 'day'. Padding them with midnight would put
    ~29% of all posts at 00:00 and fabricate a "midnight posting spike".
    """
    ts = posts["timestamp"].str.strip()
    is_iso = ts.str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    is_unix = ts.str.fullmatch(r"\d{10}")
    is_dmy = ts.str.fullmatch(r"\d{2}-\d{2}-\d{4}")
    unknown = ~(is_iso | is_unix | is_dmy)
    if unknown.any():
        raise ValueError(f"Unrecognised timestamp formats: {ts[unknown].unique()[:10]}")

    # Evidence that dd-mm-yyyy really is day-first: the 2nd field never exceeds 12,
    # while the 1st field does for a large share of rows.
    first, second = ts[is_dmy].str[:2].astype(int), ts[is_dmy].str[3:5].astype(int)
    if (second > 12).any():
        raise ValueError("dd-mm-yyyy assumption violated: month field > 12")

    iso_dt = pd.to_datetime(ts.where(is_iso), format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    unix_dt = pd.to_datetime(pd.to_numeric(ts.where(is_unix), errors="coerce"), unit="s")  # UTC by definition
    dmy_date = pd.to_datetime(ts.where(is_dmy), format="%d-%m-%Y", errors="coerce")

    dt = iso_dt.fillna(unix_dt)                      # NaT only for date-only rows
    post_date = dt.dt.normalize().fillna(dmy_date)   # always populated
    if dt[~is_dmy].isna().any() or post_date.isna().any():
        raise ValueError("Timestamp parsing produced unexpected NaT values")
    out_of_window = ~post_date.between(WINDOW_START.normalize(), WINDOW_END)
    if out_of_window.any():
        raise ValueError(f"{int(out_of_window.sum())} timestamps outside the observation window")

    posts = posts.copy()
    posts["post_date"] = post_date.dt.strftime("%Y-%m-%d")
    posts["post_datetime_utc"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").where(~is_dmy, None)
    posts["timestamp_precision"] = np.where(is_dmy, "day", "second")
    posts["timestamp_source_format"] = np.select([is_iso, is_unix], ["iso8601", "unix_epoch"], "dd-mm-yyyy")

    for mask, rule, new in [(is_unix, "unix_epoch_to_utc_datetime", posts["post_datetime_utc"]),
                            (is_dmy, "dd-mm-yyyy_to_date_time_unknown", posts["post_date"])]:
        log.add(posts.post_id[mask], "timestamp", rule, ts[mask], new[mask])

    def span(values: pd.Series) -> list[str]:
        return [str(values.min()), str(values.max())]

    audit["timestamps"] = {
        "format_counts": {"iso8601": int(is_iso.sum()), "unix_epoch": int(is_unix.sum()), "dd-mm-yyyy": int(is_dmy.sum())},
        "range_by_format_utc": {"iso8601": span(iso_dt[is_iso]), "unix_epoch": span(unix_dt[is_unix]),
                                "dd-mm-yyyy": [str(d.date()) for d in (dmy_date.min(), dmy_date.max())]},
        "dd_mm_evidence": {"max_first_field": int(first.max()), "max_second_field": int(second.max()),
                           "rows_with_first_field_gt_12": int((first > 12).sum())},
        "date_only_rows_time_unknown": int(is_dmy.sum()),
        "unparseable": 0,
        "outside_window": 0,
    }
    return posts.drop(columns="timestamp")


def clean_platform(posts: pd.DataFrame, log: ChangeLog, audit: dict) -> pd.DataFrame:
    """Canonicalise platform names; unrecoverable values -> NULL (no mode imputation)."""
    raw = posts["platform"]
    tokens = token_counts(raw)
    canon = {p.lower(): p for p in PLATFORMS}
    stripped = raw.str.strip()
    mapped = stripped.str.lower().map(canon)
    miss = missing_mask(stripped)
    invalid = mapped.isna() & ~miss
    if invalid.any():
        raise ValueError(f"Unknown platform labels: {stripped[invalid].unique()}")
    recased = ~miss & (mapped != raw)
    log.add(posts.post_id[miss], "platform", "missing_token_to_null", raw[miss], pd.Series([None] * int(miss.sum())))
    log.add(posts.post_id[recased], "platform", "canonical_case", raw[recased], mapped[recased])
    posts = posts.copy()
    posts["platform"] = mapped.where(~miss, None)
    audit["platform"] = {"missing_tokens": tokens, "missing_total": int(miss.sum()),
                         "recased_or_trimmed": int(recased.sum()), "decision": "NULL (not imputed)"}
    return posts


def clean_engagement(posts: pd.DataFrame, log: ChangeLog, audit: dict) -> pd.DataFrame:
    """likes: undo sign flips, keep unrecoverable values NULL. shares/comments: validate."""
    posts = posts.copy()
    raw = posts["likes"].str.strip()
    miss = missing_mask(raw)
    is_int = raw.str.fullmatch(r"-?\d+")
    is_float = raw.str.fullmatch(r"-?\d+\.\d+")
    bad = ~(miss | is_int | is_float)
    if bad.any():
        raise ValueError(f"Non-numeric likes: {raw[bad].unique()[:10]}")

    value = pd.to_numeric(raw.where(~miss), errors="raise")
    negative = value < 0
    # Evidence for a sign flip (not downvotes): every negative is float-formatted ("-2388.0")
    # while every non-negative value is a plain integer, i.e. the negatives were produced by a
    # separate numeric operation; and |negative| follows the same distribution as positives.
    if (value[is_float] % 1 != 0).any():
        raise ValueError("Fractional likes found; sign-flip rule assumes integral values")

    fixed = value.abs()
    log.add(posts.post_id[miss], "likes", "missing_token_to_null", raw[miss], pd.Series([None] * int(miss.sum())))
    log.add(posts.post_id[negative], "likes", "sign_flip_abs", raw[negative], fixed[negative].astype("Int64"))
    posts["likes"] = fixed.round().astype("Int64")
    posts["dq_likes_sign_corrected"] = negative.astype(int)

    audit["likes"] = {
        "missing_tokens": token_counts(raw), "missing_total": int(miss.sum()),
        "negative_values": int(negative.sum()),
        "negative_values_float_formatted": int((negative & is_float).sum()),
        "non_negative_values_float_formatted": int((~negative & is_float & ~miss).sum()),
        "range_after_fix": [int(fixed.min()), int(fixed.max())],
        "decision_missing": "NULL (not imputed)",
    }

    for col, upper in [("shares", 2000), ("comments", 1000)]:
        s = posts[col].str.strip()
        if missing_mask(s).any() or not s.str.fullmatch(r"\d+").all():
            raise ValueError(f"{col} has missing or non-integer values")
        posts[col] = s.astype(int)
        audit[col] = {"missing_total": 0, "negative_values": 0,
                      "range": [int(posts[col].min()), int(posts[col].max())], "decision": "validated, unchanged"}
    return posts


def clean_text(posts: pd.DataFrame, log: ChangeLog, audit: dict) -> pd.DataFrame:
    """Strip injected suffix artifacts, normalise whitespace, map missing tokens to NULL."""
    posts = posts.copy()
    raw = posts["text_content"]
    suffix_re = re.compile("(?:" + "|".join(f"(?:{p})" for p in TEXT_SUFFIX_ARTIFACTS.values()) + r")\Z")  # \Z: true end of string

    # Evidence: artifacts appear only as suffixes, never mid-text.
    evidence = {}
    for name, pat in TEXT_SUFFIX_ARTIFACTS.items():
        anywhere = int(raw.str.contains(pat, regex=True).sum())
        at_end = int(raw.str.contains(rf"(?:{pat})\Z", regex=True).sum())
        evidence[name] = {"rows_anywhere": anywhere, "rows_as_suffix": at_end}
        if anywhere != at_end:
            raise ValueError(f"Artifact {name} occurs mid-text; suffix rule is not safe")

    def artifact_kind(s: str) -> str:
        for name, pat in TEXT_SUFFIX_ARTIFACTS.items():
            if re.search(rf"(?:{pat})\Z", s):
                return name
        return "none"

    kind = raw.map(artifact_kind)
    no_suffix = raw.str.replace(suffix_re, "", regex=True)
    if no_suffix.str.contains(suffix_re, regex=True).any():
        raise ValueError("Stacked suffix artifacts found; extend the rule")

    # Generic safety nets (no-ops on this extract, audited to prove it).
    residual = {
        "html_entities": int(no_suffix.str.contains(r"&[#A-Za-z0-9]+;").sum()),
        "html_tags": int(no_suffix.str.contains(r"</?[A-Za-z][^>]*>").sum()),
        "mojibake_sequences": int(no_suffix.str.contains(r"Ã.|â€").sum()),
    }
    text = no_suffix.map(html.unescape).str.replace(r"</?[A-Za-z][^>]*>", " ", regex=True)

    normalised = text.str.replace(r"\s+", " ", regex=True).str.strip()
    ws_changed = (normalised != text) & ~missing_mask(normalised)
    miss = missing_mask(normalised)
    has_artifact = kind != "none"

    log.add(posts.post_id[has_artifact], "text_content", "strip_suffix_artifact", raw[has_artifact], no_suffix[has_artifact])
    log.add(posts.post_id[ws_changed], "text_content", "collapse_whitespace", text[ws_changed], normalised[ws_changed])
    log.add(posts.post_id[miss], "text_content", "missing_token_to_null", no_suffix[miss], pd.Series([None] * int(miss.sum())))

    posts["text_content"] = normalised.where(~miss, None)
    posts["dq_text_artifact_removed"] = kind

    audit["text_content"] = {
        "missing_tokens_raw": token_counts(raw),
        "missing_token_hidden_behind_artifact": int((missing_mask(no_suffix) & has_artifact).sum()),
        "missing_total_after_cleaning": int(miss.sum()),
        "artifact_counts": {k: int(v) for k, v in kind.value_counts().drop("none", errors="ignore").sort_index().items()},
        "rows_with_artifact": int(has_artifact.sum()),
        "artifact_evidence": evidence,
        "residual_after_suffix_strip": residual,
        "whitespace_collapsed_rows": int(ws_changed.sum()),
        "decision_missing": "NULL (no placeholder text)",
    }
    return posts


# ───────────────────────────── users ─────────────────────────────

def clean_users(users: pd.DataFrame, audit: dict) -> pd.DataFrame:
    """Users arrive clean; validate every column, then add standardised helper columns."""
    u = users.copy()
    report = {}
    for c in u.columns:
        report[c] = {"missing": int(missing_mask(u[c]).sum()), "padded": int((u[c] != u[c].str.strip()).sum())}
    if any(v["missing"] or v["padded"] for v in report.values()):
        raise ValueError(f"Users table has missing/padded values: {report}")
    if u.user_id.duplicated().any():
        raise ValueError("Duplicate user_id in users table")
    if not u.user_id.str.fullmatch(r"user_[a-z0-9]{8}").all():
        raise ValueError("Malformed user_id")

    created = pd.to_datetime(u.account_created, format="%Y-%m-%d", errors="coerce")
    if created.isna().any():
        raise ValueError("Invalid account_created dates")
    if not u.follower_count.str.fullmatch(r"\d+").all():
        raise ValueError("follower_count must be a non-negative integer")
    unknown_lang = set(u.language) - set(LANGUAGE_NAMES)
    if unknown_lang:
        raise ValueError(f"Unknown language codes: {unknown_lang}")

    parts = u.location.str.rsplit(", ", n=1)
    u["city"] = parts.str[0]
    u["country"] = parts.map(lambda p: p[1] if len(p) == 2 else p[0])  # "Singapore" is a city-state
    u["language_name"] = u.language.map(LANGUAGE_NAMES)
    u["account_created"] = created.dt.strftime("%Y-%m-%d")
    u["follower_count"] = u.follower_count.astype(int)

    audit["users"] = {
        "rows": int(len(u)), "column_checks": report, "duplicate_user_ids": 0,
        "account_created_range": [str(created.min().date()), str(created.max().date())],
        "follower_count_range": [int(u.follower_count.min()), int(u.follower_count.max())],
        "distinct_locations": int(u.location.nunique()), "distinct_countries": int(u.country.nunique()),
        "languages": sorted(u.language.unique().tolist()),
        "changes": "none required; added city, country, language_name",
    }
    return u[USER_COLUMNS]


# ───────────────────────────── pipeline ─────────────────────────────

def cross_table_checks(posts: pd.DataFrame, users: pd.DataFrame, audit: dict) -> None:
    orphans = set(posts.user_id) - set(users.user_id)
    silent = set(users.user_id) - set(posts.user_id)
    merged = posts[["post_date", "user_id"]].merge(users[["user_id", "account_created"]], on="user_id")
    before_signup = int((merged.post_date < merged.account_created).sum())
    if orphans or before_signup:
        raise ValueError(f"Integrity failure: {len(orphans)} orphan users, {before_signup} posts before signup")
    audit["referential_integrity"] = {
        "posts_user_ids_missing_from_users": len(orphans),
        "users_without_posts": len(silent),
        "posts_dated_before_account_creation": before_signup,
    }


def run(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    audit: dict = {"inputs": {}}
    for p in (RAW_POSTS, RAW_USERS):
        digest = sha256(p)
        audit["inputs"][p.name] = {"sha256": digest, "matches_recovered_archive": digest == EXPECTED_SHA256[p.name]}

    log = ChangeLog()
    raw_posts, raw_users = load_raw(RAW_POSTS), load_raw(RAW_USERS)
    audit["inputs"][RAW_POSTS.name]["shape"] = list(raw_posts.shape)
    audit["inputs"][RAW_USERS.name]["shape"] = list(raw_users.shape)
    audit["raw_file_counts_before_dedup"] = profile_raw_file(raw_posts)

    posts = deduplicate(raw_posts, log, audit)
    posts = normalise_timestamps(posts, log, audit)
    posts = clean_platform(posts, log, audit)
    posts = clean_engagement(posts, log, audit)
    posts = clean_text(posts, log, audit)
    posts = posts[POST_COLUMNS]
    users = clean_users(raw_users, audit)
    cross_table_checks(posts, users, audit)

    changes = log.to_frame()
    audit["output"] = {
        "posts_rows": int(len(posts)), "users_rows": int(len(users)),
        "posts_null_counts": {c: int(v) for c, v in posts.isna().sum().items() if v},
        "change_log_rows": int(len(changes)),
        "change_log_by_rule": {f"{c}:{r}": int(n) for (c, r), n in changes.groupby(["column", "rule"]).size().items()},
    }

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        csv_opts = dict(index=False, encoding="utf-8", lineterminator="\n")
        posts.to_csv(OUT / "posts_cleaned.csv", **csv_opts)
        users.to_csv(OUT / "users_cleaned.csv", **csv_opts)
        changes.to_csv(REPORTS / "change_log.csv", **csv_opts)
        audit["output"]["sha256"] = {f: sha256(OUT / f) for f in ("posts_cleaned.csv", "users_cleaned.csv")}
        (REPORTS / "cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return posts, users, audit, changes


if __name__ == "__main__":
    posts, users, audit, changes = run()
    d, t, l, x = audit["duplicates"], audit["timestamps"], audit["likes"], audit["text_content"]
    print(f"raw posts {d['raw_rows']:,} -> {len(posts):,} after removing {d['rows_removed']} duplicate rows "
          f"({d['duplicated_post_ids']} post_ids)")
    print(f"timestamps {t['format_counts']}  (date-only, time unknown: {t['date_only_rows_time_unknown']:,})")
    print(f"platform NULL {audit['platform']['missing_total']:,} | likes NULL {l['missing_total']:,}, "
          f"sign-flips fixed {l['negative_values']} | text NULL {x['missing_total_after_cleaning']:,}, "
          f"suffix artifacts stripped {x['rows_with_artifact']:,}, whitespace collapsed {x['whitespace_collapsed_rows']}")
    print(f"users {len(users):,} validated | change log {len(changes):,} rows")
    print("wrote data/cleaned/posts_cleaned.csv, data/cleaned/users_cleaned.csv, reports/cleaning_audit.json, reports/change_log.csv")
