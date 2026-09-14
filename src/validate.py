"""
Independent quality gate for the cleaned tables.

    python src/validate.py

Re-reads the files written by clean.py (not the in-memory frames) and asserts every
contract in data/cleaned/DATA_DICTIONARY.md. It also re-runs the pipeline in memory and
confirms the files on disk are byte-identical to a fresh run (reproducibility).
Writes reports/validation_report.md and exits non-zero if any check fails.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean  # noqa: E402

ROOT = clean.ROOT
POSTS = ROOT / "data" / "cleaned" / "posts_cleaned.csv"
USERS = ROOT / "data" / "cleaned" / "users_cleaned.csv"

results: list[tuple[str, str, bool, str]] = []


def check(area: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((area, name, bool(ok), detail))


def main() -> int:
    p = pd.read_csv(POSTS, dtype=str, keep_default_na=False)
    u = pd.read_csv(USERS, dtype=str, keep_default_na=False)
    raw = pd.read_csv(clean.RAW_POSTS, dtype=str, keep_default_na=False)
    audit = json.loads((ROOT / "reports" / "cleaning_audit.json").read_text(encoding="utf-8"))
    null = lambda s: s == ""  # noqa: E731  (NULL is written as an empty field)

    # ── structure
    check("structure", "posts columns match data dictionary", list(p.columns) == clean.POST_COLUMNS)
    check("structure", "users columns match data dictionary", list(u.columns) == clean.USER_COLUMNS)
    check("structure", "row count = raw rows - duplicate rows removed",
          len(p) == len(raw) - audit["duplicates"]["rows_removed"], f"{len(raw)} - {audit['duplicates']['rows_removed']} = {len(p)}")
    check("structure", "every raw post_id survives exactly once", set(p.post_id) == set(raw.post_id) and p.post_id.is_unique)

    # ── keys
    check("keys", "post_id unique, 12 lowercase alphanumerics", p.post_id.is_unique and p.post_id.str.fullmatch(r"[a-z0-9]{12}").all())
    check("keys", "user_id unique in users", u.user_id.is_unique)
    orphans = set(p.user_id) - set(u.user_id)
    check("keys", "every post.user_id exists in users (FK)", not orphans, f"{len(orphans)} orphans")

    # ── missing-value tokens
    for col in p.columns:
        leftovers = p[col].isin([t for t in clean.MISSING_TOKENS if t]).sum()
        check("missing tokens", f"no 'NULL'-style tokens left in posts.{col}", leftovers == 0, f"{leftovers} found")

    # ── platform
    check("platform", "platform in canonical set or NULL", p.platform[~null(p.platform)].isin(clean.PLATFORMS).all())

    # ── timestamps
    d = pd.to_datetime(p.post_date, format="%Y-%m-%d", errors="coerce")
    check("timestamps", "post_date always present and valid", d.notna().all())
    check("timestamps", "post_date inside 2024-05-01 .. 2025-04-30", d.between("2024-05-01", "2025-04-30").all())
    has_time = ~null(p.post_datetime_utc)
    dt = pd.to_datetime(p.post_datetime_utc[has_time], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    check("timestamps", "post_datetime_utc valid where present", dt.notna().all())
    check("timestamps", "datetime NULL <=> precision = 'day'", (has_time == (p.timestamp_precision == "second")).all())
    check("timestamps", "date part of datetime equals post_date", (p.post_datetime_utc[has_time].str[:10] == p.post_date[has_time]).all())
    check("timestamps", "no fabricated midnight: 00:00:00 share ≈ 1/86400 of timed rows",
          (p.post_datetime_utc[has_time].str[11:] == "00:00:00").sum() <= 3,
          f"{(p.post_datetime_utc[has_time].str[11:] == '00:00:00').sum()} rows at exactly midnight")
    check("timestamps", "source format labels valid", p.timestamp_source_format.isin(["iso8601", "unix_epoch", "dd-mm-yyyy"]).all())

    # ── engagement
    for col, hi, nullable in [("likes", 5000, True), ("shares", 2000, False), ("comments", 1000, False)]:
        present = p[col][~null(p[col])]
        check("engagement", f"{col} are plain non-negative integers", present.str.fullmatch(r"\d+").all())
        check("engagement", f"{col} within [0, {hi}]", present.astype(int).between(0, hi).all())
        if not nullable:
            check("engagement", f"{col} has no NULLs", not null(p[col]).any())
    check("engagement", "sign-corrected rows are exactly the raw negatives",
          int(p.dq_likes_sign_corrected.astype(int).sum()) == audit["likes"]["negative_values"])

    # ── text
    t = p.text_content[~null(p.text_content)]
    for label, pattern in [("HTML entities", r"&[#A-Za-z0-9]+;"), ("HTML tags", r"</?[A-Za-z][^>]*>"),
                           ("mojibake (Ã, â€)", r"Ã|â€"), ("newlines / tabs", r"[\n\r\t]"),
                           ("double spaces", r"  "), ("leading/trailing space", r"^\s|\s$"),
                           ("stray trailing '&'", r"&$"), ("'NULL' text", r"^NULL$")]:
        n = int(t.str.contains(pattern, regex=True).sum())
        check("text", f"no {label} in text_content", n == 0, f"{n} rows")
    check("text", "no placeholder text used for missing content", not t.str.contains(r"\[CONTENT UNAVAILABLE\]").any())

    # ── users
    ac = pd.to_datetime(u.account_created, format="%Y-%m-%d", errors="coerce")
    check("users", "account_created valid ISO dates", ac.notna().all())
    check("users", "follower_count non-negative integers", u.follower_count.str.fullmatch(r"\d+").all())
    check("users", "city/country consistent with location",
          ((u.city + ", " + u.country == u.location) | ((u.city == u.location) & (u.country == u.location))).all())
    m = p[["post_date", "user_id"]].merge(u[["user_id", "account_created"]], on="user_id")
    check("cross-table", "no post dated before its author's account was created", (m.post_date >= m.account_created).all())

    # ── reproducibility
    fresh_posts, fresh_users, _, _ = clean.run(write=False)
    for name, frame, path in [("posts", fresh_posts, POSTS), ("users", fresh_users, USERS)]:
        buf = io.StringIO()
        frame.to_csv(buf, index=False, lineterminator="\n")
        same = hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest() == clean.sha256(path)
        check("reproducibility", f"fresh pipeline run reproduces {path.name} byte-for-byte", same)

    # ── report
    failed = [r for r in results if not r[2]]
    lines = ["# Validation report", "",
             f"Generated by `src/validate.py`. **{len(results) - len(failed)} / {len(results)} checks passed.**", "",
             "| Area | Check | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {a} | {n} | {'PASS' if ok else '**FAIL**'} | {det} |" for a, n, ok, det in results]
    out = ROOT / "reports" / "validation_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for a, n, ok, det in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {a:15s} {n}" + (f"  ({det})" if det and not ok else ""))
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed -> {out.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
