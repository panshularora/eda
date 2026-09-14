"""
Loads the cleaned tables into SQLite and cross-checks the EDA in SQL.

    python src/build_sqlite.py

Creates data/social_engine.db from sql/schema.sql, loads data/cleaned/*.csv (empty field -> NULL),
enforces the foreign key, then runs sql/phase1_crosscheck_queries.sql and compares each result
with the pandas-computed values in reports/eda_stats.json. Exits non-zero on any mismatch.
Uses only the Python standard library.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "social_engine.db"
INT_COLUMNS = {"likes", "shares", "comments", "follower_count",
               "dq_duplicate_copies_removed", "dq_likes_sign_corrected"}


def load(conn: sqlite3.Connection, table: str, path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        rows = [tuple(None if row[c] == "" else int(row[c]) if c in INT_COLUMNS else row[c] for c in cols) for row in reader]
    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
    return len(rows)


def named_queries(path: Path) -> dict[str, str]:
    queries, name, buf = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-- name:"):
            if name:
                queries[name] = "\n".join(buf)
            name, buf = line.split(":", 1)[1].strip(), []
        elif name:
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf)
    return queries


def main() -> int:
    DB.unlink(missing_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))
    n_users = load(conn, "users", ROOT / "data" / "cleaned" / "users_cleaned.csv")
    n_posts = load(conn, "posts", ROOT / "data" / "cleaned" / "posts_cleaned.csv")
    conn.commit()
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    print(f"loaded users={n_users:,} posts={n_posts:,} into {DB.relative_to(ROOT)}; FK violations: {len(fk_violations)}")

    stats = json.loads((ROOT / "reports" / "eda_stats.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "reports" / "cleaning_audit.json").read_text(encoding="utf-8"))
    q = named_queries(ROOT / "sql" / "phase1_crosscheck_queries.sql")
    run = lambda name: conn.execute(q[name]).fetchall()  # noqa: E731

    checks = []
    posts, users = run("row_counts")[0]
    checks.append(("row counts", (posts, users), (stats["overview"]["posts"], stats["overview"]["users"])))
    nulls = run("null_counts")[0]
    expected_nulls = audit["output"]["posts_null_counts"]
    checks.append(("NULL counts", nulls, (expected_nulls["platform"], expected_nulls["likes"],
                                          expected_nulls["text_content"], expected_nulls["post_datetime_utc"])))
    checks.append(("complete rows", run("complete_rows")[0][0], stats["overview"]["complete_rows"]))
    checks.append(("hour-0 true vs naive", run("hour_zero_true_vs_naive")[0],
                   (stats["hour_of_day"]["true_hour0"], stats["hour_of_day"]["naive_hour0_if_padded_midnight"])))
    likes_sql = {platform: mean for platform, mean, _ in run("platform_mean_likes")}
    checks.append(("mean likes by platform", likes_sql, stats["engagement"]["by_platform"]["likes_f"]["means"]))
    per_day = {m: v for m, _, v in run("posts_per_day_by_month")}
    checks.append(("posts/day by month", per_day, stats["time"]["monthly_posts_per_day"]))
    top = [u for u, *_ in run("top_users_by_mean_engagement")]
    checks.append(("top-5 users by mean engagement", top, [u["user_id"] for u in stats["users"]["top10_by_mean_min5_known"][:5]]))
    checks.append(("users on 2+ platforms", run("users_multi_platform")[0][0], stats["users"]["users_on_2plus_platforms"]))

    failed = 0
    for name, got, want in checks:
        ok = got == want or (isinstance(got, tuple) and list(got) == list(want))
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] SQL == pandas: {name}" + ("" if ok else f"  sql={got} pandas={want}"))
    conn.close()
    if fk_violations:
        failed += 1
    print(f"{len(checks) - failed}/{len(checks)} SQL cross-checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
