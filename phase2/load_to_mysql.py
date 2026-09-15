"""Load Phase 1 cleaned CSVs into MySQL 8 with proper NULLs.

    python load_to_mysql.py --user root --password YOUR_PASSWORD

Run sql/00_schema.sql first (Workbench or this script with --create-schema).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def nan_none(values):
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append(None)
        elif pd.isna(v):
            out.append(None)
        else:
            out.append(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", required=True)
    parser.add_argument("--create-schema", action="store_true")
    args = parser.parse_args()

    import mysql.connector

    conn = mysql.connector.connect(
        host=args.host, port=args.port, user=args.user, password=args.password
    )
    cur = conn.cursor()
    if args.create_schema:
        schema_sql = (ROOT / "sql" / "00_schema.sql").read_text(encoding="utf-8")
        for stmt in schema_sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        conn.commit()

    cur.execute("USE social_engine")
    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    cur.execute("DELETE FROM posts")
    cur.execute("DELETE FROM users")

    users = pd.read_csv(DATA / "users_cleaned.csv")
    posts = pd.read_csv(DATA / "posts_cleaned.csv")
    posts["likes"] = posts["likes"].astype("Int64")

    u_sql = (
        "INSERT INTO users (user_id, location, city, country, language, "
        "language_name, account_created, follower_count) VALUES "
        "(%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    u_rows = [nan_none(r) for r in users.itertuples(index=False, name=None)]
    cur.executemany(u_sql, u_rows)

    p_sql = (
        "INSERT INTO posts (post_id, user_id, platform, text_content, post_date, "
        "post_datetime_utc, timestamp_precision, timestamp_source_format, likes, "
        "shares, comments, dq_duplicate_copies_removed, dq_likes_sign_corrected, "
        "dq_text_artifact_removed) VALUES "
        "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    p_rows = [nan_none(r) for r in posts.itertuples(index=False, name=None)]
    cur.executemany(p_sql, p_rows)

    cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()
    cur.execute(
        "SELECT (SELECT COUNT(*) FROM users), (SELECT COUNT(*) FROM posts), "
        "(SELECT SUM(platform IS NULL) FROM posts), (SELECT SUM(likes IS NULL) FROM posts)"
    )
    print("users, posts, platform_nulls, likes_nulls =", cur.fetchone())
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
