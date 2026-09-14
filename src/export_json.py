"""
Single-file JSON export of the cleaned dataset (both tables).

    python src/export_json.py

Reads   data/cleaned/posts_cleaned.csv, data/cleaned/users_cleaned.csv
Writes  data/cleaned/social_engine_cleaned.json

Same content as the two CSVs, for submission forms that accept a single file. NULL (empty CSV
field) becomes JSON null; integers stay integers. The export is read back and compared value by
value with the CSVs, and the script exits non-zero on any difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "cleaned"
OUT = CLEANED / "social_engine_cleaned.json"
INT_COLUMNS = {"likes", "shares", "comments", "follower_count",
               "dq_duplicate_copies_removed", "dq_likes_sign_corrected"}


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(CLEANED / name, dtype=str, keep_default_na=False)


def to_records(df: pd.DataFrame) -> list[dict]:
    return [{c: (None if v == "" else int(v) if c in INT_COLUMNS else v) for c, v in row.items()}
            for row in df.to_dict("records")]


def main() -> int:
    audit = json.loads((ROOT / "reports" / "cleaning_audit.json").read_text(encoding="utf-8"))
    posts, users = read_table("posts_cleaned.csv"), read_table("users_cleaned.csv")
    payload = {
        "dataset": "Social Engine: cleaned dataset (Data Vortex, Aaruush '26, Round 1 Phase 1)",
        "team": "Team SE7EN (Tanmay Singh, Panshul Arora)",
        "conventions": {
            "null": "value unrecoverable in the corrupted source (not imputed)",
            "post_datetime_utc": "null when the source only had a date; use post_date and timestamp_precision",
            "data_dictionary": "data/cleaned/DATA_DICTIONARY.md",
        },
        "source_sha256": {k: v["sha256"] for k, v in audit["inputs"].items()},
        "row_counts": {"posts": len(posts), "users": len(users)},
        "tables": {"users": to_records(users), "posts": to_records(posts)},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Round-trip check: every JSON value, rendered back to CSV text, must equal the CSV cell exactly.
    # (Compared as plain Python values; building a DataFrame would coerce nullable ints to floats.)
    back = json.loads(OUT.read_text(encoding="utf-8"))["tables"]
    for name, frame in [("posts", posts), ("users", users)]:
        restored = [["" if rec[c] is None else str(rec[c]) for c in frame.columns] for rec in back[name]]
        if restored != frame.values.tolist():
            print(f"FAIL: JSON table {name} differs from its CSV")
            return 1
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB); round-trip identical to both CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
