# Data dictionary: cleaned tables

Encoding: UTF-8, comma-separated, `\n` line endings, header row. **An empty field means NULL**, i.e. unrecoverable in the source.
Every contract below is enforced by `src/validate.py` and by CHECK constraints in `sql/schema.sql`.
`social_engine_cleaned.json` holds the same two tables in one file (`tables.posts`, `tables.users`; NULL → `null`).

## `posts_cleaned.csv`: one row per unique post (12,000 rows)

| Column | Type | NULL? | Contract / meaning |
|---|---|---|---|
| `post_id` | text(12) | no | Primary key. 12 lowercase alphanumerics, unique. |
| `user_id` | text(13) | no | Foreign key → `users_cleaned.user_id`. Every value exists. |
| `platform` | text | yes (1,784) | One of `Facebook`, `Instagram`, `Reddit`, `Twitter`, `YouTube`. NULL where the source had `''` or `'NULL'`. **Not imputed.** |
| `text_content` | text | yes (1,779) | Post text with injected suffix artifacts removed and whitespace collapsed. Original wording otherwise untouched. NULL where the source was `''`/`'NULL'`, including `'NULL'` hidden behind an artifact. |
| `post_date` | date `YYYY-MM-DD` | no | Calendar date of the post (UTC), 2024-05-01 … 2025-04-30. Use for all day, week or month analysis. |
| `post_datetime_utc` | datetime `YYYY-MM-DD HH:MM:SS` | yes (3,526) | Full UTC timestamp. NULL when the source only had a date (`dd-mm-yyyy`). **Never padded to midnight.** |
| `timestamp_precision` | text | no | `second` (full timestamp known) or `day` (date only). `post_datetime_utc IS NULL` ⇔ `day`. |
| `timestamp_source_format` | text | no | Raw encoding: `iso8601` (4,805), `unix_epoch` (3,669) or `dd-mm-yyyy` (3,526). |
| `likes` | integer | yes (1,814) | 0–5,000. Sign-flipped negatives were restored with `abs()`. NULL where missing in source. **Not imputed.** |
| `shares` | integer | no | 0–2,000. Arrived clean; validated. |
| `comments` | integer | no | 0–1,000. Arrived clean; validated. |
| `dq_duplicate_copies_removed` | integer | no | How many extra identical copies of this post were in the raw file (0, 1 or 2). |
| `dq_likes_sign_corrected` | 0/1 | no | 1 if `likes` was negative in the raw file and sign-corrected (509 rows). |
| `dq_text_artifact_removed` | text | no | Suffix stripped from the raw text: `none`, `html_entity_amp`, `html_tag_br`, `html_tag_div`, `mojibake_e_acute`, `trailing_newlines`. |

Derived fields are left out of the table on purpose and computed in SQL (`v_posts` view):
`engagement = likes + shares + comments` (NULL when likes is NULL), `post_month`, `day_of_week`, and `post_hour_utc` (NULL for day-precision rows).

## `users_cleaned.csv`: one row per account (1,500 rows)

| Column | Type | NULL? | Contract / meaning |
|---|---|---|---|
| `user_id` | text(13) | no | Primary key, `user_` + 8 lowercase alphanumerics. |
| `location` | text | no | Original `"City, Country"` string (Singapore has no country part). |
| `city` | text | no | Parsed from `location`. |
| `country` | text | no | Parsed from `location`; `Singapore` for the city-state. 19 countries. |
| `language` | text(2) | no | ISO 639-1 code as supplied (10 codes). **Independent of country in this data (see report, Insight 7).** |
| `language_name` | text | no | English name for `language` (e.g. `zh` → Chinese). |
| `account_created` | date `YYYY-MM-DD` | no | 2023-01-01 … 2023-12-31. Always on or before the user's first post. |
| `follower_count` | integer | no | 109–49,944. |

## Row-count reconciliation

| | Rows |
|---|---|
| Raw posts file | 12,360 |
| − exact duplicate copies (352 post_ids: 344 pairs + 8 triplicates) | −360 |
| **posts_cleaned.csv** | **12,000** |
| Raw users file = **users_cleaned.csv** | 1,500 |
