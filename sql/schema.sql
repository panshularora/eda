-- ============================================================================
-- Social Engine analytical schema · Team SE7EN · Data Vortex Round 1
-- Dialect: SQLite 3 (types and CHECKs port to PostgreSQL/MySQL with minor edits)
--
--   users 1 ──── N posts
--
-- Design notes
--   * Two tables at their natural grain: one row per account, one row per unique post.
--   * NULL means "unrecoverable in the source" (see docs/CLEANING_DECISIONS.md).
--     Never COALESCE likes/platform to a constant: the data is MCAR, so filtering is unbiased.
--   * post_date is always present; post_datetime_utc is NULL for the 3,526 posts whose
--     source only had a date. Hour-of-day queries must use WHERE timestamp_precision = 'second'.
--   * CHECK constraints mirror src/validate.py, so the database rejects any row that would
--     break a cleaning contract.
--   * dq_* columns keep the data-quality lineage queryable.
-- ============================================================================
PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS v_posts;
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    user_id          TEXT    PRIMARY KEY
                             CHECK (length(user_id) = 13 AND user_id GLOB 'user_*'),
    location         TEXT    NOT NULL,
    city             TEXT    NOT NULL,
    country          TEXT    NOT NULL,
    language         TEXT    NOT NULL CHECK (length(language) = 2),      -- ISO 639-1
    language_name    TEXT    NOT NULL,
    account_created  TEXT    NOT NULL CHECK (account_created = date(account_created)),  -- YYYY-MM-DD
    follower_count   INTEGER NOT NULL CHECK (follower_count >= 0)
);

CREATE TABLE posts (
    post_id                     TEXT    PRIMARY KEY CHECK (length(post_id) = 12),
    user_id                     TEXT    NOT NULL REFERENCES users (user_id),
    platform                    TEXT    CHECK (platform IN ('Facebook', 'Instagram', 'Reddit', 'Twitter', 'YouTube')),
    text_content                TEXT,
    post_date                   TEXT    NOT NULL CHECK (post_date = date(post_date)),                     -- YYYY-MM-DD
    post_datetime_utc           TEXT    CHECK (post_datetime_utc IS NULL
                                               OR post_datetime_utc = datetime(post_datetime_utc)),       -- YYYY-MM-DD HH:MM:SS
    timestamp_precision         TEXT    NOT NULL CHECK (timestamp_precision IN ('second', 'day')),
    timestamp_source_format     TEXT    NOT NULL CHECK (timestamp_source_format IN ('iso8601', 'unix_epoch', 'dd-mm-yyyy')),
    likes                       INTEGER CHECK (likes BETWEEN 0 AND 5000),
    shares                      INTEGER NOT NULL CHECK (shares BETWEEN 0 AND 2000),
    comments                    INTEGER NOT NULL CHECK (comments BETWEEN 0 AND 1000),
    dq_duplicate_copies_removed INTEGER NOT NULL DEFAULT 0 CHECK (dq_duplicate_copies_removed >= 0),
    dq_likes_sign_corrected     INTEGER NOT NULL DEFAULT 0 CHECK (dq_likes_sign_corrected IN (0, 1)),
    dq_text_artifact_removed    TEXT    NOT NULL DEFAULT 'none'
                                        CHECK (dq_text_artifact_removed IN ('none', 'html_entity_amp', 'html_tag_br',
                                                                           'html_tag_div', 'mojibake_e_acute', 'trailing_newlines')),
    CHECK ((timestamp_precision = 'day') = (post_datetime_utc IS NULL)),
    CHECK (post_datetime_utc IS NULL OR substr(post_datetime_utc, 1, 10) = post_date)
);

CREATE INDEX idx_posts_user_id   ON posts (user_id);
CREATE INDEX idx_posts_post_date ON posts (post_date);
CREATE INDEX idx_posts_platform  ON posts (platform);

-- Convenience view with the derived fields most analytical questions need.
CREATE VIEW v_posts AS
SELECT p.*,
       CASE WHEN p.likes IS NOT NULL THEN p.likes + p.shares + p.comments END            AS engagement,
       strftime('%Y-%m', p.post_date)                                                     AS post_month,
       CAST(strftime('%w', p.post_date) AS INTEGER)                                       AS day_of_week,  -- 0 = Sunday
       CASE WHEN p.timestamp_precision = 'second'
            THEN CAST(strftime('%H', p.post_datetime_utc) AS INTEGER) END                 AS post_hour_utc,
       u.city, u.country, u.language, u.follower_count, u.account_created
FROM posts AS p
JOIN users AS u USING (user_id);
