-- =============================================================================
-- Data Vortex 2026  |  Round 1 Phase 2  |  Team SE7EN
-- MySQL 8.0 schema for the cleaned Social Engine tables
--
-- Grain
--   users : one row per account          (1,500)
--   posts : one row per unique post      (12,000)
--   posts.user_id  ->  users.user_id     (every post has a matching user)
--
-- NULL means the source value was unrecoverable in Phase 1. It is not a zero.
-- Do not COALESCE likes or platform. AVG() and SUM() already skip NULL, which
-- is the unbiased treatment under MCAR missingness.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS social_engine
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE social_engine;

SET FOREIGN_KEY_CHECKS = 0;
DROP VIEW  IF EXISTS v_user_engagement;
DROP VIEW  IF EXISTS v_post_engagement;
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS users;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE users (
    user_id         VARCHAR(13)  NOT NULL,
    location        VARCHAR(64)  NOT NULL,
    city            VARCHAR(32)  NOT NULL,
    country         VARCHAR(32)  NOT NULL,
    language        CHAR(2)      NOT NULL,
    language_name   VARCHAR(32)  NOT NULL,
    account_created DATE         NOT NULL,
    follower_count  INT UNSIGNED NOT NULL,
    PRIMARY KEY (user_id),
    KEY idx_users_followers (follower_count),
    KEY idx_users_location  (location)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE posts (
    post_id                     VARCHAR(12)  NOT NULL,
    user_id                     VARCHAR(13)  NOT NULL,
    platform                    VARCHAR(16)  NULL,
    text_content                TEXT         NULL,
    post_date                   DATE         NOT NULL,
    post_datetime_utc           DATETIME     NULL,
    timestamp_precision         VARCHAR(8)   NOT NULL,
    timestamp_source_format     VARCHAR(16)  NOT NULL,
    likes                       INT UNSIGNED NULL,
    shares                      INT UNSIGNED NOT NULL,
    comments                    INT UNSIGNED NOT NULL,
    dq_duplicate_copies_removed INT UNSIGNED NOT NULL DEFAULT 0,
    dq_likes_sign_corrected     TINYINT(1)   NOT NULL DEFAULT 0,
    dq_text_artifact_removed    VARCHAR(32)  NOT NULL DEFAULT 'none',
    PRIMARY KEY (post_id),
    KEY idx_posts_user     (user_id),
    KEY idx_posts_platform (platform),
    KEY idx_posts_date     (post_date),
    CONSTRAINT fk_posts_user
        FOREIGN KEY (user_id) REFERENCES users (user_id)
            ON UPDATE CASCADE ON DELETE RESTRICT
    -- platform is one of Facebook/Instagram/Reddit/Twitter/YouTube, or NULL.
    -- CHECK is omitted so Workbench CSV import does not reject the 1,784
    -- blank platform cells (the wizard writes them as '' before we NULL them).
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Derived metric used by every Phase 2 question.
-- likes + shares + comments is NULL when likes is NULL, on purpose.
CREATE VIEW v_post_engagement AS
SELECT
    p.post_id,
    p.user_id,
    p.platform,
    p.post_date,
    p.likes,
    p.shares,
    p.comments,
    (p.likes + p.shares + p.comments) AS engagement,
    u.location,
    u.follower_count
FROM posts AS p
INNER JOIN users AS u
    ON u.user_id = p.user_id;

CREATE VIEW v_user_engagement AS
SELECT
    u.user_id,
    u.location,
    u.follower_count,
    COUNT(p.post_id)                              AS post_count,
    COUNT(p.likes)                                AS posts_with_engagement,
    ROUND(AVG(p.likes + p.shares + p.comments), 2) AS avg_engagement,
    SUM(p.likes + p.shares + p.comments)          AS total_engagement
FROM users AS u
INNER JOIN posts AS p
    ON p.user_id = u.user_id
GROUP BY
    u.user_id,
    u.location,
    u.follower_count;
