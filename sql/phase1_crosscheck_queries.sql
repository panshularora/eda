-- ============================================================================
-- Phase 1 cross-checks in SQL · Team SE7EN
-- Each query recomputes a headline number from the EDA independently of pandas.
-- src/build_sqlite.py runs them and compares the results with reports/eda_stats.json.
-- Queries are separated by lines starting with "-- name:".
-- ============================================================================

-- name: row_counts
SELECT (SELECT COUNT(*) FROM posts) AS posts,
       (SELECT COUNT(*) FROM users) AS users;

-- name: null_counts
SELECT SUM(platform IS NULL)          AS platform_null,
       SUM(likes IS NULL)             AS likes_null,
       SUM(text_content IS NULL)      AS text_null,
       SUM(post_datetime_utc IS NULL) AS time_unknown
FROM posts;

-- name: complete_rows
SELECT COUNT(*) AS complete_rows
FROM posts
WHERE platform IS NOT NULL AND likes IS NOT NULL AND text_content IS NOT NULL;

-- name: hour_zero_true_vs_naive
-- True hour-0 volume uses only timed posts; the naive variant shows the artifact.
SELECT SUM(post_hour_utc = 0)                                    AS true_hour0,
       SUM(post_hour_utc = 0) + SUM(timestamp_precision = 'day') AS naive_hour0_if_padded
FROM v_posts;

-- name: platform_mean_likes
SELECT platform, ROUND(AVG(likes), 1) AS mean_likes, COUNT(likes) AS n_known_likes
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY platform;

-- name: posts_per_day_by_month
WITH monthly AS (
    SELECT post_month, COUNT(*) AS posts,
           CAST(strftime('%d', date(post_month || '-01', '+1 month', '-1 day')) AS INTEGER) AS days_in_month
    FROM v_posts
    GROUP BY post_month
)
SELECT post_month, posts, ROUND(1.0 * posts / days_in_month, 1) AS posts_per_day
FROM monthly
ORDER BY post_month;

-- name: top_users_by_mean_engagement
-- Per-post ranking with a minimum-activity threshold (>= 5 posts with known likes).
WITH per_user AS (
    SELECT user_id, COUNT(*) AS posts, COUNT(engagement) AS known, AVG(engagement) AS mean_engagement
    FROM v_posts
    GROUP BY user_id
)
SELECT user_id, posts, ROUND(mean_engagement, 1) AS mean_engagement,
       RANK() OVER (ORDER BY mean_engagement DESC) AS rnk
FROM per_user
WHERE known >= 5
ORDER BY rnk
LIMIT 5;

-- name: users_multi_platform
SELECT SUM(n_platforms >= 2) AS users_on_2plus_platforms
FROM (SELECT user_id, COUNT(DISTINCT platform) AS n_platforms FROM posts GROUP BY user_id);
