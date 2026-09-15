-- =============================================================================
-- Data Vortex 2026  |  Round 1 Phase 2  |  Team SE7EN
-- Selected questions: E3, M2, H4
-- Engine: MySQL 8.0 (CTEs + window functions)
--
-- Run after 00_schema.sql and after the two CSVs have been loaded.
-- Each statement is self-contained against `users` and `posts` so a judge
-- can execute them one at a time. Nothing in the SELECT list is hardcoded.
-- =============================================================================

USE social_engine;

-- -----------------------------------------------------------------------------
-- E3  Average Engagement by Platform
-- Challenge: average likes, shares, comments per platform, and which platform
--            has the highest average total engagement.
--
-- Filter: drop the 1,784 posts with a missing platform. Those rows have no
-- group to sit in; inventing a bucket called "Unknown" would be a different
-- question.
-- Metric: AVG(likes + shares + comments). If likes is NULL the sum is NULL
-- and AVG skips it. We do not COALESCE likes to 0 — that would treat "not
-- recorded" as "nobody liked this", pull the mean down, and mix two different
-- denominators.
-- Answer is row 1 after ORDER BY avg_total_engagement DESC.
-- -----------------------------------------------------------------------------
SELECT
    platform,
    COUNT(*)                                       AS post_count,
    COUNT(likes)                                   AS posts_with_likes,
    ROUND(AVG(likes), 2)                           AS avg_likes,
    ROUND(AVG(shares), 2)                          AS avg_shares,
    ROUND(AVG(comments), 2)                        AS avg_comments,
    ROUND(AVG(likes + shares + comments), 2)       AS avg_total_engagement
FROM posts
WHERE platform IS NOT NULL
GROUP BY platform
ORDER BY avg_total_engagement DESC;


-- -----------------------------------------------------------------------------
-- M2  Do High Follower Users Get More Engagement?
-- Challenge: split users at 25,000 followers and compare average engagement
--            per post.
--
-- Grain is the post, not the user: the question asks for engagement per post,
-- so a 22-post account should count 22 times, not once. The CTE only labels
-- the group; it does not aggregate. AVG() again skips NULL likes.
-- vs_overall_avg is the group mean minus the platform-wide post mean, so a
-- near-zero number is a real finding, not a query that "failed to differ".
-- -----------------------------------------------------------------------------
WITH labelled AS (
    SELECT
        u.user_id,
        u.follower_count,
        p.likes,
        p.shares,
        p.comments,
        CASE
            WHEN u.follower_count >= 25000 THEN 'High follower (>= 25,000)'
            ELSE 'Low follower (< 25,000)'
        END AS follower_group
    FROM users AS u
    INNER JOIN posts AS p
        ON p.user_id = u.user_id
)
SELECT
    follower_group,
    COUNT(DISTINCT user_id)                        AS user_count,
    COUNT(*)                                       AS post_count,
    COUNT(likes)                                   AS posts_with_engagement,
    ROUND(AVG(follower_count), 0)                  AS avg_followers,
    ROUND(AVG(likes), 2)                           AS avg_likes,
    ROUND(AVG(shares), 2)                          AS avg_shares,
    ROUND(AVG(comments), 2)                        AS avg_comments,
    ROUND(AVG(likes + shares + comments), 2)       AS avg_engagement_per_post,
    ROUND(
        AVG(likes + shares + comments)
        - (SELECT AVG(likes + shares + comments) FROM posts),
        2
    )                                              AS vs_overall_avg
FROM labelled
GROUP BY follower_group
ORDER BY follower_group;


-- -----------------------------------------------------------------------------
-- H4  Follower to Engagement Anomaly
-- Challenge: users with fewer than 5,000 followers whose total post
--            engagement is in the top 10% of all users.
--
-- Step 1  collapse posts to one row per user (COUNT / AVG / SUM).
-- Step 2  NTILE(10) OVER (ORDER BY total_engagement DESC): decile 1 is the
--         top 150 of 1,500 users. We use a window, not a hardcoded cutoff
--         such as "total > 40000", because the brief forbids hardcoded
--         outputs and the 90th percentile should move if the table does.
-- Step 3  keep follower_count < 5000 inside that decile.
--
-- SUM(likes + shares + comments) is NULL for the two users who never have a
-- recorded like. ORDER BY total_engagement DESC sends those NULLs to the
-- bottom in MySQL 8, so they cannot land in decile 1. PERCENT_RANK is
-- returned only as a reading aid (100 = highest total in the 1,500).
-- -----------------------------------------------------------------------------
WITH user_engagement AS (
    SELECT
        u.user_id,
        u.location,
        u.follower_count,
        COUNT(p.post_id)                               AS post_count,
        COUNT(p.likes)                                 AS posts_with_engagement,
        ROUND(AVG(p.likes + p.shares + p.comments), 2) AS avg_engagement,
        SUM(p.likes + p.shares + p.comments)           AS total_engagement
    FROM users AS u
    INNER JOIN posts AS p
        ON p.user_id = u.user_id
    GROUP BY
        u.user_id,
        u.location,
        u.follower_count
),
ranked AS (
    SELECT
        user_engagement.*,
        NTILE(10) OVER (ORDER BY total_engagement DESC)            AS engagement_decile,
        ROUND(PERCENT_RANK() OVER (ORDER BY total_engagement) * 100, 2)
                                                                   AS percentile_rank
    FROM user_engagement
)
SELECT
    user_id,
    location,
    follower_count,
    post_count,
    posts_with_engagement,
    avg_engagement,
    total_engagement,
    percentile_rank
FROM ranked
WHERE follower_count < 5000
  AND engagement_decile = 1
  AND total_engagement IS NOT NULL
ORDER BY total_engagement DESC;
