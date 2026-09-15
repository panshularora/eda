-- Run this after importing the two CSVs. It turns blank strings into NULL
-- and then checks the row counts we expect from the Phase 1 clean extract.

USE social_engine;

UPDATE posts
SET platform = NULL
WHERE platform = '' OR platform = 'NULL';

UPDATE posts
SET text_content = NULL
WHERE text_content = '' OR text_content = 'NULL';

-- likes is an INT column. Blank CSV cells should already have landed as NULL.
-- If they landed as 0 instead, likes_nulls below will be 0 instead of 1814
-- and the file has to be re-imported (there is a real likes = 0 row, so we
-- cannot recover missing likes from zeros).

SELECT
    (SELECT COUNT(*) FROM users) AS users_rows,          -- 1500
    (SELECT COUNT(*) FROM posts) AS posts_rows,          -- 12000
    (SELECT SUM(platform IS NULL) FROM posts) AS platform_nulls,  -- 1784
    (SELECT SUM(likes IS NULL) FROM posts)    AS likes_nulls,     -- 1814
    (SELECT SUM(text_content IS NULL) FROM posts) AS text_nulls,  -- 1779
    (SELECT COUNT(DISTINCT user_id) FROM posts) AS distinct_posters,
    (SELECT COUNT(*) FROM posts p
        LEFT JOIN users u ON u.user_id = p.user_id
        WHERE u.user_id IS NULL) AS orphan_posts;        -- 0
