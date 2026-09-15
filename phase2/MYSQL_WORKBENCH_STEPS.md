# MySQL Workbench 8.0 — run this Phase 2 pack

The SQL in `sql/01_queries.sql` is written for **MySQL 8.0**. Workbench is already installed on this machine (`C:\Program Files\MySQL\MySQL Workbench 8.0`). Follow these steps in order. If a step is skipped, E3 / M2 / H4 will not match the PDFs.

## 0. Files you need

| File | What it is |
|---|---|
| `sql/00_schema.sql` | Creates `social_engine`, `users`, `posts`, and two views |
| `data/users_cleaned.csv` | 1,500 accounts (load this first) |
| `data/posts_cleaned.csv` | 12,000 posts |
| `sql/02_verify_load.sql` | Turns blank cells into NULL and checks counts |
| `sql/01_queries.sql` | E3, M2, H4 |

Blank cells in the CSVs are missing values, not zeros.

## 1. Connect

1. Open **MySQL Workbench 8.0 CE**.
2. Click the local connection (**Local instance MySQL80** / `root@127.0.0.1:3306`).
3. Enter the root password you set when MySQL was installed.

## 2. Build the schema

1. File → Open SQL Script → `sql/00_schema.sql`.
2. Click the lightning bolt (**Execute**).
3. In the SCHEMAS tree on the left, click the refresh icon.
4. You should see `social_engine` with tables `users` and `posts`.

If the schema already exists from a previous attempt, the script drops and recreates it.

## 3. Import users (first)

Foreign key: every post points at a user, so users have to exist first.

1. In SCHEMAS, right-click **social_engine** → **Table Data Import Wizard**.
2. Browse to `data/users_cleaned.csv`. Next.
3. Choose **Use existing table** → `social_engine.users`. Next.
4. Encoding: **utf-8**. Leave column mapping 1:1 (`user_id`, `location`, `city`, `country`, `language`, `language_name`, `account_created`, `follower_count`).
5. Next → Next → Finish.
6. Confirm 1,500 records imported.

## 4. Import posts (second)

1. Right-click **social_engine** → **Table Data Import Wizard** again.
2. Browse to `data/posts_cleaned.csv`. Next.
3. **Use existing table** → `social_engine.posts`. Next.
4. Encoding: **utf-8**. Keep the 14-column mapping as-is.
5. Finish. 12,000 records.

Workbench may write blank `platform` cells as empty strings. That is why step 5 exists.

## 5. Verify the load (do not skip)

1. Open `sql/02_verify_load.sql` and execute it.
2. The last result set must read:

| users_rows | posts_rows | platform_nulls | likes_nulls | text_nulls | distinct_posters | orphan_posts |
|---:|---:|---:|---:|---:|---:|---:|
| 1500 | 12000 | 1784 | 1814 | 1779 | 1500 | 0 |

If `likes_nulls` is **0**, the wizard stored missing likes as `0`. Re-import posts. There is a genuine `likes = 0` row in this extract, so zeros cannot be turned back into NULLs after the fact.

If `platform_nulls` is **0** after the UPDATE in that script, the import did not load the CSV you think it did.

## 6. Run the three questions

1. File → Open SQL Script → `sql/01_queries.sql`.
2. Execute the whole file (lightning bolt). Workbench opens **three Result Grid tabs**, one per SELECT.
3. Click through the tabs. You should see:
   - **E3:** 5 rows, Instagram on top, `avg_total_engagement` = 4040.02
   - **M2:** 2 rows, High 4000.91 vs Low 4003.01
   - **H4:** 15 rows, first `user_fgjkkrie` / Lyon / 62,690

To run one question at a time: highlight that statement only, then Execute (the lightning bolt with the cursor, or Ctrl+Shift+Enter).

## 7. Screenshot for the Google Form

The form asks for **one JPEG**. Capture all three grids.

1. View → Panels → **Hide Sidebar** if you need width.
2. For each Result Grid tab: make sure every column is readable (drag column edges).
3. Use Snipping Tool / Win+Shift+S. Include the SQL editor above the grid so the query and the output sit in the same picture.
4. Paste the three snips into Paint, stack them top to bottom, label them E3 / M2 / H4, save as `Phase2_Output_Screenshot.jpeg` (JPEG, not PNG, not .jpg if the form is picky).

A stacked JPEG is already in `SUBMISSION/2_Output_Screenshot.jpeg` from the same query results, for backup.

## 8. If the wizard fights you

Run `python load_to_mysql.py --user root --password YOUR_PASSWORD` from this folder. It uses `mysql-connector-python`, writes proper NULLs, and then prints the verification counts. You still run `01_queries.sql` inside Workbench for the screenshots the judges expect.
