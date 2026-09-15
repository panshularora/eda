# Data Vortex 2026 — Round 1 Phase 2 (Team SE7EN)

Repo folder for the Phase 2 Google Form pack. Schema and queries are MySQL 8.0.

One question from each band of the Phase 2 questionnaire:

| Band | ID | Title | Why this one |
|---|---|---|---|
| Easy | **E3** | Average Engagement by Platform | Volume (E1) is a one-post gap. E3 asks where a post actually works. |
| Medium | **M2** | Do High Follower Users Get More Engagement? | Direct test of the follower-equals-reach assumption. |
| Hard | **H4** | Follower to Engagement Anomaly | The only hard item that needs two layers (percentile, then a filter) and does not return an empty set by construction. |

## Upload these four files

They live in `SUBMISSION/`:

1. `1_SQL_Queries.pdf`
2. `2_Output_Screenshot.jpeg`
3. `3_Logic_Explanation.pdf`
4. `4_Phase2_Insight_Report.pdf`

## Run it yourself in MySQL Workbench

Step-by-step: [`MYSQL_WORKBENCH_STEPS.md`](MYSQL_WORKBENCH_STEPS.md)

Short version:

1. Execute `sql/00_schema.sql`
2. Import `data/users_cleaned.csv` into `users`, then `data/posts_cleaned.csv` into `posts`
3. Execute `sql/02_verify_load.sql` — expect 1500 / 12000 / 1784 / 1814
4. Execute `sql/01_queries.sql` — three result grids
5. Screenshot the grids into one JPEG if you want a capture from your own machine

## Reproduce the numbers without Workbench

```
python build_deliverables.py
```

Rebuilds the SQLite check database, the three result CSVs, the JPEG, and the three PDFs from the cleaned CSVs. Same SQL as Workbench; SQLite 3.38 window functions match MySQL 8 for these queries.
