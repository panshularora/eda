# Rebuilding the Social Engine: Data Vortex

**Team SE7EN**: Tanmay Singh · Panshul Arora
Data Vortex @ Aaruush '26, SRM Institute of Science & Technology

Round 1 restored the intake pipeline and the analytical core in MySQL. **Round 2
rebuilds the semantic comprehension layer with NLP** ([`round2/`](round2/README.md)).
**Round 3 puts it to work on live data** ([`round3/`](round3/README.md)).

---

## Round 2 — NLP comprehension layer

Two supervised tasks over 9,000 labelled social posts: sentiment (3 classes) and topic
(4 classes). TF-IDF word + character + surface features into regularised linear
classifiers, selected under grouped cross-validation that never splits a duplicated
post across folds. Held-out macro-F1 **0.6268** on sentiment (κ 0.441, against a
0.3369 stratified-guess floor) and **0.9426** on topic.

| Deliverable (form) | File |
|---|---|
| NLP model notebook | [`round2/SUBMISSION/Round2_NLP_Model_Notebook_Team_SE7EN.ipynb`](round2/SUBMISSION/Round2_NLP_Model_Notebook_Team_SE7EN.ipynb) |
| Trained models | [`round2/SUBMISSION/Round2_Trained_Models_Team_SE7EN.pkl`](round2/SUBMISSION/Round2_Trained_Models_Team_SE7EN.pkl) |
| Evaluation metrics report | [`round2/SUBMISSION/Round2_Evaluation_Metrics_Report_Team_SE7EN.pdf`](round2/SUBMISSION/Round2_Evaluation_Metrics_Report_Team_SE7EN.pdf) |
| Technical report | [`round2/SUBMISSION/Round2_Technical_Report_Team_SE7EN.pdf`](round2/SUBMISSION/Round2_Technical_Report_Team_SE7EN.pdf) |

The headline finding is about the data, not the model: **`topic_category` is not an
annotation.** A case-insensitive substring switch, recovered from the file by greedy
minimum-cover mining, reproduces **9,000 of 9,000** topic labels exactly — which makes
*Happy Friday friends!* a `Technical_Issues` post, because "happy" contains "app". We
report it rather than submitting the rule for a free 1.000. Full write-up and
reproduction: [`round2/README.md`](round2/README.md).

Rebuild everything from the raw CSV with `python round2/run_round2.py`.

---

## Round 3 — live monitoring of delivery and service delays

**Assigned topic:** *Reaction to a Major Delivery or Service Delay*

123,195 reaction records collected live from 5 platforms over 45 days, plus 244
documented incidents and an independent Wikipedia attention signal. The Round 2
model is applied and then **measured** against 74,013 star ratings the reviewers
wrote themselves: 89.4% agreement on polarity in a completely different domain.

| Deliverable (form) | File |
|---|---|
| Self-collected dataset | [`round3/SUBMISSION/Round3_Delay_Reactions_Dataset_Team_SE7EN.csv`](round3/SUBMISSION/Round3_Delay_Reactions_Dataset_Team_SE7EN.csv) |
| Collection / scraping code | [`round3/SUBMISSION/Round3_Collection_Script_Team_SE7EN.py`](round3/SUBMISSION/Round3_Collection_Script_Team_SE7EN.py) |
| Analysis notebook | [`round3/SUBMISSION/Round3_Analysis_Notebook_Team_SE7EN.ipynb`](round3/SUBMISSION/Round3_Analysis_Notebook_Team_SE7EN.ipynb) |

9 sentiment change points and 27 engagement spikes, the largest at 12.35x the
median. Full write-up and reproduction: [`round3/README.md`](round3/README.md).

```bash
python round3/run_round3.py
```

---

## Round 1 phase 2 — SQL analytical core

| Deliverable (form) | File |
|---|---|
| SQL queries | [`phase2/SUBMISSION/1_SQL_Queries.pdf`](phase2/SUBMISSION/1_SQL_Queries.pdf) |
| Output screenshot | [`phase2/SUBMISSION/2_Output_Screenshot.jpeg`](phase2/SUBMISSION/2_Output_Screenshot.jpeg) |
| Logic explanation | [`phase2/SUBMISSION/3_Logic_Explanation.pdf`](phase2/SUBMISSION/3_Logic_Explanation.pdf) |
| Insight report | [`phase2/SUBMISSION/4_Phase2_Insight_Report.pdf`](phase2/SUBMISSION/4_Phase2_Insight_Report.pdf) |

Runnable SQL: [`phase2/sql/00_schema.sql`](phase2/sql/00_schema.sql), [`phase2/sql/01_queries.sql`](phase2/sql/01_queries.sql). Workbench steps: [`phase2/MYSQL_WORKBENCH_STEPS.md`](phase2/MYSQL_WORKBENCH_STEPS.md). Notes: [`phase2/README.md`](phase2/README.md).

What the three queries say, in one line each: Instagram leads on average total engagement because of shares, not volume (E3). High- vs low-follower posts have the same engagement per post, 4,000.91 vs 4,003.01 (M2). Fifteen accounts with fewer than 5,000 followers sit in the top 10% of total engagement; the list is a mix of posting volume and a few high-rate tails, not a fraud ring (H4).

---

## Round 1 phase 1 — cleaned extract and EDA

Recovery, cleaning and exploratory analysis of the corrupted Social Engine intake dataset: evidence-first,
fully reproducible, and with **zero invented values**.

| Deliverable (rulebook) | Where |
|---|---|
| Cleaned dataset (CSV / JSON) | [`data/cleaned/posts_cleaned.csv`](data/cleaned/posts_cleaned.csv), [`data/cleaned/users_cleaned.csv`](data/cleaned/users_cleaned.csv) · both tables in one file: [`data/cleaned/social_engine_cleaned.json`](data/cleaned/social_engine_cleaned.json) · [data dictionary](data/cleaned/DATA_DICTIONARY.md) |
| EDA report | [`reports/EDA_Report_Team_SE7EN.pdf`](reports/EDA_Report_Team_SE7EN.pdf) (13 pages, 10 figures, 8 tested insights) |
| Cleaning code | [`src/clean.py`](src/clean.py) (documented, assertion-backed) |
| Documentation | This README · [cleaning decisions](docs/CLEANING_DECISIONS.md) · [dataset recovery](docs/DATA_RECOVERY.md) |
| Reproducible workflow | `python run_all.py` · [walkthrough notebook](notebooks/Phase1_Walkthrough.ipynb) (executed) |

---

## Results at a glance

| | |
|---|---|
| Raw → clean posts | **12,360 → 12,000** (360 exact duplicate rows removed) |
| Corruption types found | **9** (table below) |
| Values changed | **15,730**, each logged with its rule in [`reports/change_log.csv`](reports/change_log.csv) |
| Values imputed / invented | **0** (unrecoverable values are `NULL`; they are MCAR) |
| Validation | **53 / 53** contract checks pass, including byte-for-byte reproducibility ([report](reports/validation_report.md)) |
| SQL cross-check | **8 / 8** EDA numbers recomputed in SQLite match pandas |

## Corruptions found and fixed

| # | Corruption | Evidence (computed in code) | Fix | Raw | Clean |
|---|---|---|---|---:|---:|
| 1 | Duplicate rows | 352 post_ids ×2–3, all copies identical in every column | keep first | 360 | — |
| 2 | Unix-epoch timestamps | decode into the same window as ISO values | → UTC datetime | 3,788 | 3,669 |
| 3 | `dd-mm-yyyy` dates, **time of day lost** | month field ≤ 12 always; no time part | keep date, time = NULL, `precision='day'` | 3,622 | 3,526 |
| 4 | Missing platform (`''` + `'NULL'`) | MCAR (χ² tests) | NULL, not imputed | 1,846 | 1,784 |
| 5 | Missing likes (`''` + `'NULL'`) | MCAR; independent of all columns | NULL, not imputed | 1,858 | 1,814 |
| 6 | Negative likes | all float-formatted (`-2388.0`); \|neg\| ~ positives (KS p = 0.63) | `abs()` + flag | 525 | 509 |
| 7 | Missing text (`''`, `'NULL'`, and `'NULL'` hidden behind a suffix) | — | NULL (no placeholder) | 1,840 | 1,779 |
| 8 | Injected text suffixes: `&amp;` `<div>` `<br>` `\n\n` `Ã©` | only ever the last characters of a post | strip + flag kind | 1,657 | 1,609 |
| 9 | Double spaces from empty template slots | — | collapse | — | 680 |

The Users table passed every check; we only added `city`, `country` and `language_name`.
Reasoning, alternatives and assumptions for each fix: **[docs/CLEANING_DECISIONS.md](docs/CLEANING_DECISIONS.md)**.

## Key insights (details and tests in the report)

1. **The corruption is random (MCAR) and injected at fixed rates.** 17 independence tests are all non-significant, so available-case analysis is unbiased and imputation would only add false structure.
2. **The "midnight posting spike" is a parsing artifact.** Padding the 3,526 date-only posts with 00:00 makes hour 0 hold 32.5% of posts (11.1× any hour). Real activity is flat (χ² p = 0.475).
3. **Volume is stationary:** 32.9 posts/day with Poisson noise. Month totals are fully explained by month length (p = 0.986), and no day is anomalous after Bonferroni correction.
4. **Likes, shares and comments are independent uniform variables** (|Spearman ρ| ≤ 0.024, no heavy tail), a synthetic-data signature.
5. **No platform has a real engagement advantage.** Gaps are ≤ 5.6% of the mean and none survives multiple-testing correction.
6. **Volume, not reach, drives total engagement.** Followers vs per-post engagement ρ = 0.011; post count vs total ρ = 0.84. The top-10 by total and top-10 by per-post average share zero users.
7. **Language is independent of location** (Cramér's V = 0.11). The national-language match rate is 12.8% vs 12.4% by chance.
8. **Post text is templated:** 40.9% of emotional posts contradict their own verdict, sentiment does not move engagement, and seasonal campaigns are mentioned out of season.

Each insight in the report ends with a **"So what / Phase 2"** note on how to query the data correctly.

## How to run

```bash
pip install -r requirements.txt     # Python 3.11; pandas 2.2.2, numpy 1.26.4, scipy 1.17.1, matplotlib 3.10.9
python run_all.py                   # clean → validate → JSON → EDA → SQLite → report (~15 s)
```

| Step | Command | Produces |
|---|---|---|
| 1 | `python src/clean.py` | `data/cleaned/*.csv`, `reports/cleaning_audit.json`, `reports/change_log.csv` |
| 2 | `python src/validate.py` | `reports/validation_report.md` (non-zero exit on any failure) |
| 3 | `python src/export_json.py` | `data/cleaned/social_engine_cleaned.json` (round-trip checked against the CSVs) |
| 4 | `python src/eda.py` | `reports/figures/*.png`, `reports/eda_stats.json` |
| 5 | `python src/build_sqlite.py` | `data/social_engine.db` + SQL-vs-pandas cross-checks |
| 6 | `python src/build_report.py` | `reports/EDA_Report_Team_SE7EN.html` / `.pdf` (PDF via headless Chrome or Edge) |

The pipeline is deterministic: re-running it reproduces the cleaned CSVs byte-for-byte (SHA-256 recorded in the audit).
Raw inputs are hash-pinned to the files recovered from the site; `.gitattributes` stops Git from rewriting their line endings.

## Repository layout

```
├── data/
│   ├── raw/                      files recovered from Archive Node 07 (+ SHA256SUMS.txt)
│   └── cleaned/                  posts_cleaned.csv, users_cleaned.csv, social_engine_cleaned.json, DATA_DICTIONARY.md
├── src/
│   ├── clean.py                  cleaning pipeline (evidence → assert → fix → log)
│   ├── validate.py               53 contract checks + reproducibility check
│   ├── export_json.py            single-file JSON of both cleaned tables
│   ├── eda.py                    statistics and figures
│   ├── build_sqlite.py           SQLite load + SQL cross-checks (stdlib only)
│   └── build_report.py           report generator (numbers read from JSON, none hand-typed)
├── sql/
│   ├── schema.sql                typed schema, PK/FK, CHECK constraints, v_posts view
│   └── phase1_crosscheck_queries.sql
├── notebooks/Phase1_Walkthrough.ipynb   executed step-by-step walkthrough
├── reports/                      PDF/HTML report, figures, audit JSON, change log, validation report
├── docs/                         CLEANING_DECISIONS.md, DATA_RECOVERY.md
├── phase2/                       Round 1 Phase 2 SQL pack (E3, M2, H4)
│   ├── sql/                      MySQL 8 schema, verify-load, queries
│   ├── SUBMISSION/               PDFs + JPEG for the Google Form
│   └── MYSQL_WORKBENCH_STEPS.md
├── run_all.py
└── requirements.txt
```

## Round 1 phase 2 notes (how we queried)

NULL likes stay NULL — `AVG()` skips them; we never `COALESCE` to 0. Platform-missing rows are dropped only when the question is about a platform. User totals use `NTILE(10)` over all 1,500 accounts rather than a hardcoded cutoff. Details: [`phase2/README.md`](phase2/README.md).
