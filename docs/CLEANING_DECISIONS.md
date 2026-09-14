# Cleaning decisions: evidence, action, alternatives

Every transformation in `src/clean.py`, with the evidence that justifies it. The evidence is **computed and
asserted in code**: if a future extract breaks an assumption, the pipeline raises an error instead of cleaning wrongly.
All counts are generated in `reports/cleaning_audit.json`. "Raw" means before de-duplication; "clean" means after.

**Guiding principle.** A value is changed only when the original can be recovered deterministically.
Anything unrecoverable becomes `NULL` and is never imputed, which satisfies the rulebook's rule that fabrication of data is prohibited.
Each change is written to `reports/change_log.csv` (15,730 rows: post_id, column, rule, old, new).

---

## D1 · Read everything as text
- **Action:** `pd.read_csv(dtype=str, keep_default_na=False)`.
- **Why:** pandas' defaults silently convert `'NULL'`, `'NaN'` and `''` to NaN and guess numeric types. That hides the
  corruption we need to measure. For example, `'NULLÃ©'` would survive as literal text while `'NULL'` vanished.

## D2 · Exact duplicate rows → keep first
- **Evidence:** 352 `post_id`s occur more than once (344 twice, 8 three times) = 360 extra rows. Every copy is
  identical in all 8 columns (asserted).
- **Action:** drop the 360 extra copies; record `dq_duplicate_copies_removed` on the kept row.
- **Alternative rejected:** merging or coalescing copies. It is unnecessary because the copies never disagree.

## D3 · Timestamps: three encodings
| Raw format | Raw rows | Clean rows | Action |
|---|---|---|---|
| `2025-04-13T20:12:18` (ISO 8601, naive) | 4,950 | 4,805 | Parsed with an explicit format, assumed UTC |
| `1722528840` (Unix epoch, seconds) | 3,788 | 3,669 | Converted to UTC (epoch is UTC by definition) |
| `25-09-2024` (dd-mm-yyyy, **no time**) | 3,622 | 3,526 | `post_date` kept; `post_datetime_utc` = NULL; `timestamp_precision` = `day` |

- **Evidence for day-first:** across all dd-mm-yyyy values the 2nd field never exceeds 12, while the 1st field
  exceeds 12 in 2,113 rows (asserted).
- **Evidence for one clock:** ISO and epoch values span the same window (2024-05-01 → 2025-04-30) with flat,
  matching hour-of-day profiles.
- **Why not pad date-only values with 00:00:00:** that would place 3,902 posts (32.5%) at hour 0, 11.1× any real
  hour, and fabricate a "midnight bot spike". The true hourly profile is flat (χ² p = 0.475).
- **Output format:** `YYYY-MM-DD HH:MM:SS`, which SQLite, PostgreSQL and MySQL all parse natively.

## D4 · Missing values: two spellings → NULL, not imputed
| Column | Raw `''` | Raw `'NULL'` | Clean NULLs |
|---|---|---|---|
| platform | 1,219 | 627 | 1,784 |
| likes | 1,229 | 629 | 1,814 |
| text_content | 1,196 | 550 | 1,779 (includes 91 `'NULL'` tokens hidden behind a suffix artifact) |

- **Evidence the data is MCAR:** each missing or corruption flag was tested against platform, source timestamp format
  and month (17 χ² tests). All are non-significant (min p = 0.145). Co-missingness matches independence
  (likes & platform: 260 observed vs 270 expected).
- **Why no imputation:**
  - *platform:* five near-equal classes, so the mode is right ~20% of the time. It would inflate Facebook from 17.3% to 32.1% of posts.
  - *likes:* independent of every other variable (|ρ| ≤ 0.024), so any model reduces to the median. Median fill puts
    1,817 posts on the single value 2,498 and shrinks the standard deviation by 7.9%. That distorts Phase 2 correlation and anomaly queries.
  - *text:* placeholder strings such as `[CONTENT UNAVAILABLE]` pollute word counts and search. NULL is unambiguous.
- **Why not drop incomplete rows:** only 7,366 posts (61.4%) are complete. Dropping the rest discards 39% of
  the data for zero bias reduction under MCAR.

## D5 · Negative likes → sign flip, `abs()`
- **Evidence:** all 525 raw negatives (509 after de-duplication) are float-formatted (`-2388.0`) while every
  non-negative value is a plain integer, the fingerprint of a separate numeric operation. The absolute values follow the
  same distribution as positive likes (KS p = 0.626; means 2,460.7 vs 2,493.5). No negative value is fractional (asserted).
- **Action:** `abs()`, cast to integer, flag `dq_likes_sign_corrected = 1`.
- **Alternative rejected:** NULLing negatives, which throws away recoverable information.

## D6 · Injected text suffixes → strip
| Artifact | Raw rows | Clean rows | Example ending |
|---|---|---|---|
| `&amp;` | 341 | 328 | `…#Lifestyle&amp;` |
| `<div>` | 338 | 335 | `…your feedback!<div>` |
| `<br>` | 325 | 311 | `…your thoughts!<br>` |
| `\n\n` | 337 | 329 | `…#Trending\n\n` |
| `Ã©` (UTF-8 "é" mis-decoded as Latin-1) | 316 | 306 | `…#TravelÃ©` |

- **Evidence:** each pattern occurs **only** as the final characters of a text, never mid-text and never stacked,
  and there is at most one per post (asserted). No `&`, `<` or `Ã` appears anywhere else in the corpus.
- **Why strip rather than decode:** `html.unescape("…#Lifestyle&amp;")` yields a stray trailing `&`, and fixing the mojibake
  yields a stray `é` glued to punctuation. Neither was ever part of the post. Decoding would also leave the tags and newlines behind.
- **Safety nets (audited no-ops on this extract):** `html.unescape` and tag removal for any residual entity or tag.

## D7 · Whitespace → collapse
- **Evidence:** 680 posts contain double spaces where a template slot was empty (`"Wouldn't recommend.  #Beauty"`).
- **Action:** collapse whitespace runs to one space and trim. Meaning is unchanged.

## D8 · What we deliberately did NOT change
| Observation | Why left as is |
|---|---|
| Template grammar (`"How do I fix about Apple's iPhone 15?"`, `"Pepsi Pepsi Zero Sugar"`) | Original content at template scale. Rewriting it would be fabrication. |
| Contradictory sentiment (`"Bummed out … Absolutely loving it."`) | A property of the data (40.9% of emotional posts), reported as an insight. |
| `language` inconsistent with `location` (`Berlin, Germany → ja`) | True language unknowable. Reported as Insight 7, not "corrected". |
| Users table | Passed every check (no missing, padded, duplicate or malformed values). Only helper columns were added (`city`, `country`, `language_name`). |

## Assumptions register
| # | Assumption | Status |
|---|---|---|
| A1 | Naive ISO timestamps share the UTC clock of the epoch values | Supported (same window, same hourly profile); not provable |
| A2 | `dd-mm-yyyy` is day-first | Proven by field ranges; asserted |
| A3 | Negative likes are sign flips | Supported by format fingerprint + distribution test; flagged |
| A4 | The five suffixes are injected artifacts | Supported (terminal-only occurrence); asserted |
| A5 | Duplicate post_ids are repeated ingestions | Proven (identical copies); asserted |
| A6 | Missing values are MCAR | Supported by 17 independence tests |
