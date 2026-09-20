# Reaction to a Major Delivery or Service Delay — Data Vortex · Round 3

**Team SE7EN**: Tanmay Singh · Panshul Arora
Data Vortex A'26 @ Aaruush, SRM Institute of Science & Technology

**Assigned topic:** *Reaction to a Major Delivery or Service Delay*

Round 2 restored the Social Engine's ability to read meaning. It still could not
watch a conversation **move**. Round 3 collects live public reaction to delivery
and service failures, applies the Round 2 model to it, and answers the question a
monitoring system actually exists to answer: *when did the mood change, and what
changed it?*

---

## Submission

| # | Form slot | File |
|---|---|---|
| 1 | Self-Collected Structured Dataset | [`SUBMISSION/Round3_Delay_Reactions_Dataset_Team_SE7EN.csv`](SUBMISSION/Round3_Delay_Reactions_Dataset_Team_SE7EN.csv) |
| 2 | Data Collection / Scraping Code | [`SUBMISSION/Round3_Collection_Script_Team_SE7EN.py`](SUBMISSION/Round3_Collection_Script_Team_SE7EN.py) |
| 3 | Analysis Notebook | [`SUBMISSION/Round3_Analysis_Notebook_Team_SE7EN.ipynb`](SUBMISSION/Round3_Analysis_Notebook_Team_SE7EN.ipynb) |
| 4 | Round 3 Analytical Report | *(PDF)* |

---

## What the run produced

| | |
|---|---|
| Reaction records collected | **123,195** across 5 platforms |
| After normalisation & dedupe | **76,121** rows, 44.98 days (2026-08-06 → 2026-09-20) |
| Delay-related | **15,207** (20.0%) |
| Brands · distinct authors | 45 · 70,182 |
| Ground-truth incidents | 244 |
| Attention series | 1,440 daily brand-days |
| Live HTTP requests · failures | 101 · 9 |

**Sentiment shifts: 9** change points at |Cohen's *d*| ≥ 0.20, in both directions.
**Engagement spikes: 27** at robust *z* > 3.5 — the largest being
2026-09-15 at **12.35× the median** (*z* = 20.5).

Both exceed the rulebook's minimum of two shifts and one spike.

### Does the Round 2 model still work here?

It is measured, not assumed — Google Play hands us an answer key, because every
review carries a star rating written by the same person as the text.

| | |
|---|---|
| Independent labels | **74,013** star ratings |
| Three-class accuracy | 0.6945 (macro-F1 0.5520, κ 0.4958) |
| **Polarity only** (Neutral dropped) | **0.8942** on 56,406 rows, κ 0.785 |

A model trained on 2015-era tweets reaches **89.4% agreement on
positive-vs-negative** in 2026 app-store reviews — a completely different domain.
It struggles on Neutral, which is where a tweet-trained three-class model is
least at home, and that limitation is carried forward explicitly rather than
hidden behind the aggregate.

### How the shifts were found — and one method that failed

The first approach tested every candidate day with Welch's t-test and corrected
with Benjamini–Hochberg. Pooled honestly across all **1,793 tests**,
**nothing survived**. That result is kept in `analysis.json` and reported in the
notebook rather than quietly dropped, because it is a real property of the data.

The diagnosis is that the question was being asked wrongly: *"when did the level
change?"* is one segmentation question, and turning it into ~1,800 significance
questions guarantees that an honest multiplicity correction rejects everything.
**PELT change-point detection** asks it directly and finds 9.

Two instruments run independently — the Round 2 model's sentiment, and the
reviewer's own star rating — so a break appearing in both is corroborated rather
than resting on the model whose transfer we just measured at 0.69.

### Trigger evidence is relevance-gated

Temporal proximity is not relevance. Something is always broken somewhere, so a
±2-day window over ten status pages will always return *an* incident. An earlier
version offered a Discord media-proxy outage as the reason Amazon India's
delivery sentiment moved — exactly the plausible-sounding story this pipeline
exists to avoid.

Evidence is now tiered. **Internal** evidence (distinctive vocabulary, delay-type
mix, app versions in the field) is always available and directly about the event.
**External** corroboration counts only when it actually names a brand in the
window, and every event carries an explicit verdict — including *"no external
corroboration found"*, which is what several honestly report.

### The submitted dataset

**15,207 rows — 100% of all delay-related reactions, text intact**, at
9.0 MB. 76,000 rows of real review prose is ~50 MB of CSV and cannot fit a
9.5 MB cap with readable text, so the choice was made on what the dataset is
*for*: the delay-related rows **are** the assigned topic. The full corpus —
including the non-delay comparison baseline — ships in the repo as
`round3_delay_reactions_full.csv.gz` and `.json.gz`.

---

## Why the dataset is built in three layers

Most answers to "explain this spike" are a plausible story told over a chart. A
plausible story is not evidence. So the collector gathers three independent
kinds of data, and the analysis is designed backwards from the requirement to
*attribute* a change rather than merely notice one.

| layer | question it answers | sources |
|---|---|---|
| **L1 reaction** | what did people say, when, and did others endorse it? | Google Play reviews (44 apps), Reddit, Mastodon, Google News, Hacker News |
| **L2 trigger** | what actually broke, and at what time? | FAA national airspace delay register, 10 public status pages |
| **L3 attention** | did the outside world notice too? | Wikipedia pageviews |

L2 is what turns *"sentiment dropped"* into *"sentiment dropped **because**"*.
L3 is the check against fooling ourselves: if a volume spike in our scraped text
coincides with a pageview spike recorded by a third party, the spike is about the
world and not about our scraper.

### Why Google Play is the backbone

It is the only source that supplies four things at once:

- a **1–5 star rating** written by the same person as the text — an *independent
  sentiment label* on ~100k rows, which is what lets us **measure** whether the
  Round 2 model survived the domain change instead of assuming it;
- a **thumbs-up count** — an engagement signal with a real denominator;
- a **precise timestamp**, so reactions bin hourly;
- an **app version** and any **company reply**, giving two candidate trigger
  mechanisms beyond the incident feeds.

Apple's customer-review RSS was tested first and is deprecated — it returns HTTP
200 with an empty entry list for every app, country and sort order. That negative
result is recorded rather than quietly dropped.

---

## Delay and reaction taxonomies

The brief asks about *reaction to delay*, so every row carries two labels applied
as ordered first-match rules — and, crucially, the **literal text span that
fired the rule**, in `delay_type_evidence` and `reaction_type_evidence`. Anyone
can read why a row is labelled the way it is and disagree. An unsupervised
clustering would look more sophisticated and be impossible to defend to a judge
asking "why is this row in that bucket?".

**Delay types:** `never_arrived` · `missing_items` · `stuck_in_transit` ·
`cancelled` · `refund_delay` · `outage` · `support_delay` · `long_wait` ·
`late_delivery` · `unspecified_delay`

**Reaction types:** `churn_threat` · `refund_demand` · `escalation` · `anger` ·
`sarcasm_humour` · `resigned` · `praise_recovery` · `informational` ·
`neutral_report`

**Delay domains:** `food_delivery` · `quick_commerce` · `parcel_courier` ·
`ecommerce` · `ride_hailing` · `airline` · `telecom_isp`

---

## Method

**Sentiment shifts** are located with **PELT change-point detection** (L2 cost,
BIC-scaled penalty, minimum 4-day segment) on the daily series, run separately
over two instruments — the Round 2 model's sentiment and the reviewer's own star
rating. Cohen's *d* is reported beside each break, with a floor of 0.20, so a
detectable-but-trivial move is visible as trivial.

The Welch/Benjamini–Hochberg scan is retained as a deliberately conservative
second opinion and its result — *nothing survives* — is published rather than
discarded. See "How the shifts were found" above for why that is a statement
about the framing of the question rather than about the data being flat.

Welch p-values attached to change points are **uncorrected** and labelled as
such: they describe the size of a break PELT located, they do not certify its
discovery.

**Engagement spikes** use a median/MAD robust z-score. A spike inflates the very
mean and standard deviation a conventional z-score would test it against, which
systematically under-detects the events we are hunting.

**Distinctive vocabulary** uses log-odds with an informative Dirichlet prior
(Monroe et al.), whose z-scores are comparable across words of very different
frequency — raw counts and ratios are not.

**Trigger attribution** searches a ±2-day window around each event, but counts
external evidence only when it is *about* the brands in that window — an
incident whose entity names one of them, or a headline that does. Temporal
coincidence alone is discarded, and the discard count is published. Every event
carries an explicit verdict, *including when no external corroboration is
found*.

---

## Conduct

Every request carries one identifying User-Agent with a contact address. Per-host
minimum intervals are enforced globally, backoff respects `Retry-After`, and an
on-disk cache prevents re-runs from re-requesting. Only public, unauthenticated
endpoints are read.

Author handles are replaced with salted SHA-256 pseudonyms **at the point of
collection** — no raw identifier is ever written to disk. The dataset studies
what was said and when, never who said it.

---

## Reproduce

```bash
pip install -r round3/requirements.txt
python round3/run_round3.py
```

Collection is the only step that touches the network; everything downstream is a
pure function of `data/raw`, so the analysis reproduces from the published raw
files long after the live endpoints have moved on.

```bash
python round3/run_round3.py --skip-play   # reuse reviews already on disk
python round3/run_round3.py --offline     # analyse on-disk data, fetch nothing
```

---

## Repository map

| Path | Contents |
|---|---|
| `src/config.py` | topic, pinned app roster, taxonomies, rate limits |
| `src/fetch.py` | polite HTTP: throttling, backoff, cache, ledger, anonymisation |
| `src/sources/play_reviews.py` | L1 primary — Google Play, 44 apps |
| `src/sources/social_news.py` | L1 secondary — Reddit, Mastodon, News, Hacker News |
| `src/sources/incidents.py` | L2 triggers — FAA register, status pages |
| `src/sources/attention.py` | L3 attention — Wikipedia pageviews |
| `src/collect.py` | orchestration + source audit |
| `src/normalise.py` | five schemas → one, dedupe, brand attribution |
| `src/classify.py` | delay/reaction taxonomies + Round 2 model application |
| `src/validate.py` | Round 2 transfer measured against star ratings |
| `src/analyse.py` | shift/spike detection, distinctive terms, trigger attribution |
| `src/figures.py` | every chart |
| `src/build_dataset.py` | published tables, data dictionary, quality report |
| `data/raw/` | one JSONL per source, exactly as collected |
| `data/processed/` | dataset, dictionary, time series |
| `reports/` | `analysis.json`, `round2_transfer.json`, `data_quality.json`, figures |
| `SUBMISSION/` | the files to upload |
