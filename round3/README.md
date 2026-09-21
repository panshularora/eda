# Reaction to a Major Delivery or Service Delay — Data Vortex · Round 3

**Team SE7EN**: Tanmay Singh · Panshul Arora
Data Vortex A'26 @ Aaruush, SRM Institute of Science & Technology

**Assigned topic:** *Reaction to a Major Delivery or Service Delay*

Round 2 restored the Social Engine's ability to read meaning. It still could not
watch a conversation **move**. Round 3 collects public reaction to delivery and
service failures, applies the Round 2 model to it, and answers the question a
monitoring system exists to answer: *when did the mood change, what changed it,
and who should be woken up?*

Every number below is read from `reports/CLAIMS.json`, which the pipeline
writes. Nothing in this file is typed by hand.

---

## Submission

| # | Form slot | File |
|---|---|---|
| 1 | Self-Collected Structured Dataset | [`SUBMISSION/Round3_Delay_Reactions_Dataset_Team_SE7EN.csv`](SUBMISSION/Round3_Delay_Reactions_Dataset_Team_SE7EN.csv) |
| 2 | Data Collection / Scraping Code | [`SUBMISSION/Round3_Collection_Script_Team_SE7EN.py`](SUBMISSION/Round3_Collection_Script_Team_SE7EN.py) |
| 3 | Analysis Notebook | [`SUBMISSION/Round3_Analysis_Notebook_Team_SE7EN.ipynb`](SUBMISSION/Round3_Analysis_Notebook_Team_SE7EN.ipynb) |
| 4 | Round 3 Analytical Report | *(PDF — built separately)* |

---

## What the run produced

| | |
|---|---|
| Reviews **enumerated** in the window | **432,191** |
| Reviews **kept** (quota sample) | **91,192** (21.1% of the frame) |
| Apps reaching the full window | **43 / 44** |
| Corpus after normalisation | **92,689** rows, 45.0 days (2026-08-06 → 2026-09-20) |
| Delay-related | **10,918** (11.8%) |
| Brands · brands in the balanced panel | 44 · **39** |

---

## The finding the round is built around

Two rating collapses of near-identical size, in the same brand, ten days apart.

| | Rapido 2026-08-19 - no_service_signature | Rapido 2026-08-27 - service_failure |
|---|---|---|
| mean rating | 3.54 → 1.97 | 3.53 → 2.49 |
| effect on reviewers | *d* = -0.90 | *d* = -0.56 |
| delay-related share | 7.5% → **7.8%** | 7.3% → **15.2%** |
| outage language | 1.4% → 0.0% | 1.0% → **17.7%** |

**The same drop. Different events, needing opposite responses.** A sentiment
series reports both as "about one star down". The second is an app, login or
payment failure — operations should have been paged. The first is not explained
by anything in the text.

That last sentence is the point. An earlier draft called the first event
*reputational*, on the strength of six reviews containing `ban`, `shame` and
`boycott`. The aggregate did not support it — safety vocabulary did not move —
so the label was withdrawn. The classifier returns `no_service_signature`,
which is what is actually known, and ships the diagnostics so a human can form
a hypothesis the system is not entitled to assert.

### Then the outside world was checked — independently

Both events were passed to the external-evidence layer, which only accepts a
headline that was returned by **that brand's own failure query**, names the
brand, describes a service failure, and is not about a trial, a share price or
a TV show.

- **27 August — confirmed by two outlets, same day.** *The Economic Times*:
  "Rapido faces tech outage, users report issues in several cities".
  *Dailyhunt*: "Rapido faces app outage as ride booking issue hit users". The
  outage was detected from public reaction text alone, before the news was
  consulted, and the news agrees on the day and on the kind of failure.
- **17–20 August — a candidate, not a finding.** On 19 August, the trough day,
  Odisha outlets report an Ola–Uber–Rapido **driver strike** in Bhubaneswar,
  with fifty drivers arrested. A strike would produce exactly what the text
  shows — short, angry one-stars about rides not being accepted, with no outage
  language. But the strike is regional and the review base is national, so the
  system surfaces it as a dated lead for a human to verify rather than naming it
  as the cause. That is the correct behaviour for a monitor that does not know.

Every market-wide event in the window — the volume spikes and domain-level
shifts — returns **"no external corroboration found"**. That is not a failure
of the evidence layer. With thirty-plus brands in a window, *some* brand always
has *some* failure headline, so only brands that dominate an event are eligible
to explain it.

Of 85 brand-days where a rating fell more than 1.5 SD below that
brand's own median, most carry **no signature at all**. A monitor that named a
cause for every dip would emit a finding a day and be ignored by week two.

---

## Before any finding: is the trend in the public, or in the scraper?

This is the control that decides whether anything else counts, and the first
version of this work failed it.

Reviews were paginated newest-first on a flat 5,000-per-app budget. A quiet app
reached the window edge on that budget; a busy one ran out after five days.
**Eighteen of forty-four brands entered the corpus part-way through the
window.** Daily volume climbed from 59 rows to 822 — correlating **r = 0.89
with the day index** and **ρ = −0.06 with Wikipedia pageviews for the same
brands**. It was a picture of our own pagination, and it had already produced
three findings that were not findings:

- a volume trend that was a pagination curve;
- twelve "ride-hailing engagement spikes" that were Rapido appearing on 5
  September and Uber on 8 September (Lyft and Ola, present throughout, were
  flat at 5–11 rows a day);
- a set of "newly distinctive" words at the headline change point — *rapido*,
  *jio*, *bigbasket*, *zepto* — which were the names of the brands that had
  just arrived.

**The collector was rebuilt as a survey.** It now *enumerates* every review in
the window and keeps a **quota sample of at most 80 per
brand-day**, drawn by reservoir sampling so the kept rows are a uniform sample
of the day rather than its most recent hour.

The test is whether restricting to the balanced panel still *changes* the
series. On the first collection it changed it completely. On this one it barely
moves it — which is what a corpus without an entry bias looks like.

| daily delay volume vs the day index | all brands | balanced panel |
|---|---:|---:|
| first collection (flat 5,000/app budget) | **r = +0.891** | r = +0.037 |
| this collection (census + quota sample) | **0.214** | 0.223 |

On the first corpus the entire trend lived in the brands that entered
mid-window: remove them and it vanished. On this one the two agree, because
there are no brands entering mid-window left to remove. That agreement is the
evidence the artefact is gone — not evidence it never existed. The panel
restriction stays in place anyway, because it is what makes the claim
checkable rather than asserted.

On the balanced panel the series now averages 216.80 delay-related
reactions a day at a coefficient of variation of 0.12.

Census counts and quota shares combine into a ratio estimator, so the activity
series is an estimate of **how many people actually complained** rather than of
how many rows we kept — with a finite-population-corrected binomial interval
attached.

### What the control cost, and what it bought

| | first version | after the fix |
|---|---:|---:|
| PELT change points | 9 | 2 |
| "engagement spikes" | 27 | 4 (3 volume, 1 endorsement) |
| shifts surviving Benjamini–Hochberg over the whole scan | **0** | **3** of 2,819 tests |

Cleaning the sampling did not cost findings. It converted fake ones into real
ones: the conservative multiplicity-corrected scan went from finding *nothing*
to finding 3 shifts that survive correction across
2,819 tests.

---

## Three numbers the first version reported that were wrong

**1. Cohen's *d* was computed on daily means.** PELT segments a daily series, so
its *d* divides by the SD of ~14 daily averages — small by construction. The
headline break was reported as **d = 1.54, "a large effect"**, for a shift of
**0.12 stars on a five-point scale**. Recomputed over the reviews themselves it
is **d = 0.10**. Both are now reported, under names that say which is which
(`cohens_d_daily`, `cohens_d_records`), and selection and ranking use the one
that describes people.

**2. A "12.35× engagement spike" was two reviews.** 88% of that day's
endorsement came from the top two reactions; the median reaction received zero
thumbs-up. Every endorsement spike now ships its top-1 and top-5 concentration
and a Gini coefficient, and volume spikes are separated from endorsement spikes
because 23 of the original 27 were volume.

**3. The notebook printed a number that contradicted the paragraph above it.**
`significant_after_fdr` was assigned the PELT count, so the notebook printed
"1,793 tests, 9 survive pooled FDR<0.05" directly beneath a paragraph
explaining that nothing survived. Fixed.

---

## Does the Round 2 model still work here?

Measured, not assumed — Google Play hands us an answer key, because every review
carries a star rating written by the same person as the text.

| | |
|---|---|
| Independent labels | **87,978** star ratings |
| Three-class accuracy | 0.7448 (macro-F1 0.5736, κ 0.5583) |
| **Polarity only** (Neutral dropped) | **0.9224** on 70,162 rows, κ 0.836 |
| At confidence ≥ 0.70 | accuracy 0.906, covering 59.6% of rows |

A model trained on 2015-era tweets reaches **0.9224 agreement on
positive-vs-negative** in 2026 app-store reviews — from a completely different
domain, register and decade.

**Neutral is where it breaks.** Recall 0.23, precision
0.05, and it emits **15,155** Neutral
predictions where there are only **3,475** three-star reviews —
over-emitting the class by 4.36×. Because
`sentiment_score` maps Neutral to 0.0, every one of those misrouted rows drags
a daily mean toward zero by an amount that varies with text length and brand.

**So the decision rule was repaired without touching the model.** Predict
Neutral only when *P*(Neutral) ≥ τ, else take the better of Negative and
Positive. τ = 0.66 is fitted on half the star-labelled rows; every
number below is measured on the other 43,915, which the fitting never
saw.

| held-out | accuracy | Cohen's κ | Neutral predictions |
|---|---:|---:|---:|
| argmax (as shipped) | 0.7465 | 0.5605 | 7,530 |
| recalibrated | **0.8323** | **0.6742** | 1,606 |
| *truth* | — | — | *1,741* |

**+0.0857 accuracy and +0.1136 κ from a single
threshold**, and the Neutral prediction count goes from over four times the
truth to almost exactly it. The Round 2 deliverable ships unchanged; what
changed is how its output is read. Retraining would have been out of scope —
a fine-tuned model is no longer the Round 2 model.

### The topic head, applied and then put down

Round 2's central finding was that `topic_category` was not an annotation: a
case-insensitive substring switch reproduced 9,000 of 9,000 labels exactly.
Round 3 can ask a question Round 2 could not — **did the switch travel?**
Replaying the recovered rule over this corpus, which it was never fitted to,
the learned topic head agrees with it on **92.4% of rows**.

So `r2_topic` ships because the rulebook requires the Round 2 model to be
applied, and it is **not interpreted anywhere**. Reporting "6,806 reactions
about Technical Issues" would be reporting a substring count with a topic's
name on it. The topic layer that *is* interpreted is the delay/reaction
taxonomy, which ships the literal span that fired every label.

---

## Why the dataset is built in three layers

| layer | question it answers | sources |
|---|---|---|
| **L1 reaction** | what did people say, when, and did others endorse it? | Google Play (44 apps, census + quota sample), Reddit (feeds + topic search), Lemmy, Mastodon (6 instances), Google News, Hacker News |
| **L2 trigger** | what actually broke, and at what time? | FAA airspace delay register; status pages restricted to vendors in the delivery and commerce chain |
| **L3 attention** | did the outside world notice too? | Wikipedia pageviews |

### The trigger layer, rebuilt — and a structural limit worth reporting

The first roster was Discord, Dropbox, Twilio, Squarespace, Datadog, Zoom and
GitHub. It returned 238 of 244 incidents and **not one of them could make
somebody's dinner late.** After the relevance gate they contributed nothing, so
the layer was ornamental.

The reason is a finding about the topic rather than about the code:
**the operators whose delays the public reacts to do not publish
machine-readable status.** `status.doordash.com`, `status.uber.com`,
`status.zomato.com` and `status.lyft.com` do not resolve. Only infrastructure
vendors run public Statuspage instances. A trigger layer built from status feeds
therefore cannot, *even in principle*, corroborate a food-delivery spike.

The roster is now restricted to vendors that sit in the delivery and commerce
chain, each tagged `direct` or `infra`, and the structural limit is reported
rather than papered over with a coincidence.

### Two gates now stand between a headline and the word "trigger"

The first version required only that a headline in the window contain a brand
name, and so offered *"Flipkart widens lead over Amazon in India as quick
commerce surges"* as corroboration that delivery sentiment had moved. A brand
appearing in a business story is not evidence its service failed. A headline now
has to **name the brand** *and* **describe a failure**, and news is collected
per brand with the failure terms in the query rather than by free-text topic
search.

---

## Sources tested and found unusable

Reported rather than quietly dropped, because a source roster that lists only
what worked is not a method section.

| source | status |
|---|---|
| **Apple customer-review RSS** | deprecated — HTTP 200 with an empty entry list for every app, country and sort order |
| **Bluesky** | HTTP 403 from `public.api.bsky.app` and `api.bsky.app` on every query and User-Agent; `bsky.social` returns 401 without a session |
| **Reddit JSON API** | 403 to unauthenticated readers, including `search.json` with a browser User-Agent |
| **X / Twitter** | not free to read; not attempted, and not claimed |

Reddit's *Atom* surface is open and is what we use, but it rate-limits
unpredictably — the same query returns 429 twice and 200 seconds later. The
polite retry ladder can therefore spend minutes on a URL that was never going to
answer, so both Reddit collectors run under a **wall-clock budget** and report
how many feeds answered. A smaller dataset honestly described beats a pipeline
one hostile host can hold.

---

## Delay and reaction taxonomies

Every row carries two labels applied as ordered first-match rules, **plus the
literal span that fired the rule**, in `delay_type_evidence` and
`reaction_type_evidence`. Anyone can read why a row is labelled the way it is
and disagree.

**Delay types:** `never_arrived` · `missing_items` · `stuck_in_transit` ·
`cancelled` · `refund_delay` · `outage` · `support_delay` · `long_wait` ·
`late_delivery` · `unspecified_delay`

**Reaction types:** `churn_threat` · `refund_demand` · `escalation` · `anger` ·
`sarcasm_humour` · `resigned` · `recovery_acknowledged` · `informational` ·
`unmarked`

Two of those names changed, for cause:

- **`praise_recovery` → `recovery_acknowledged`.** The rule had no guard against
  the recovery verb being negated, so it fired on *"but it never gets fixed"*
  and *"but still haven't refunded me"*. All 17 rows it selected were
  complaints, mean sentiment −0.88. A label that says the opposite of the text.
- **`neutral_report` → `unmarked`.** This is the residual bucket, and calling it
  *neutral* asserted something false: its mean star rating was 1.9.

An ordered taxonomy is also mutually exclusive by construction, so it cannot
answer "how often do people demand a refund" — `anger` outranks
`refund_demand` and takes every complaint containing "worst" with it. Eight
**independent co-occurrence flags** now sit beside it, plus `competitor_named`,
which records who the customer says they are switching to. That last is
invisible to a brand-count entity analysis, because the brand named is not the
brand being reviewed.

---

## The relevance filter is audited, not trusted

`is_delay_related` decides which rows are the assigned topic, so everything
downstream inherits its errors. Three checks, in `reports/relevance_audit.md`:

**Word boundaries.** The first draft matched `late` inside *chocolate*, `down`
inside *download*, `eta` inside *retail* — the same class of defect the team
documented in the Round 2 labels.

**Validation against the star ratings**, which were never used to build it:

| | rows | mean stars | five-star share |
|---|---:|---:|---:|
| word-bounded only (first submission) | 15,207 | 1.600 | 8.9% |
| after the audit | **10,918** | **1.40** | **4.3%** |
| non-selected baseline | — | 3.13 | — |

The contrast against the baseline widens from 1.53 stars to **1.73**, and the
five-star share inside the "delay" corpus halves. Recall was paid for
precision, deliberately.

**Hand adjudication of 120 random rows: precision 0.875** (95% Wilson
[0.803, 0.923]). Reading them found three false-positive families that no amount
of staring at the regex had:

> *"Though **late night** food is not good, we still have Royal Dominos doing
> their duties sincerely"* — five stars, in the delay corpus because "late
> night" contains "late".

> *"Have place my very first order, **can't wait** to see how the results will
> be"* — anticipation, not a delay.

> *"...incurred **late-payment** or overdraft fees"* — and the hyphen is a word
> boundary, so the first attempt at a guard still let it through.

All three are now regression tests in `src/audit_relevance.py`.

---

## One more defect worth naming: the dedupe rule was deleting real people

Text-level deduplication was keyed on `source | normalised_text`. Every review
whose text was "good" collapsed to **one row** — across 44 brands, 45 days and
9,408 different people. "worst app" was written by **63 distinct authors with 63
distinct Play review IDs**, and one survived. On the current corpus that rule
discarded **1,291 rows non-randomly**: short, common texts
go first, so the survivors skew verbose, and verbose skews angry.

Cross-posting is real on social platforms and is still deduped there. It is not
real on an app store, where Play issues one immutable `reviewId` per review.
The key is now `source + brand + author` for the store and `source + text` for
the social sources.

---

## Method

**Shifts** are located with **PELT** change-point detection on the daily series,
run over two instruments — the Round 2 model's sentiment and the reviewer's own
star rating — so a break appearing in both is corroborated rather than resting
on the model whose transfer we just measured. Every break carries **both**
effect sizes, and a **balanced-panel re-test** that asks whether it survives
with composition held fixed.

A day-by-day Welch scan with a single pooled Benjamini–Hochberg family runs
alongside as a deliberately conservative second opinion. On the first corpus it
found nothing; on this one it finds **3** across
2,819 tests.

**Spikes** use a median/MAD robust *z*, separated into **volume** and
**endorsement**, each with concentration diagnostics. Note a bias we state
rather than discover later: thumbs-up is cumulative to the moment of
collection, so older days have had longer to accrue it. That works *against*
finding a recent spike, which makes any late-window endorsement spike a
conservative finding.

**Distinctive vocabulary** uses log-odds with an informative Dirichlet prior
(Monroe et al.), whose *z*-scores are comparable across words of very different
frequency — raw counts and ratios are not.

**Events** are classified per brand-day against **that brand's own baseline**,
because a 39% delay share is alarming for Domino's and an ordinary Tuesday for
Amazon India.

---

## Fields the collector gathered and the first analysis never opened

| field | filled on | what it says |
|---|---|---|
| `company_replied` | 32.8% of delay rows | whether the operator answered — operator-controlled, fast-moving, needs no model to read |
| `stated_delay_hours` | 18.4% of delay rows | median 3.00 h — the severity axis polarity cannot see |
| `app_version` | most Play rows | whether a release landed in the window |

A one-star review saying "late" and a one-star review saying "eleven days late,
still nothing" are the same point to a three-class sentiment model and very
different points to whoever has to decide what to do.

---

## Real-time, honestly

A single pull, however deep, is a **retrospective snapshot**: it reconstructs
45 days from whatever the platforms still hold today. Collection therefore runs
in **waves** (`src/waves.py`), appended to a per-source archive keyed on record
id. Only what a wave *adds* is genuinely live, and only that increment is
described as live anywhere in this submission.

Each wave measures **arrival latency** (how long after writing we see a
reaction) and **backfill** — records dated *before* the previous wave that only
surfaced now. Play moderates and releases reviews with a lag, so yesterday's
number keeps changing after yesterday. A monitoring system that reports a daily
figure without knowing its backfill rate is reporting a figure that will move
underneath it.

```bash
python round3/run_round3.py --wave     # record a wave without rebuilding
```

---

## Conduct

Every request carries one identifying User-Agent with a contact address. Per-host
minimum intervals are enforced globally, backoff respects `Retry-After`, and an
on-disk cache prevents re-runs from re-requesting. Only public, unauthenticated
endpoints are read. The Play census made **2,186 page requests**;
Reddit's final pass recorded 22 refusals under its wall-clock budget. Only
measured counts are reported — collection ran as several passes and no
cross-pass total is claimed.

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
python round3/src/audit_relevance.py      # reproduce the topic-filter audit
```

---

## Repository map

| Path | Contents |
|---|---|
| `src/config.py` | topic, pinned app roster, taxonomies, rate limits |
| `src/fetch.py` | polite HTTP: throttling, backoff, cache, ledger, anonymisation |
| `src/sources/play_census.py` | L1 primary — Google Play census + per-brand-day quota sample |
| `src/sources/social_news.py` | L1 secondary — Reddit, Lemmy, Mastodon, News, Hacker News |
| `src/sources/incidents.py` | L2 triggers — FAA register, supply-chain status pages |
| `src/sources/attention.py` | L3 attention — Wikipedia pageviews |
| `src/waves.py` | append-only wave bookkeeping: what each run actually added |
| `src/normalise.py` | schemas → one, dedupe, brand attribution |
| `src/classify.py` | delay/reaction taxonomies, flags, Round 2 model |
| `src/validate.py` | Round 2 transfer measured against stars, plus recalibration |
| `src/panel.py` | coverage, the balanced panel, the population estimator |
| `src/analyse.py` | shifts, spikes, distinctive terms, trigger attribution |
| `src/cases.py` | which brand-days are events, and what kind |
| `src/audit_relevance.py` | reproduces the topic-filter audit and its regression tests |
| `src/claims.py` | every reported number, written to `reports/CLAIMS.json` |
| `data/raw/` | one JSONL per source, exactly as collected |
| `data/waves/` | per-wave archives and the increment each run added |
| `reports/` | `analysis.json`, `case_studies.json`, `CLAIMS.json`, audits, figures |
| `SUBMISSION/` | the files to upload |
