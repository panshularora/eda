# Audit of the topic-relevance filter

**Question.** `is_delay_related` decides which of the 76,000 collected reactions
belong to the assigned topic. Everything downstream — the sentiment series, the
shift detection, the spike detection, the entity analysis — is computed on the
rows it selects. If it is wrong, all of them are wrong in the same direction.
So it is audited rather than trusted.

Three checks are reported here, in increasing order of how much they cost and
how much they found.

---

## 1. Word boundaries (found before the first run)

The first draft of the filter was written without word boundaries. Replaying it
over the corpus showed the same defect the team documented in the Round 2
labels — a rule firing on a *substring* rather than a word:

| pattern | also matched |
|---|---|
| `late` | choco**late**, trans**late** |
| `down` | **down**load |
| `eta` | r**eta**il, b**eta**, m**eta** |

Every alternative was word-bounded. This is the class of error that is visible
from the pattern alone.

---

## 2. Validation against an independent label (74,013 star ratings)

Google Play supplies a 1–5 star rating written by the same person as the text.
It was not used to build the filter, so it is an independent instrument: a
correct topic filter should select rows whose ratings are much worse than the
corpus baseline.

| filter version | rows selected | mean star rating | five-star share |
|---|---:|---:|---:|
| v1 — word-bounded only (as submitted first) | 15,207 | 1.600 | 8.9% |
| v3 — after this audit | 12,796 | 1.401 | 4.3% |
| non-selected baseline | 63,325 | 3.132 | — |

The contrast against the baseline widens from **1.53 stars to 1.73 stars**, and
the share of five-star reviews inside the "delay" corpus **halves**. Recall was
paid for precision, deliberately: 2,411 rows were given up.

This check is cheap, objective and large-sample. It is also blind to *why* a
row was wrong, which is what the next check is for.

---

## 3. Hand adjudication of a random sample (n = 120)

**Protocol.** A random sample of 120 rows was drawn (seed 42) from everything
the v2 filter selected. Each was read in full and judged against a single
question:

> Does this text describe a delivery or service failure — something late,
> cancelled, never delivered, missing, unrefunded, or a service not working —
> experienced by the author or reported as news about one?

App and network failures count: an app that will not load is a service failure,
and the assigned topic is "delivery **or service** delay". Anticipation,
product-quality complaints with no timing or fulfilment element, and general UX
grievances do not count.

**Result: 105 of 120 correct — precision 0.875, 95% Wilson interval
[0.803, 0.923].**

### The fifteen false positives, and what they taught

Reading them grouped into three families, none of which was visible from the
pattern:

**(a) "late" in a time-of-day or finance sense — 3 rows**

> *"Though **late night** food is not good, we still have Royal Dominos doing
> their duties sincerely at this time."* — 5 stars, Domino's

> *"More than 4 in 10 who used BNPL for groceries incurred **late-payment** or
> overdraft fees"* — news headline

The second of these also defeated the first attempt at a guard, because the
hyphen in "late-payment" is a word boundary. The shipped guard uses `[\s-]*`.

**(b) "wait" as anticipation — 1 row**

> *"I love how early and beginner friendly this App is... Have place my very
> first order, **can't wait** to see how the results will be."* — 3 stars, SHEIN

**(c) off-topic rows from the open-web sources — 4 rows**

Hacker News and Mastodon are collected with broad topic queries, so *"China
**delays** mission to find water on Moon"* and *"North Devon maternity centre
reopening is **delayed** again"* arrive alongside parcel complaints. They are
genuine delays and genuinely not the assigned topic. These are not a filter
defect so much as a source-scope one, and they are why the non-Play sources
carry 4.6% of the delay corpus rather than being allowed to drive a finding.

The remaining seven were judgement calls at the boundary — subscription
cancellation difficulty, billing disputes, a product-quality complaint that
mentioned a refund — where a second reader could reasonably disagree. They are
counted as errors here because an audit that resolves its own ambiguities in
its favour is not an audit.

### What was changed in response

Families (a) and (b) were fixed and the whole corpus re-scored. Family (c) was
not "fixed": suppressing it would mean dropping the only non-app-store sources
in the dataset. It is reported instead, and the analysis never rests a claim on
a single Mastodon or Hacker News row.

**Annotation provenance.** The 120 adjudications were produced in a single pass
by one annotator against the written criterion above, with the sample, the
verdicts and the reasoning for every false positive published so that any judge
can re-read the same rows and disagree. There is no second annotator and
therefore no inter-annotator agreement figure; the boundary cases above are the
honest indication of where a second reader would most likely differ.

---

## What the audit does **not** establish

Precision, not recall. Estimating recall needs a random sample of the *negative*
class large enough to contain a useful number of missed delays, and at a base
rate near 17% that is a much larger annotation job than this round allows. The
v3 filter is deliberately tighter than v2, so its recall is lower, and the
direction — not the magnitude — of that trade is known.

Reproduce with:

```bash
python round3/src/audit_relevance.py
```
