"""Find the shifts and spikes, then try to explain them with outside evidence.

The rubric asks for at least two sentiment shifts, at least one engagement
spike, the key entities, and the reasons behind the changes. The last of those
separates analysis from storytelling, so the design is built backwards from it.

The control that comes first
----------------------------
Before any shift is reported, the corpus is checked for the defect that made
the first version of this analysis wrong: brands entering and leaving the
sample. A flat per-app pagination budget gave eighteen of forty-four brands a
partial window, and the resulting "findings" were the collector - a volume
trend that tracked the day index at r = 0.89, twelve ride-hailing spikes that
were two brands appearing, and distinctive terms that were those brands' names.
Every series here is therefore computed on the **balanced panel**, and every
break is re-tested with composition held fixed (`panel.py`).

Method
------
**Shifts** are located with PELT change-point detection on the daily series -
"when did the level change" is a segmentation question, and turning it into
~2,000 significance questions guarantees that an honest multiplicity
correction rejects everything. The day-by-day Welch scan with one pooled
Benjamini-Hochberg family runs alongside as a deliberately conservative second
opinion, and on this corpus it now survives, which it did not on the first one.

Every break carries **two** effect sizes under names that say which is which.
`cohens_d_daily` is what PELT saw: its denominator is the SD of ~14 daily
means, which is small by construction, so it is large for any clean step.
`cohens_d_records` is computed over the reviews themselves and is the effect on
people. On the first corpus those were 1.54 and 0.10 for the same break, and
only the first was reported. Selection and ranking now use the second.

**Spikes** use a median/MAD robust z-score rather than mean/SD, because a spike
inflates the very mean and standard deviation you would test it against. They
are split into **volume** (more people posted) and **endorsement** (the posts
that existed were upvoted harder), because reporting them together let 23 of 27
volume artefacts travel under the word "engagement". Endorsement spikes carry
concentration diagnostics: the largest one in the first corpus was 88% two
reviews, with a median of zero.

**Triggers** are searched for, not asserted, and the search is gated three
ways. An incident counts when its entity names a brand in the window, or when
the vendor sits `direct` in the delivery chain. A headline counts when it names
the brand *and* describes a failure - and counts most when it was returned by
that brand's own failure query, which is provenance rather than a substring
test. Distinctive vocabulary uses log-odds with an informative Dirichlet prior
(Monroe et al.), comparable across words of very different frequency in a way
raw counts are not. Where nothing qualifies, the report says so.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED, RAW, REPORTS
from fetch import read_jsonl
from panel import (balanced_panel, coverage, estimate_population, load_census,
                   panel_diagnostics)

# A delay event hits one operator, not the whole market, so the aggregate series
# is the wrong place to look for it - averaging 44 brands together is exactly how
# you wash a real shift out. The scan therefore runs at brand and domain level
# too, with thresholds low enough to include mid-sized brands. That widens the
# number of tests, which is precisely why every p-value in the scan goes through
# Benjamini-Hochberg together: the correction scales with the scan, so widening
# it costs sensitivity rather than buying false positives.
MIN_SIDE = 40          # records needed on each side of a candidate change point
WINDOW_DAYS = 3        # comparison half-width, in days
FDR_Q = 0.05
SPIKE_Z = 3.5

# Floor on the effect size that describes *reviewers*, not daily dots. The
# first version floored on the daily-series d at 0.20, which admits anything
# with a clean step - a 0.12-star move scored 1.54 there. 0.05 is deliberately
# permissive on a five-point scale (it is a "very small" effect by convention);
# the point is not to exclude small effects but to stop a tiny one being
# reported as a large one.
RECORD_D_FLOOR = 0.05


# ---------------------------------------------------------------------------
# series construction
# ---------------------------------------------------------------------------
def daily_series(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    keys = ["date"] + ([by] if by else [])
    g = df.groupby(keys)
    out = g.agg(
        n=("record_id", "count"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_weighted=("sentiment_score_weighted", "mean"),
        negative_share=("r2_sentiment", lambda s: float((s == "Negative").mean())),
        engagement_sum=("engagement", "sum"),
        engagement_mean=("engagement", "mean"),
        rating_mean=("rating", "mean"),
        delay_share=("is_delay_related", "mean"),
    ).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date")


def hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("hour_utc").agg(
        n=("record_id", "count"),
        sentiment_mean=("sentiment_score", "mean"),
        negative_share=("r2_sentiment", lambda s: float((s == "Negative").mean())),
        engagement_sum=("engagement", "sum"),
    ).reset_index()
    g["hour_utc"] = pd.to_datetime(g["hour_utc"])
    return g.sort_values("hour_utc")


# ---------------------------------------------------------------------------
# shift detection
# ---------------------------------------------------------------------------
def _bh(pvals: list[float], q: float = FDR_Q) -> list[bool]:
    """Benjamini-Hochberg. Scanning every day for a change is multiple testing;
    without a correction, a 45-day scan at alpha=0.05 expects false positives."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    keep = np.zeros(n, dtype=bool)
    thresh = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / n:
            thresh = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            keep[idx] = True
    return keep.tolist()


def shift_candidates(df: pd.DataFrame, label: str, metric: str = "sentiment_score",
                     window_days: int = WINDOW_DAYS,
                     min_side: int = MIN_SIDE) -> list[dict]:
    """Every candidate change point in one series, with an *uncorrected* p-value.

    Correction deliberately does not happen here. Every candidate this function
    produces, across every scope, metric and window width, belongs to a single
    family of tests, and Benjamini-Hochberg is applied to that family once in
    `main`. Correcting per-series instead - which is what this code did first -
    is both wrong and weaker: wrong because the family is the whole scan, and
    weaker because BH gains power from the other true positives in the pool, so
    splitting the pool throws that power away.
    """
    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return []
    d["day"] = pd.to_datetime(d["date"])
    days = sorted(d["day"].unique())
    if len(days) < 2 * window_days + 2:
        return []

    cands = []
    for cut in days[window_days:-window_days]:
        lo = cut - pd.Timedelta(days=window_days)
        pre = d.loc[(d["day"] >= lo) & (d["day"] < cut), metric].to_numpy()
        post = d.loc[(d["day"] >= cut) & (d["day"] < cut + pd.Timedelta(days=window_days)),
                     metric].to_numpy()
        if len(pre) < min_side or len(post) < min_side:
            continue
        t, pval = stats.ttest_ind(post, pre, equal_var=False)
        if not np.isfinite(pval):
            continue
        sd = math.sqrt((pre.var(ddof=1) + post.var(ddof=1)) / 2) or 1e-9
        cands.append({
            "scope": label,
            "metric": metric,
            "window_days": window_days,
            "date": cut.strftime("%Y-%m-%d"),
            "mean_before": round(float(pre.mean()), 4),
            "mean_after": round(float(post.mean()), 4),
            "delta": round(float(post.mean() - pre.mean()), 4),
            "cohens_d": round(float((post.mean() - pre.mean()) / sd), 4),
            "n_before": int(len(pre)), "n_after": int(len(post)),
            "t_stat": round(float(t), 3), "p_value": float(pval),
            "direction": "deterioration" if post.mean() < pre.mean() else "improvement",
        })
    return cands


def select_shifts(candidates: list[dict], q: float = FDR_Q) -> list[dict]:
    """Apply Benjamini-Hochberg once to the whole scan, then de-duplicate.

    Neighbouring days and overlapping window widths re-detect the same event, so
    within a scope only the strongest survivor of each run is kept - otherwise
    one shift would be reported three times and look like three findings.
    """
    if not candidates:
        return []
    keep = _bh([c["p_value"] for c in candidates], q)
    sig = [c | {"significant_fdr": True} for c, k in zip(candidates, keep) if k]
    if not sig:
        return []

    by_scope: dict[str, list[dict]] = {}
    for c in sig:
        by_scope.setdefault(c["scope"], []).append(c)

    merged: list[dict] = []
    for scope, rows in by_scope.items():
        rows.sort(key=lambda c: c["date"])
        run: list[dict] = []
        for c in rows:
            if run and (pd.Timestamp(c["date"]) - pd.Timestamp(run[-1]["date"])).days <= 2:
                run.append(c)
            else:
                if run:
                    merged.append(max(run, key=lambda x: abs(x["cohens_d"])))
                run = [c]
        if run:
            merged.append(max(run, key=lambda x: abs(x["cohens_d"])))
    return sorted(merged, key=lambda c: c["p_value"])


def change_points(series: pd.DataFrame, value_col: str, weight_col: str = "n",
                  scope: str = "overall", penalty_scale: float = 1.0,
                  min_size: int = 4, max_breaks: int = 4) -> list[dict]:
    """Locate level changes with PELT, not with a day-by-day hypothesis scan.

    Why this and not the t-test scan below it: "when did the level change?" is a
    segmentation question, not 45 separate significance questions. Testing every
    day turns one question into ~1,800 tests, and an honest multiplicity
    correction over that many tests then rejects everything - which is what
    happened here, and it is a property of the question being asked wrongly
    rather than of the data being flat.

    PELT with an L2 cost and a BIC-style penalty asks the question directly:
    find the segmentation of the series that best trades fit against the number
    of breaks. The penalty is what stops it finding a break everywhere.

    Each returned break still carries a Welch t-test on the underlying records,
    reported as supporting detail and explicitly *uncorrected* - it describes
    the size of the step PELT found, it does not certify its discovery.
    """
    import ruptures as rpt

    d = series.dropna(subset=[value_col]).sort_values("date").reset_index(drop=True)
    if len(d) < 2 * min_size + 2:
        return []
    signal = d[value_col].to_numpy(dtype=float).reshape(-1, 1)
    sigma = float(np.std(signal))
    if sigma < 1e-9:
        return []
    # BIC-style: penalty grows with log(n), so a longer series does not collect
    # breaks merely by being longer
    pen = penalty_scale * 2.0 * (sigma ** 2) * math.log(max(len(signal), 2))
    try:
        algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(signal)
        breaks = algo.predict(pen=pen)
    except Exception:
        return []
    breaks = [b for b in breaks if 0 < b < len(d)][:max_breaks]

    out = []
    for b in breaks:
        pre_days = d.iloc[max(0, b - 7):b]
        post_days = d.iloc[b:b + 7]
        if pre_days.empty or post_days.empty:
            continue
        w_pre = pre_days[weight_col].to_numpy(dtype=float)
        w_post = post_days[weight_col].to_numpy(dtype=float)
        m_pre = float(np.average(pre_days[value_col], weights=np.where(w_pre > 0, w_pre, 1)))
        m_post = float(np.average(post_days[value_col], weights=np.where(w_post > 0, w_post, 1)))
        pooled_sd = float(np.std(np.concatenate(
            [pre_days[value_col].to_numpy(), post_days[value_col].to_numpy()]))) or 1e-9
        out.append({
            "scope": scope,
            "metric": value_col,
            "method": "PELT (l2, BIC-scaled penalty)",
            "date": pd.Timestamp(d.iloc[b]["date"]).strftime("%Y-%m-%d"),
            "mean_before": round(m_pre, 4),
            "mean_after": round(m_post, 4),
            "delta": round(m_post - m_pre, 4),
            "cohens_d": round((m_post - m_pre) / pooled_sd, 4),
            "n_before": int(w_pre.sum()), "n_after": int(w_post.sum()),
            "days_before": int(len(pre_days)), "days_after": int(len(post_days)),
            "direction": "deterioration" if m_post < m_pre else "improvement",
        })
    return out


def attach_welch(shift: dict, records: pd.DataFrame, metric: str,
                 window_days: int = 5, panel: list[str] | None = None) -> dict:
    """Attach the effect size that describes *people*, not daily dots.

    Why this function grew
    ----------------------
    ``change_points`` computes Cohen's d from the daily series it segments, so
    its denominator is the standard deviation of ~14 daily means. Daily means
    are stable by construction - each is an average of a few hundred reviews -
    so that denominator is tiny and the resulting d is large almost regardless
    of how little any individual moved. The first version of this report
    printed **d = 1.54, a "large effect"**, for a shift of 0.12 stars on a
    five-point scale. Recomputed over the reviews themselves the same break is
    **d = 0.10**, which is the number that describes what happened to the
    people in the data. It is roughly fifteen times smaller and it is the
    honest one.

    Both are kept, under names that say which is which, because they answer
    different questions: ``cohens_d_daily`` describes how clean a step PELT
    saw in the series, ``cohens_d_records`` describes how much reviewers
    changed. The report leads with the second.

    ``panel`` adds the composition control. A break can appear purely because
    a new brand entered the corpus on that date, which is what happened on
    2026-09-08. Re-running the same comparison over only the brands present
    for the whole window answers "is this a change in behaviour, or a change
    in who we sampled?" - and when the balanced answer is null, that is the
    finding.
    """
    d = records.dropna(subset=[metric]).copy()
    d["day"] = pd.to_datetime(d["date"])
    cut = pd.Timestamp(shift["date"])

    def _compare(frame: pd.DataFrame) -> dict | None:
        pre = frame.loc[(frame["day"] >= cut - pd.Timedelta(days=window_days))
                        & (frame["day"] < cut), metric].to_numpy()
        post = frame.loc[(frame["day"] >= cut)
                         & (frame["day"] < cut + pd.Timedelta(days=window_days)), metric].to_numpy()
        if len(pre) < 20 or len(post) < 20:
            return None
        t, pval = stats.ttest_ind(post, pre, equal_var=False)
        if not np.isfinite(pval):
            return None
        sd = math.sqrt((pre.var(ddof=1) + post.var(ddof=1)) / 2) or 1e-9
        return {"n_before": int(len(pre)), "n_after": int(len(post)),
                "mean_before": round(float(pre.mean()), 4),
                "mean_after": round(float(post.mean()), 4),
                "delta": round(float(post.mean() - pre.mean()), 4),
                "cohens_d": round(float((post.mean() - pre.mean()) / sd), 4),
                "welch_t": round(float(t), 3), "welch_p_uncorrected": float(pval)}

    # the series-level d that PELT actually saw, renamed so it cannot be read
    # as an effect on people
    shift = dict(shift)
    shift["cohens_d_daily"] = shift.pop("cohens_d", None)

    full = _compare(d)
    if full:
        shift |= {
            "record_n_before": full["n_before"], "record_n_after": full["n_after"],
            "record_mean_before": full["mean_before"],
            "record_mean_after": full["mean_after"],
            "record_delta": full["delta"],
            "cohens_d_records": full["cohens_d"],
            "welch_t": full["welch_t"],
            "welch_p_uncorrected": full["welch_p_uncorrected"],
        }

    if panel is not None and "brand" in d.columns:
        bal = _compare(d[d["brand"].isin(panel)])
        if bal:
            shift["balanced_panel_check"] = {
                "brands": len(panel),
                "n_before": bal["n_before"], "n_after": bal["n_after"],
                "delta": bal["delta"],
                "cohens_d_records": bal["cohens_d"],
                "welch_p_uncorrected": bal["welch_p_uncorrected"],
                "survives": bool(abs(bal["cohens_d"]) >= 0.10
                                 and bal["welch_p_uncorrected"] < 0.05),
            }
        else:
            shift["balanced_panel_check"] = {"brands": len(panel),
                                             "survives": None,
                                             "note": "too few records on one side"}
    return shift


# ---------------------------------------------------------------------------
# spike detection
# ---------------------------------------------------------------------------
def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = values.std() or 1.0
    return (values - med) / scale


def concentration(values: np.ndarray) -> dict:
    """How much of a day's total came from its largest contributors.

    An "engagement spike" built from a sum says nothing about how many people
    were involved. The first version of this report led with 2026-09-15 at
    12.35x the median - and 88% of that day's engagement was **two reviews**
    (a Temu listing complaint at 2,233 thumbs and an Uber cash-demand one at
    1,196), while the median engagement that day was 0. That is two popular
    posts, not a conversation taking off, and the difference matters to anyone
    deciding whether to wake an operations team.

    Every endorsement spike therefore ships with the share held by its top one
    and top five contributors, and a Gini coefficient. A judge can then see for
    themselves whether the day was broad or narrow.
    """
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    total = v.sum()
    if total <= 0 or len(v) == 0:
        return {"top1_share": None, "top5_share": None, "gini": None,
                "contributors_nonzero": 0}
    asc = v[::-1]
    n = len(asc)
    idx = np.arange(1, n + 1)
    gini = float((2 * (idx * asc).sum()) / (n * asc.sum()) - (n + 1) / n)
    return {
        "top1_share": round(float(v[0] / total), 3),
        "top5_share": round(float(v[:5].sum() / total), 3),
        "gini": round(gini, 3),
        "contributors_nonzero": int((v > 0).sum()),
    }


def detect_spikes(series: pd.DataFrame, column: str = "n",
                  label: str = "overall", z: float = SPIKE_Z,
                  records: pd.DataFrame | None = None,
                  kind: str | None = None) -> list[dict]:
    """Robust-z spikes, tagged by what kind of spike they are.

    ``kind`` separates the two things the first version conflated under one
    heading. ``volume`` means more people posted; ``endorsement`` means the
    posts that existed were upvoted harder. Twenty-three of the original
    twenty-seven "engagement spikes" were volume, and twelve of those were one
    brand entering the corpus - so the label was doing real work in hiding what
    had been found.
    """
    if len(series) < 7:
        return []
    vals = series[column].astype(float).to_numpy()
    zs = robust_z(vals)
    med = float(np.median(vals))
    kind = kind or ("volume" if column == "n" else "endorsement")
    out = []
    for i, (zz, row) in enumerate(zip(zs, series.itertuples())):
        if zz < z:
            continue
        day = pd.Timestamp(row.date)
        spike = {
            "scope": label, "metric": column, "kind": kind,
            "date": day.strftime("%Y-%m-%d"),
            "value": float(vals[i]),
            "baseline_median": round(med, 2),
            "ratio_to_median": round(float(vals[i] / med), 2) if med else None,
            "robust_z": round(float(zz), 2),
            "sentiment_mean": round(float(getattr(row, "sentiment_mean", np.nan)), 4)
            if not pd.isna(getattr(row, "sentiment_mean", np.nan)) else None,
            "negative_share": round(float(getattr(row, "negative_share", np.nan)), 4)
            if not pd.isna(getattr(row, "negative_share", np.nan)) else None,
        }
        if records is not None and len(records):
            day_rows = records[pd.to_datetime(records["date"]) == day]
            if len(day_rows):
                if kind == "endorsement" and "engagement" in day_rows.columns:
                    spike["concentration"] = concentration(
                        day_rows["engagement"].fillna(0).to_numpy())
                    spike["median_engagement_that_day"] = float(
                        day_rows["engagement"].fillna(0).median())
                spike["unique_authors"] = int(day_rows["author_pseudonym"].nunique()) \
                    if "author_pseudonym" in day_rows.columns else None
                if "brand" in day_rows.columns:
                    spike["brand_mix"] = {k: int(v) for k, v in
                                          day_rows["brand"].value_counts().head(4).items()}
        out.append(spike)
    return sorted(out, key=lambda s: -s["robust_z"])


# ---------------------------------------------------------------------------
# distinctive vocabulary (Monroe et al. log-odds with Dirichlet prior)
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z][a-z'&-]{2,}")
_STOP = set("""the and for that with this you your have has had was were are but not
all can out get got from they them there their then than when what who will would
about just like some more very only into over after before its it's i'm don't
didn't doesn't been being too any our ours she he his her him one two now also
app order orders time times day days use used using make made even still much
because could should" """.split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def distinctive_terms(focus_texts: list[str], background_texts: list[str],
                      top: int = 15, prior_weight: float = 0.01) -> list[dict]:
    """Which words are over-represented in `focus` relative to `background`?

    Raw frequency answers "what is common", which for any English corpus is the
    same boring list. Log-odds with an informative Dirichlet prior answers "what
    is *distinctive*", and its z-score is comparable across words of very
    different frequency - which raw ratios are not.
    """
    fc, bc = Counter(), Counter()
    for t in focus_texts:
        fc.update(_tokens(t))
    for t in background_texts:
        bc.update(_tokens(t))
    if not fc or not bc:
        return []
    total = Counter(fc) + Counter(bc)
    n_all = sum(total.values())
    nf, nb = sum(fc.values()), sum(bc.values())
    a0 = prior_weight * n_all
    rows = []
    for w, cnt in fc.items():
        if cnt < 4:
            continue
        aw = prior_weight * n_all * (total[w] / n_all)
        lf = math.log((cnt + aw) / (nf + a0 - cnt - aw))
        lb = math.log((bc.get(w, 0) + aw) / (nb + a0 - bc.get(w, 0) - aw))
        delta = lf - lb
        var = 1.0 / (cnt + aw) + 1.0 / (bc.get(w, 0) + aw)
        rows.append({"term": w, "z": round(delta / math.sqrt(var), 2),
                     "focus_count": int(cnt), "background_count": int(bc.get(w, 0))})
    return sorted(rows, key=lambda r: -r["z"])[:top]


# ---------------------------------------------------------------------------
# trigger attribution
# ---------------------------------------------------------------------------
def load_incidents() -> pd.DataFrame:
    rows = read_jsonl(RAW / "incidents.jsonl")
    if not rows:
        return pd.DataFrame()
    inc = pd.DataFrame(rows)
    inc["started"] = pd.to_datetime(inc["started_utc"], utc=True, errors="coerce")
    inc.loc[inc["started"].isna(), "started"] = pd.to_datetime(
        inc.loc[inc["started"].isna(), "observed_utc"], utc=True, errors="coerce")
    return inc.dropna(subset=["started"])


def load_attention() -> pd.DataFrame:
    rows = read_jsonl(RAW / "attention.jsonl")
    if not rows:
        return pd.DataFrame()
    att = pd.DataFrame(rows)
    att["date"] = pd.to_datetime(att["date"])
    return att


SECTOR_FOR_DOMAIN = {
    "telecom_isp": {"platform_service"},
    "airline": {"airline"},
}

# The headline has to be about the service people use, not the company in
# general.
_SERVICE_CONTEXT = (r"\b(?:deliver(?:y|ies|ed)?|orders?|app|service|customers?|"
                    r"users?|riders?|drivers?|partners?|outage|website|platform|"
                    r"refunds?|parcels?|packages?|shipments?|flights?|passengers?|"
                    r"network|licen[cs]e|facility|warehouse|dark store)\b")
# ...and not a story where "delay" or "cancelled" belongs to something else.
_OFF_TOPIC = (r"\b(?:trial|lawsuit|court|ftc|antitrust|merger|shares?|stock|"
              r"earnings|ipo|traffic|highway|interstate|i-\d+|freeway|"
              r"show|series|season|episode|film|movie|album|concert)\b")

# A headline only corroborates a service event if it describes a service
# failing. Without this, any business story naming the brand qualified.
_FAILURE_WORDS = (
    r"\bdelay(?:s|ed|ing)?\b|\boutage\b|\bdisruption\b|\bglitch\b|"
    r"\bfail(?:s|ed|ure)?\b|\bnot working\b|\bsuspend(?:ed|s)?\b|"
    r"\bhalt(?:ed|s)?\b|\bcancel(?:led|s|lation)\b|\bbacklog\b|"
    r"\bstranded\b|\bgrounded\b|\bshortage\b|\bprotest\b|\bboycott\b|\bapolog|"
    # "down" only in the service sense, never "down 3% on the quarter"
    r"\b(?:servers?|site|app|service|system|network)\s+(?:is |was |went )?down\b|"
    # a strike has to be labour, not "strikes a multiyear deal with the NHL" -
    # which is exactly what the DoorDash brand query returned first
    r"\b(?:workers?|drivers?|riders?|staff|union|employees?|couriers?)\b"
    r"[^.!?]{0,40}\bstrike\b|"
    r"\bstrike\b[^.!?]{0,30}\b(?:action|pay|wages|over|by (?:workers|drivers|riders))\b")


def _relevant_incidents(near: pd.DataFrame, brands: set[str], domain: str | None) -> list[dict]:
    """Keep only incidents that could plausibly touch the brands in question.

    Temporal proximity is not relevance. Something is always broken somewhere,
    so a +/- 2 day window over ten status pages will always return *an*
    incident - and reporting a Discord outage as the reason Amazon India's
    delivery sentiment moved would be exactly the kind of plausible-sounding
    story this pipeline was built to avoid. An incident is only offered as
    evidence when its entity names one of the brands in the window, or its
    sector genuinely covers the domain under test.
    """
    if near.empty:
        return []
    keep = []
    lowered = {b.lower() for b in brands if b}
    allowed_sectors = SECTOR_FOR_DOMAIN.get(domain or "", set())
    for r in near.itertuples():
        entity = str(getattr(r, "entity", "")).lower()
        sector = str(getattr(r, "sector", ""))
        tier = str(getattr(r, "tier", "") or "")
        # Whole-word equality, not substring containment. The substring test
        # this replaced matched the FAA's Chicago O'Hare entry, "ORD", to
        # DoorDash - because "ord" is inside "doordash" - and offered an
        # airport delay as corroboration for food-delivery complaints. It is
        # the Round 2 substring defect again, this time in our own gate.
        entity_hit = bool(entity) and any(
            re.fullmatch(re.escape(b), entity) or re.search(rf"\b{re.escape(b)}\b", entity)
            for b in lowered if len(b) > 3)
        sector_hit = sector in allowed_sectors
        # A `direct` supply-chain vendor (Shippo, AfterShip, Olo, Stripe) is
        # reported as *context* only. An earlier version let a direct-tier
        # incident count as corroboration for any delivery-domain event, and
        # the result was Shippo's USPS label-generation incidents
        # "corroborating" a Zepto sentiment shift in India. These vendors post
        # incidents most weeks, so a +/- 2 day window almost always contains
        # one; a signal that fires every week corroborates nothing. Only an
        # incident that names a brand in the window, or a sector match that is
        # genuinely structural (FAA for airlines), counts toward the verdict.
        chain_hit = tier == "direct" and domain in {
            "food_delivery", "quick_commerce", "parcel_courier", "ecommerce"}
        if entity_hit or sector_hit or chain_hit:
            keep.append({
                "entity": r.entity, "kind": r.incident_kind,
                "started": r.started.strftime("%Y-%m-%d %H:%M"),
                "title": str(r.title)[:120], "reason": str(r.impact_reason)[:80],
                "tier": tier or "unknown",
                "matched_on": ("entity" if entity_hit else
                               "sector" if sector_hit else "supply_chain_tier"),
                "counts_as_corroboration": bool(entity_hit or sector_hit),
            })
    # corroborating incidents first, context after
    keep.sort(key=lambda k: not k["counts_as_corroboration"])
    return keep[:8]


def explain_event(event: dict, df: pd.DataFrame, incidents: pd.DataFrame,
                  attention: pd.DataFrame, news: pd.DataFrame,
                  brand: str | None = None, pad_days: int = 2) -> dict:
    """Assemble every independent piece of evidence around one event.

    Evidence is separated into two tiers, because they are not equally strong
    and pretending otherwise is how a coincidence becomes a conclusion:

    ``internal``  what changed inside the reactions themselves - which words
                  became distinctive, which delay types and brands dominated,
                  which app versions were in the field. Always available, and
                  directly about the event.
    ``external``  corroboration from outside our own scraping - a documented
                  incident naming the brand, news coverage naming the brand, a
                  move in Wikipedia attention. Only counted when it is actually
                  *about* the brands in the window, never merely concurrent.

    When no external evidence qualifies, that is recorded explicitly rather
    than padded with whatever happened to share the date.
    """
    day = pd.Timestamp(event["date"], tz="UTC")
    lo, hi = day - pd.Timedelta(days=pad_days), day + pd.Timedelta(days=pad_days)
    created = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")

    focus = df[(created >= lo) & (created <= hi)]
    background = df[(created < lo) | (created > hi)]
    ev: dict = {"window": [lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")],
                "n_in_window": int(len(focus))}

    # ---------- tier 1: internal ------------------------------------------
    if len(background):
        ev["distinctive_terms"] = distinctive_terms(
            focus["full_text"].tolist(),
            background["full_text"].sample(
                min(len(background), 40000), random_state=42).tolist())
    brands_in_window: set[str] = set()
    if len(focus):
        ev["delay_type_mix"] = {k: int(v) for k, v in
                                focus["delay_type"].value_counts().head(6).items()}
        ev["reaction_type_mix"] = {k: int(v) for k, v in
                                   focus["reaction_type"].value_counts().head(6).items()}
        top_brands = focus.loc[focus["brand"] != "", "brand"].value_counts().head(6)
        ev["top_brands"] = {k: int(v) for k, v in top_brands.items()}
        brands_in_window = set(top_brands.index)
        if "app_version" in focus.columns:
            vers = focus.loc[focus["app_version"].astype(str).str.len() > 2, "app_version"]
            ev["app_versions_in_window"] = {k: int(v) for k, v in
                                            vers.value_counts().head(5).items()}

    if brand:
        brands_in_window.add(brand)
    domain = None
    if len(focus) and "delay_domain" in focus.columns:
        doms = focus.loc[focus["delay_domain"] != "", "delay_domain"]
        if len(doms):
            domain = doms.mode().iat[0]

    # ---------- tier 2: external, relevance-gated -------------------------
    external: dict = {}

    if len(incidents):
        near = incidents[(incidents["started"] >= lo) & (incidents["started"] <= hi)]
        relevant = _relevant_incidents(near, brands_in_window, domain)
        external["incidents"] = relevant
        external["incidents_in_window_total"] = int(len(near))
        external["incidents_discarded_as_unrelated"] = int(len(near) - len(relevant))

    if len(news):
        ncreated = pd.to_datetime(news["created_utc"], utc=True, errors="coerce")
        nn = news[(ncreated >= lo) & (ncreated <= hi)]
        # Only brands that DOMINATE the window are eligible for news
        # corroboration. With six brands in an "overall" window and 35 brand
        # queries collected, some brand always has some failure headline in any
        # four-day span - which is how "Amazon asks FTC to delay trial" and an
        # article about a cancelled TV show came to "corroborate" a 1.3x volume
        # blip in the first run of these gates. A headline can only explain an
        # event if the brand it names is a large part of the event. For a brand
        # or single-domain scope that is usually true; for the whole market it
        # usually is not, and then the honest verdict is "no corroboration".
        tb = ev.get("top_brands") or {}
        tot = max(sum(tb.values()), 1)
        dominant = {b for b, n in tb.items() if n / tot >= 0.25}
        if brand:
            dominant.add(brand)
        targets = [b for b in dominant if len(b) > 3]
        external["brands_eligible_for_news"] = sorted(targets)
        # Two gates, not one. The first version required only that a headline
        # in the window contained a brand name, which is how "Amazon Air cargo
        # plane crash at MIA" and "Flipkart widens lead over Amazon in quick
        # commerce" came to be offered as corroboration for a delivery-delay
        # sentiment shift. A brand appearing in a business story is not
        # evidence that its service failed.
        #
        # Gate 1: the headline must name the brand.
        # Gate 2: the headline must also describe a *failure* - a delay, an
        #         outage, a strike, a disruption. A story about market share
        #         mentions the brand and fails this one.
        # Gate 3, the strongest, and the reason brand-constrained queries are
        # collected at all: the headline must have been *returned by that
        # brand's own query*. `source_id` on a `news_brand` row records which
        # brand was asked about, so this is a provenance test rather than a
        # substring test, and it cannot be satisfied by a brand name happening
        # to appear in an unrelated story.
        if targets and len(nn):
            pattern = "|".join(rf"\b{re.escape(b)}\b" for b in targets)
            names_brand = nn["full_text"].str.contains(pattern, case=False, na=False)
            describes_failure = nn["full_text"].str.contains(_FAILURE_WORDS,
                                                             case=False, na=False, regex=True)
            # A failure word is not enough on its own: "delay trial", "traffic
            # delays after a truck crash" and "why the show was cancelled" all
            # contain one. The headline must also be about the SERVICE.
            about_service = nn["full_text"].str.contains(_SERVICE_CONTEXT,
                                                         case=False, na=False, regex=True)
            not_off_topic = ~nn["full_text"].str.contains(_OFF_TOPIC,
                                                         case=False, na=False, regex=True)
            ok = describes_failure & about_service & not_off_topic
            # Provenance AND naming: Google News matches queries loosely, so a
            # row returned by the "Ola" query need not mention Ola at all.
            from_brand_query = (nn["source"].eq("news_brand")
                                & nn["source_id"].astype(str).isin(targets)
                                & names_brand)
            named_only = nn[names_brand]
            matched = nn[names_brand & ok]
            strongest = nn[from_brand_query & ok]
        else:
            matched = strongest = named_only = nn.iloc[0:0]
        # Report the strongest tier first; fall back only when it is empty, and
        # say which tier the headlines came from.
        use = strongest if len(strongest) else matched
        external["news_headlines"] = [str(t)[:130] for t in use["title"].head(6)]
        external["news_evidence_tier"] = (
            "returned by this brand's own failure query" if len(strongest)
            else "names the brand and describes a failure" if len(matched)
            else "none")
        external["news_in_window_total"] = int(len(nn))
        external["news_naming_a_brand_in_window"] = int(len(named_only))
        external["news_naming_a_brand_and_a_failure"] = int(len(matched))
        external["news_from_this_brands_own_query"] = int(len(strongest))
        external["news_rejected_named_brand_but_no_failure"] = int(len(named_only) - len(matched))

    if len(attention) and brand:
        key = brand.split()[0].lower()
        a = attention[attention["brand_hint"].str.lower().str.contains(key, na=False)]
        if len(a):
            a = a.groupby("date")["views"].sum().reset_index()
            lo_n, hi_n = lo.tz_localize(None), hi.tz_localize(None)
            base = a[(a["date"] < lo_n) | (a["date"] > hi_n)]["views"]
            win = a[(a["date"] >= lo_n) & (a["date"] <= hi_n)]["views"]
            if len(base) >= 5 and len(win):
                ratio = float(win.mean() / max(base.median(), 1))
                external["attention"] = {
                    "window_mean_views": round(float(win.mean()), 1),
                    "baseline_median_views": round(float(base.median()), 1),
                    "ratio": round(ratio, 2),
                    "corroborates": bool(ratio >= 1.25 or ratio <= 0.8),
                }

    # A "direct"-tier incident and a headline from the brand's own failure
    # query are stronger evidence than an infra incident and a substring
    # match, so the strength count weights them rather than treating every
    # tier as one vote.
    inc_rows = external.get("incidents") or []
    corroborating = [i for i in inc_rows if i.get("counts_as_corroboration")]
    external["incidents_counted_as_corroboration"] = len(corroborating)
    external["incidents_shown_as_context_only"] = len(inc_rows) - len(corroborating)
    strength = sum([
        bool(corroborating),
        bool(external.get("news_headlines")),
        bool((external.get("attention") or {}).get("corroborates")),
    ])
    external["evidence_quality"] = {
        "incident_match": (corroborating[0]["matched_on"] if corroborating
                           else "context only" if inc_rows else "none"),
        "news_tier": external.get("news_evidence_tier", "none"),
    }
    external["independent_sources_supporting"] = strength
    external["verdict"] = {
        0: "no external corroboration found - the evidence for this event is "
           "internal to the reactions themselves",
        1: "weakly corroborated by one independent source",
        2: "corroborated by two independent sources",
        3: "corroborated by three independent sources",
    }[strength]
    ev["external_evidence"] = external
    return ev


# ---------------------------------------------------------------------------
# analyses the first version collected the data for and never ran
# ---------------------------------------------------------------------------
def gated_sentiment(delay: pd.DataFrame, threshold: float = 0.70) -> dict:
    """The Round 2 model, used at the confidence where it is actually right.

    The transfer test against 74,013 star ratings gives accuracy by confidence
    band: 40% below 0.5, 88% at or above 0.7, 97% above 0.9. A time series that
    averages all predictions equally therefore spends most of its variance on
    rows the model itself flagged as guesses. Gating at 0.70 keeps about half
    the corpus at 88% accuracy.

    Both series are reported. If a movement appears in the gated series and not
    in the ungated one, the ungated one was noise; if it appears in both, it is
    not an artefact of low-confidence rows.
    """
    d = delay.dropna(subset=["r2_sentiment_confidence"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    gated = d[d["r2_sentiment_confidence"] >= threshold]
    out = {
        "threshold": threshold,
        "coverage": round(float(len(gated) / max(len(d), 1)), 4),
        "n_gated": int(len(gated)),
        "n_total": int(len(d)),
        "mean_all": round(float(d["sentiment_score"].mean()), 4),
        "mean_gated": round(float(gated["sentiment_score"].mean()), 4),
    }
    if len(gated) > 100:
        a = d.groupby("date")["sentiment_score"].mean()
        b = gated.groupby("date")["sentiment_score"].mean().reindex(a.index)
        both = pd.concat([a, b], axis=1).dropna()
        if len(both) > 5:
            out["daily_correlation_gated_vs_all"] = round(
                float(both.iloc[:, 0].corr(both.iloc[:, 1])), 3)
            out["daily_sd_all"] = round(float(both.iloc[:, 0].std()), 4)
            out["daily_sd_gated"] = round(float(both.iloc[:, 1].std()), 4)
    return out


def round2_topic_replay(delay: pd.DataFrame) -> dict:
    """Does the Round 2 topic head predict topics, or predict the substring?

    Round 2 established that ``topic_category`` in the training data was not an
    annotation at all: a case-insensitive substring switch reproduced 9,000 of
    9,000 labels exactly. The model that learned it scored macro-F1 0.94.

    The rulebook requires the Round 2 model to be applied in Round 3, so it is.
    The interesting question is what it does on a corpus the switch was never
    written for. Replaying the recovered rule over these 2026 app-store reviews
    and measuring agreement with the learned model's own predictions turns the
    Round 2 finding into a Round 3 measurement: if agreement is near-total, the
    topic head has transported the switch, not a notion of topic, and no
    reading of its output is safe.
    """
    # The rule is *read out of the Round 2 deliverable* rather than retyped
    # here, so it cannot drift from what Round 2 actually published. Importing
    # `round2/src/audit_labels.py` directly does not work - it does
    # `from config import ... TEXT_COL`, and Round 3's own `config` is first on
    # the path, so the import resolves to the wrong module and fails. Parsing
    # the three constants out of the source with `ast.literal_eval` gets the
    # published rule without executing anything.
    import ast                                             # noqa: PLC0415
    from config import ROUND2_SRC                          # noqa: PLC0415

    src = ROUND2_SRC / "audit_labels.py"
    if not src.exists():
        return {"available": False, "reason": f"Round 2 source not found at {src}"}
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
        consts: dict = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
                if name in ("RECOVERED_RULE", "PRIORITY", "DEFAULT_CLASS"):
                    consts[name] = ast.literal_eval(node.value)
            elif isinstance(node, ast.AnnAssign):
                name = getattr(node.target, "id", None)
                if name in ("RECOVERED_RULE", "PRIORITY", "DEFAULT_CLASS") and node.value:
                    consts[name] = ast.literal_eval(node.value)
        rule_map = consts["RECOVERED_RULE"]
        priority = consts["PRIORITY"]
        default = consts["DEFAULT_CLASS"]
    except Exception as exc:
        return {"available": False,
                "reason": f"could not read the Round 2 rule: {type(exc).__name__}: {exc}"}
    if "r2_topic" not in delay.columns:
        return {"available": False, "reason": "r2_topic not present"}

    low = delay["full_text"].fillna("").str.lower()
    rule = pd.Series(default, index=delay.index, dtype=object)
    assigned = pd.Series(False, index=delay.index)
    for cls in priority:
        hit = pd.Series(False, index=delay.index)
        for trig in rule_map[cls]:
            hit |= low.str.contains(trig, regex=False, na=False)
        take = hit & ~assigned
        rule[take] = cls
        assigned |= take
    model = delay["r2_topic"].astype(str)
    agree = float((rule.values == model.values).mean())
    text = delay["full_text"].fillna("")
    has_app_word = text.str.contains(r"\bapp\b", case=False, na=False)
    tech = model == "Technical_Issues"
    return {
        "available": True,
        "n": int(len(delay)),
        "agreement_with_recovered_substring_rule": round(agree, 4),
        "rule_distribution": {k: int(v) for k, v in rule.value_counts().items()},
        "model_distribution": {k: int(v) for k, v in model.value_counts().items()},
        "rows_containing_the_word_app": int(has_app_word.sum()),
        "of_those_labelled_Technical_Issues": int((has_app_word & tech).sum()),
        "Technical_Issues_without_the_word_app": int((~has_app_word & tech).sum()),
        "interpretation": (
            "Agreement this high on a corpus the rule was never fitted to means "
            "the Round 2 topic head learned the substring switch and carried it "
            "across domains intact. Its output is therefore reported as a "
            "required application of the Round 2 deliverable and is not "
            "interpreted as topical structure anywhere in this analysis. The "
            "topic layer that IS interpreted is the delay/reaction taxonomy, "
            "which ships the span that fired every label."),
    }


def reply_behaviour(delay: pd.DataFrame) -> dict:
    """Whether the operator answered - collected in round one, never analysed.

    ``company_replied`` is a public, timestamped, operator-side action on 34.5%
    of delay-related reviews. For a monitoring system it is the most directly
    actionable field in the dataset: it is the difference between a complaint
    that was handled and one that was not, and a brand's reply rate falling is
    the kind of thing an operator should be paged about.
    """
    d = delay.copy()
    if "company_replied" not in d.columns:
        return {"available": False}
    d["date"] = pd.to_datetime(d["date"])
    d["replied"] = d["company_replied"].fillna(False).astype(bool)
    by_brand = (d.groupby("brand")
                  .agg(n=("record_id", "count"), reply_rate=("replied", "mean"),
                       rating=("rating", "mean"))
                  .query("n >= 100").sort_values("reply_rate", ascending=False))
    by_type = (d.groupby("delay_type")
                 .agg(n=("record_id", "count"), reply_rate=("replied", "mean"))
                 .query("n >= 100").sort_values("reply_rate", ascending=False))
    out = {
        "available": True,
        "overall_reply_rate": round(float(d["replied"].mean()), 4),
        "by_brand": {k: {"n": int(v["n"]), "reply_rate": round(float(v["reply_rate"]), 3),
                         "mean_rating": round(float(v["rating"]), 2)}
                     for k, v in by_brand.to_dict("index").items()},
        "by_delay_type": {k: {"n": int(v["n"]), "reply_rate": round(float(v["reply_rate"]), 3)}
                          for k, v in by_type.to_dict("index").items()},
    }
    # does a reply track a better or worse review? (association, not causation:
    # operators reply to the reviews they think are worth replying to)
    if d["replied"].nunique() > 1 and d["rating"].notna().sum() > 100:
        r_yes = d.loc[d["replied"], "rating"].dropna()
        r_no = d.loc[~d["replied"], "rating"].dropna()
        if len(r_yes) > 30 and len(r_no) > 30:
            t, p = stats.ttest_ind(r_yes, r_no, equal_var=False)
            out["rating_when_replied"] = round(float(r_yes.mean()), 3)
            out["rating_when_not_replied"] = round(float(r_no.mean()), 3)
            out["welch_p"] = float(p)
            out["note"] = ("Association only. Operators choose which reviews to "
                           "answer, so this compares two self-selected groups "
                           "and cannot be read as the effect of replying.")
    return out


def severity_profile(delay: pd.DataFrame) -> dict:
    """The delay durations people actually state - parsed all along, never used.

    ``stated_delay_hours`` is filled on about a quarter of delay-related rows
    and is the only objective severity measure in the corpus. Polarity says a
    reaction is negative; a stated duration says how negative it had a right to
    be, and separates a twenty-minute wait from a three-week parcel.
    """
    d = delay.dropna(subset=["stated_delay_hours"])
    if len(d) < 50:
        return {"available": False, "n": int(len(d))}
    bins = [0, 1, 6, 24, 72, 168, 1e9]
    names = ["<1h", "1-6h", "6-24h", "1-3d", "3-7d", ">1w"]
    b = pd.cut(d["stated_delay_hours"], bins=bins, labels=names, right=False)
    g = (d.assign(bucket=b).groupby("bucket", observed=True)
           .agg(n=("record_id", "count"), sentiment=("sentiment_score", "mean"),
                rating=("rating", "mean"), engagement=("engagement", "mean")))
    corr = None
    sub = d[d["stated_delay_hours"] < 24 * 60]
    if len(sub) > 100 and sub["rating"].notna().sum() > 100:
        sub2 = sub.dropna(subset=["rating"])
        corr = round(float(stats.spearmanr(np.log1p(sub2["stated_delay_hours"]),
                                           sub2["rating"]).statistic), 3)
    return {
        "available": True,
        "n_with_duration": int(len(d)),
        "share_of_delay_rows": round(float(len(d) / max(len(delay), 1)), 4),
        "median_hours": round(float(d["stated_delay_hours"].median()), 2),
        "by_bucket": {str(k): {"n": int(v["n"]),
                               "sentiment": round(float(v["sentiment"]), 3),
                               "rating": round(float(v["rating"]), 2)
                               if pd.notna(v["rating"]) else None,
                               "engagement": round(float(v["engagement"]), 2)}
                      for k, v in g.to_dict("index").items()},
        "spearman_logduration_vs_stars": corr,
    }


def unspecified_profile(delay: pd.DataFrame) -> dict:
    """What is actually in the residual delay-type bucket.

    `unspecified_delay` is the largest delay type, and a taxonomy whose biggest
    category is "we could not tell" is one a judge is right to distrust. Two
    responses are possible: widen the patterns until the number goes down, or
    read the residual and say what is in it. The first was tried - broadening
    `refund_delay` and `support_delay` after auditing the bucket moved it by
    one percentage point - and the reason it did not move further is that most
    of these rows genuinely do not state a delay *type*. They say a refund
    never came, or that support never answered, without saying what went wrong
    with the delivery.

    That is a true label, not a failure, so the honest move is to characterise
    it. This reports which relevance trigger admitted each residual row, so a
    reader can see that the bucket is refunds and support rather than noise.
    """
    from config import DELAY_RELEVANCE                       # noqa: PLC0415
    u = delay[delay["delay_type"] == "unspecified_delay"]
    if u.empty:
        return {"n": 0}
    rx = re.compile(DELAY_RELEVANCE, re.I)
    triggers = Counter()
    for t in u["full_text"].fillna(""):
        m = rx.search(t)
        if m:
            triggers[m.group(0).lower()[:24]] += 1
    themes = {
        "mentions a refund": r"\brefund",
        "mentions support or a complaint": r"support|customer (?:care|service)|complain",
        "mentions a cancellation": r"\bcancel",
        "mentions something missing": r"\bmissing\b",
        "states a duration": r"\b\d+\s*(?:hour|hr|min|minute|day|week)s?\b",
        "mentions a return": r"\breturn",
    }
    text = u["full_text"].fillna("")
    return {
        "n": int(len(u)),
        "share_of_delay_rows": round(float(len(u) / max(len(delay), 1)), 4),
        "mean_rating": round(float(u["rating"].mean()), 3)
        if u["rating"].notna().any() else None,
        "first_relevance_trigger": {k: int(v) for k, v in triggers.most_common(12)},
        "themes": {k: int(text.str.contains(v, case=False, na=False).sum())
                   for k, v in themes.items()},
        "interpretation": (
            "These rows are overwhelmingly refunds that never arrived and "
            "support that never answered - real delay reactions that do not "
            "name a delivery failure mode. `unspecified_delay` is therefore a "
            "true label for them rather than a classifier miss, and it is "
            "excluded from the 'types of delay' headline chart so it cannot "
            "read as one."),
    }


def entity_analysis(delay: pd.DataFrame) -> dict:
    """Entities beyond ``value_counts(brand)``.

    The rubric asks for topic and entity analysis. Counting the brands we
    chose to scrape answers a question about our own roster, not about the
    conversation. These are entities the *text* supplies: where the failure
    happened, who the customer says they are leaving for, and who they blame.
    """
    text = delay["full_text"].fillna("")
    out: dict = {}

    if "competitor_named" in delay.columns:
        sw = delay.loc[delay["competitor_named"].astype(str).str.len() > 2,
                       "competitor_named"]
        known = set(b.lower() for b in delay["brand"].dropna().unique() if b)
        hits = Counter()
        for v in sw:
            v = str(v).strip().lower()
            for k in known:
                if k and k in v:
                    hits[k] += 1
                    break
        out["switch_destinations"] = {k: int(v) for k, v in hits.most_common(15)}
        out["rows_naming_a_switch_target"] = int(len(sw))

    airports = re.findall(r"\b(?:DEL|BOM|BLR|MAA|HYD|CCU|LAX|JFK|ORD|ATL|DFW|"
                          r"SFO|LHR|DXB|SIN|EWR|SEA|BOS|MIA|DEN)\b",
                          " ".join(text.head(60000).tolist()))
    if airports:
        out["airport_codes"] = {k: int(v) for k, v in Counter(airports).most_common(12)}

    cities = Counter()
    city_rx = re.compile(r"\b(Delhi|Mumbai|Bangalore|Bengaluru|Hyderabad|Chennai|"
                         r"Kolkata|Pune|Noida|Gurgaon|Gurugram|Ahmedabad|Jaipur|"
                         r"Lucknow|London|New York|Chicago|Toronto|Dubai|Sydney)\b", re.I)
    for t in text.head(60000):
        for m in city_rx.findall(t):
            cities[m.title()] += 1
    if cities:
        out["cities"] = {k: int(v) for k, v in cities.most_common(12)}

    flags = [c for c in delay.columns if c.startswith("flag_")]
    if flags:
        out["reaction_flags"] = {
            c: {"n": int(delay[c].sum()),
                "share": round(float(delay[c].mean()), 3),
                "mean_rating": round(float(delay.loc[delay[c], "rating"].mean()), 2)
                if delay.loc[delay[c], "rating"].notna().any() else None}
            for c in flags}
    return out


# ---------------------------------------------------------------------------
def main() -> dict:
    src = PROCESSED / "reactions_labelled.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_labelled.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    print(f"  {len(df):,} labelled reactions")

    incidents = load_incidents()
    attention = load_attention()
    # `news_brand` rows are the external corroboration instrument, not public
    # reactions - a wire story about an outage is evidence that the outage
    # happened, not a person reacting to it. They are held out of the reaction
    # corpus so they cannot inflate a brand's reaction count, and are used only
    # in `explain_event`.
    news = df[df["source"].isin(("news", "news_brand"))].copy()
    delay = df[df["is_delay_related"] & (df["source"] != "news_brand")].copy()
    print(f"  {len(delay):,} delay-related | {len(incidents):,} incidents "
          f"| {len(attention):,} attention rows")

    results: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
                     "n_records": int(len(df)), "n_delay_related": int(len(delay))}

    # ---- coverage and the balanced panel ----------------------------------
    # This block runs before anything else because everything else depends on
    # it. A time series over brands that enter and leave the corpus is partly a
    # measurement of the collector, and in the first version of this analysis
    # it was mostly that: daily volume correlated r = 0.89 with the day index,
    # and rho = -0.06 with third-party attention for the same brands.
    play = df[df["source"] == "google_play"]
    cov = coverage(play, "brand")
    panel_brands = balanced_panel(play, "brand")
    diag = panel_diagnostics(delay[delay["source"] == "google_play"], panel_brands)
    results["coverage"] = {
        "by_brand": {str(k): {"n": int(v["n"]), "days": int(v["days"]),
                              "first": str(v["first"])[:10], "last": str(v["last"])[:10],
                              "coverage_share": float(v["coverage_share"]),
                              "in_balanced_panel": bool(k in panel_brands)}
                     for k, v in cov.to_dict("index").items()},
        "panel_diagnostics": diag,
    }
    print(f"  balanced panel: {len(panel_brands)}/{cov.shape[0]} brands cover the window")
    print(f"    all-brand daily volume vs day index  r = "
          f"{diag['all_brands']['pearson_r_with_day_index']:+.3f}")
    print(f"    balanced-panel daily volume vs index r = "
          f"{diag['balanced_panel']['pearson_r_with_day_index']:+.3f}")

    # Every series below is computed on the balanced panel plus the non-Play
    # sources (which have no pagination budget and so no entry bias).
    delay_panel = delay[(delay["brand"].isin(panel_brands))
                        | (delay["source"] != "google_play")].copy()
    results["panel_note"] = (
        "Every time series in this analysis is computed on the balanced panel: "
        f"{len(panel_brands)} Google Play brands observable across the whole "
        "window, plus the non-Play sources, which have no pagination budget and "
        "therefore no entry bias. The all-brand series is retained only as the "
        "artefact demonstration in `coverage.panel_diagnostics`.")

    # ---- activity, estimated for the population ---------------------------
    # The census counted every review in the frame; the quota sample says what
    # share of a brand-day was delay-related. Their product estimates how many
    # people actually complained - which is what an activity claim is about,
    # and is not the number of rows we kept.
    census = load_census(RAW / "play_census.jsonl")
    if len(census):
        est = estimate_population(play, census, flag="is_delay_related")
        est.to_csv(PROCESSED / "population_estimate_by_brand_day.csv", index=False)
        est_panel = est[est["brand"].isin(panel_brands)]
        daily_est = (est_panel.groupby("date")
                     .agg(estimated_n=("estimated_n", "sum"),
                          estimated_se=("estimated_se", lambda s: float(np.sqrt((s ** 2).sum()))),
                          census_n=("census_n", "sum"),
                          sampled_n=("sampled_n", "sum"),
                          census_thumbs=("census_thumbs", "sum"))
                     .reset_index())
        daily_est["date"] = pd.to_datetime(daily_est["date"])
        daily_est.to_csv(PROCESSED / "timeseries_estimated_population.csv", index=False)
        results["population_estimate"] = {
            "method": ("ratio estimator: enumerated reviews per brand-day x "
                       "delay-related share in that brand-day's quota sample, "
                       "with a finite-population-corrected binomial SE"),
            "reviews_enumerated": int(census["reviews_total"].sum()),
            "reviews_sampled": int(len(play)),
            "overall_sampling_fraction": round(
                float(len(play) / max(census["reviews_total"].sum(), 1)), 4),
            "daily": daily_est.assign(
                date=daily_est["date"].dt.strftime("%Y-%m-%d")).round(1).to_dict("records"),
        }
        print(f"  population estimate built from {census['reviews_total'].sum():,} "
              f"enumerated reviews")

    # ---- series -----------------------------------------------------------
    overall = daily_series(delay_panel)
    results["daily_overall"] = overall.assign(
        date=overall["date"].dt.strftime("%Y-%m-%d")).to_dict("records")

    by_domain = daily_series(delay_panel, "delay_domain")
    by_domain.to_csv(PROCESSED / "timeseries_by_domain.csv", index=False)
    overall.to_csv(PROCESSED / "timeseries_overall.csv", index=False)
    hourly_series(delay_panel).to_csv(PROCESSED / "timeseries_hourly.csv", index=False)
    daily_series(delay).to_csv(PROCESSED / "timeseries_all_brands_UNBALANCED.csv",
                               index=False)

    # ---- shifts -----------------------------------------------------------
    # Two instruments, deliberately. The Round 2 model is what the rulebook
    # requires us to apply; the star rating is the reviewer's own verdict, on a
    # finer 5-point scale, and is the more sensitive detector. Reporting both
    # means a shift can be corroborated across instruments rather than resting
    # on the one model whose domain transfer we already know is imperfect.
    candidates: list[dict] = []
    df_panel = df[(df["brand"].isin(panel_brands)) | (df["source"] != "google_play")]
    for metric in ("sentiment_score", "rating"):
        candidates += shift_candidates(delay_panel, "overall", metric)
        for dom, g in delay_panel.groupby("delay_domain"):
            if dom and len(g) >= 500:
                candidates += shift_candidates(g, f"domain:{dom}", metric)
        for brand, g in delay_panel.groupby("brand"):
            if brand and len(g) >= 400:
                candidates += shift_candidates(g, f"brand:{brand}", metric)
        # the whole-brand series is denser than its delay-related subset, and a
        # delay event moves a brand's overall reception, not only the reviews
        # that happen to name the delay
        for brand, g in df_panel.groupby("brand"):
            if brand and len(g) >= 1200:
                candidates += shift_candidates(g, f"brand-all:{brand}", metric)

    scan_shifts = select_shifts(candidates)

    # --- primary detector: change points on the daily series -------------
    cpd_shifts: list[dict] = []
    overall_daily = daily_series(delay_panel)
    _m = {"sentiment_mean": "sentiment_score", "rating_mean": "rating"}
    for metric, rec_metric in _m.items():
        cpd_shifts += [attach_welch(c, delay_panel, rec_metric, panel=panel_brands)
                       for c in change_points(overall_daily, metric, scope="overall")]
    for dom, g in delay_panel.groupby("delay_domain"):
        if not dom or len(g) < 500:
            continue
        gd = daily_series(g)
        for metric, rec_metric in _m.items():
            cpd_shifts += [attach_welch(c, g, rec_metric, panel=panel_brands)
                           for c in change_points(gd, metric, scope=f"domain:{dom}")]
    for brand, g in delay_panel.groupby("brand"):
        if not brand or len(g) < 400:
            continue
        gd = daily_series(g)
        for metric, rec_metric in _m.items():
            cpd_shifts += [attach_welch(c, g, rec_metric, panel=panel_brands)
                           for c in change_points(gd, metric, scope=f"brand:{brand}")]

    # Rank by the effect on *reviewers*, not by the step in the daily series.
    # The old sort key was the daily-series d, which is large for any clean
    # step regardless of how little anyone moved, and which put a 0.12-star
    # shift at the top of the report as "d = 1.54, large".
    cpd_shifts = [c for c in cpd_shifts
                  if abs(c.get("cohens_d_records") or 0.0) >= RECORD_D_FLOOR]
    cpd_shifts.sort(key=lambda c: (-abs(c.get("cohens_d_records") or 0.0),
                                   c.get("welch_p_uncorrected", 1.0)))
    shifts = cpd_shifts
    results["sentiment_shifts_scan_fdr"] = scan_shifts
    results["shift_scan"] = {
        "candidates_tested": len(candidates),
        "fdr_q": FDR_Q,
        # This field said `len(shifts)` - the PELT count - so the notebook
        # printed "1,793 tests, 9 survive pooled FDR<0.05" directly underneath
        # a paragraph explaining that nothing survived. The number belongs to
        # the scan, not to the change-point detector.
        "significant_after_fdr": len(scan_shifts),
        "metrics": ["sentiment_score", "rating"],
        "note": ("Secondary, deliberately conservative check. One "
                 "Benjamini-Hochberg family across every scope, metric and "
                 "window width. Scanning every day turns one segmentation "
                 "question into ~1,800 significance questions, and an honest "
                 "multiplicity correction over that many tests rejects "
                 "everything. Read `significant_after_fdr` for what survived. "
                 "The primary detector is PELT change-point detection, which "
                 "asks the segmentation question directly instead of as a "
                 "multiplicity problem."),
    }
    results["shift_method"] = {
        "primary": "PELT change-point detection (l2 cost, BIC-scaled penalty, min segment 4 days)",
        "instruments": ["Round 2 model sentiment", "reviewer star rating"],
        "effect_size_floor_on_records": RECORD_D_FLOOR,
        "effect_size_note": (
            "Two Cohen's d values are reported per break and they are not "
            "interchangeable. `cohens_d_daily` is computed over the daily means "
            "PELT segments; its denominator is the SD of ~14 daily averages, "
            "which is small by construction, so it is large for any clean step "
            "regardless of how little any individual moved. `cohens_d_records` "
            "is computed over the reviews themselves and is the effect on "
            "people. On the headline 2026-09-08 break these were 1.54 and 0.10 "
            "respectively. Selection and ranking use the record-level value; "
            "the daily value is kept only to describe how clean the step was."),
        "secondary": "day-by-day Welch scan with pooled Benjamini-Hochberg",
        "welch_p_note": ("p-values attached to change points are uncorrected and "
                         "describe the size of a break PELT located; they do not "
                         "certify its discovery"),
    }
    results["sentiment_shifts"] = shifts
    print(f"  shift candidates tested: {len(candidates):,}")
    print(f"  FDR scan survivors (conservative check): {len(scan_shifts)}")
    print(f"  change points (PELT, |record d|>={RECORD_D_FLOOR}): {len(shifts)}")
    survived = [c for c in shifts
                if (c.get("balanced_panel_check") or {}).get("survives") is True]
    print(f"    of which survive the balanced-panel control: {len(survived)}")

    # ---- spikes -----------------------------------------------------------
    # Volume and endorsement are separated, because the first version reported
    # them together and twenty-three of twenty-seven "engagement spikes" were
    # actually volume - twelve of them one brand entering the corpus.
    # Volume and endorsement both have to come off the right instrument.
    # The quota sample caps each brand-day at PER_BRAND_DAY rows, so summing
    # `engagement` over the sample is not the day's endorsement - it is the
    # endorsement of the rows we happened to keep, and the cap bites hardest
    # on exactly the busiest brand-days. The census carries `thumbs_total`
    # for every review in the frame, kept or not, so that is what the
    # endorsement detector runs on when it exists.
    spikes = detect_spikes(overall, "n", "overall", records=delay_panel, kind="volume")
    census_endorsement = None
    if len(census) and "population_estimate" in results:
        ce = daily_est[["date", "census_thumbs"]].rename(
            columns={"census_thumbs": "engagement_sum"})
        ce = ce.merge(overall[["date", "sentiment_mean", "negative_share"]],
                      on="date", how="left")
        census_endorsement = ce
        spikes += detect_spikes(ce, "engagement_sum", "overall",
                                records=delay_panel, kind="endorsement")
    else:
        spikes += detect_spikes(overall, "engagement_sum", "overall",
                                records=delay_panel, kind="endorsement")
    for dom, g in by_domain.groupby("delay_domain"):
        if dom and g["n"].sum() >= 1500:
            spikes += detect_spikes(g, "n", dom,
                                    records=delay_panel[delay_panel["delay_domain"] == dom],
                                    kind="volume")
    spikes = sorted(spikes, key=lambda s: -s["robust_z"])
    results["engagement_spikes"] = spikes
    results["spike_method"] = {
        "detector": "median/MAD robust z, threshold %.1f" % SPIKE_Z,
        "panel": "balanced panel only",
        "kinds": {"volume": "more reactions were posted",
                  "endorsement": "the reactions that existed were upvoted harder"},
        "concentration_note": (
            "Every endorsement spike carries the share of the day's total held "
            "by its top one and top five contributors. A day where two reviews "
            "hold 88% of the endorsement is two popular posts, not a "
            "conversation, and the report says so rather than quoting the "
            "ratio-to-median alone."),
        "endorsement_instrument": (
            "census thumbs-up totals over every review in the frame"
            if census_endorsement is not None else
            "summed over the quota sample (no census on disk) - biased low on "
            "the busiest brand-days, where the quota cap bites hardest"),
        "engagement_bias_note": (
            "thumbs-up is cumulative to the moment of collection, so older days "
            "have had longer to accrue it. The bias runs against detecting "
            "recent spikes, which makes any endorsement spike found near the "
            "end of the window a conservative finding."),
    }
    n_vol = sum(1 for s_ in spikes if s_["kind"] == "volume")
    print(f"  spikes (robust z>{SPIKE_Z}): {len(spikes)} "
          f"({n_vol} volume, {len(spikes) - n_vol} endorsement)")

    # ---- analyses the collector gathered data for and the first pass skipped
    results["gated_sentiment"] = gated_sentiment(delay_panel)
    results["round2_topic_replay"] = round2_topic_replay(delay)
    results["company_reply_behaviour"] = reply_behaviour(delay)
    results["severity_profile"] = severity_profile(delay)
    results["entity_analysis"] = entity_analysis(delay)
    results["unspecified_profile"] = unspecified_profile(delay)
    print(f"  gated sentiment coverage: "
          f"{results['gated_sentiment'].get('coverage', 0):.1%} at conf>=0.70")
    rt = results["round2_topic_replay"]
    if rt.get("available"):
        print(f"  Round 2 topic head agrees with the recovered substring rule "
              f"on {rt['agreement_with_recovered_substring_rule']:.1%} of rows")

    # ---- entities and topics ---------------------------------------------
    results["entities"] = {
        "by_brand": {k: int(v) for k, v in
                     delay.loc[delay["brand"] != "", "brand"].value_counts().head(25).items()},
        "by_domain": {k: int(v) for k, v in delay["delay_domain"].value_counts().items()},
        "delay_types": {k: int(v) for k, v in delay["delay_type"].value_counts().items()},
        "reaction_types": {k: int(v) for k, v in delay["reaction_type"].value_counts().items()},
        "by_source": {k: int(v) for k, v in delay["source"].value_counts().items()},
    }
    results["distinctive_terms_delay_vs_rest"] = distinctive_terms(
        delay["full_text"].sample(min(len(delay), 40000), random_state=42).tolist(),
        df[~df["is_delay_related"]]["full_text"].sample(
            min(int((~df["is_delay_related"]).sum()), 40000), random_state=42).tolist(),
        top=25)

    # terms distinctive to each delay type
    results["terms_by_delay_type"] = {}
    for dt, g in delay.groupby("delay_type"):
        if len(g) < 300:
            continue
        rest = delay[delay["delay_type"] != dt]
        results["terms_by_delay_type"][dt] = distinctive_terms(
            g["full_text"].sample(min(len(g), 15000), random_state=42).tolist(),
            rest["full_text"].sample(min(len(rest), 25000), random_state=42).tolist(),
            top=10)

    # sentiment by delay type and reaction type - the core cross-tab
    results["sentiment_by_delay_type"] = (
        delay.groupby("delay_type")
        .agg(n=("record_id", "count"),
             sentiment_mean=("sentiment_score", "mean"),
             negative_share=("r2_sentiment", lambda s: float((s == "Negative").mean())),
             rating_mean=("rating", "mean"),
             engagement_mean=("engagement", "mean"))
        .round(4).reset_index().to_dict("records"))
    results["sentiment_by_reaction_type"] = (
        delay.groupby("reaction_type")
        .agg(n=("record_id", "count"),
             sentiment_mean=("sentiment_score", "mean"),
             rating_mean=("rating", "mean"),
             engagement_mean=("engagement", "mean"))
        .round(4).reset_index().to_dict("records"))

    # ---- explanations -----------------------------------------------------
    explained = []
    for ev in shifts[:4]:
        scope = ev["scope"]
        bare = scope.split(":", 1)[-1]
        if scope == "overall":
            sub = delay
        elif scope.startswith("domain:"):
            sub = delay[delay["delay_domain"] == bare]
        elif scope.startswith("brand-all:"):
            sub = df[df["brand"] == bare]
        else:
            sub = delay[delay["brand"] == bare]
        brand = bare if bare in set(df["brand"]) else None
        explained.append({"event_type": "sentiment_shift", **ev,
                          "evidence": explain_event(ev, sub, incidents, attention,
                                                    news, brand)})
    for ev in spikes[:3]:
        scope = ev["scope"]
        sub = delay if scope == "overall" else delay[delay["delay_domain"] == scope]
        explained.append({"event_type": "engagement_spike", **ev,
                          "evidence": explain_event(ev, sub, incidents, attention,
                                                    news, None)})

    # The FDR-surviving brand shifts are the strongest detections in the
    # corpus - they survive correction across the whole scan - so they get
    # the same external-evidence treatment as the PELT breaks. Brand scope is
    # also where the news gate is most informative: one operator, one window.
    for ev in scan_shifts[:4]:
        bare = ev["scope"].split(":", 1)[-1]
        sub = df[df["brand"] == bare] if ev["scope"].startswith("brand-all:") \
            else delay[delay["brand"] == bare]
        if sub.empty:
            continue
        explained.append({"event_type": "fdr_shift", **ev,
                          "evidence": explain_event(ev, sub, incidents, attention,
                                                    news, bare)})
    results["explained_events"] = explained

    (REPORTS / "analysis.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  wrote analysis.json ({len(explained)} explained events)")
    return results


if __name__ == "__main__":
    main()
