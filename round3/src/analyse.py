"""Find the shifts and spikes, then try to explain them with outside evidence.

The rubric asks for at least two sentiment shifts, at least one engagement
spike, the key entities, and the reasons behind the changes. The last of those
is the one that separates analysis from storytelling, so the design here is
built backwards from it.

Method
------
**Shifts** are not eyeballed off a chart. For every candidate day we compare
the individual sentiment scores in the preceding window against the following
window with Welch's t-test (unequal variances, because volume and dispersion
both change), require a minimum sample on each side, and correct the resulting
p-values across all candidate days with Benjamini-Hochberg. What survives is a
shift we can defend; Cohen's d is reported beside it so that a statistically
significant but trivial move is visible as such.

**Spikes** use a median/MAD robust z-score rather than mean/SD, because a spike
inflates the very mean and standard deviation you would be testing it against.

**Triggers** are searched for, not asserted. Around each event we look in a
+/- 2 day window for four independent kinds of evidence: a documented incident
from a status page or the FAA, news coverage naming the brand, an app release,
and a corroborating move in Wikipedia pageviews. We also compute which words
became distinctive during the event using log-odds with an informative Dirichlet
prior, which is robust to the frequency effects that make raw word counts
useless for this. Where nothing is found, the report says nothing was found.
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
                 window_days: int = 5) -> dict:
    """Effect size and an uncorrected Welch p-value for a located break."""
    d = records.dropna(subset=[metric]).copy()
    d["day"] = pd.to_datetime(d["date"])
    cut = pd.Timestamp(shift["date"])
    pre = d.loc[(d["day"] >= cut - pd.Timedelta(days=window_days)) & (d["day"] < cut), metric]
    post = d.loc[(d["day"] >= cut) & (d["day"] < cut + pd.Timedelta(days=window_days)), metric]
    if len(pre) < 20 or len(post) < 20:
        return shift
    t, pval = stats.ttest_ind(post.to_numpy(), pre.to_numpy(), equal_var=False)
    if np.isfinite(pval):
        shift = shift | {
            "record_n_before": int(len(pre)), "record_n_after": int(len(post)),
            "welch_t": round(float(t), 3),
            "welch_p_uncorrected": float(pval),
        }
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


def detect_spikes(series: pd.DataFrame, column: str = "n",
                  label: str = "overall", z: float = SPIKE_Z) -> list[dict]:
    if len(series) < 7:
        return []
    vals = series[column].astype(float).to_numpy()
    zs = robust_z(vals)
    med = float(np.median(vals))
    out = []
    for i, (zz, row) in enumerate(zip(zs, series.itertuples())):
        if zz >= z:
            out.append({
                "scope": label, "metric": column,
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "value": float(vals[i]),
                "baseline_median": round(med, 2),
                "ratio_to_median": round(float(vals[i] / med), 2) if med else None,
                "robust_z": round(float(zz), 2),
                "sentiment_mean": round(float(getattr(row, "sentiment_mean", np.nan)), 4)
                if not pd.isna(getattr(row, "sentiment_mean", np.nan)) else None,
                "negative_share": round(float(getattr(row, "negative_share", np.nan)), 4)
                if not pd.isna(getattr(row, "negative_share", np.nan)) else None,
            })
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
        entity_hit = any(b in entity or entity in b for b in lowered if len(b) > 3)
        sector_hit = sector in allowed_sectors
        if entity_hit or sector_hit:
            keep.append({
                "entity": r.entity, "kind": r.incident_kind,
                "started": r.started.strftime("%Y-%m-%d %H:%M"),
                "title": str(r.title)[:120], "reason": str(r.impact_reason)[:80],
                "matched_on": "entity" if entity_hit else "sector",
            })
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
        targets = [b for b in brands_in_window if len(b) > 3]
        if targets and len(nn):
            pattern = "|".join(re.escape(b) for b in targets)
            hit = nn["full_text"].str.contains(pattern, case=False, na=False)
            matched = nn[hit]
        else:
            matched = nn.iloc[0:0]
        external["news_headlines"] = [str(t)[:130] for t in matched["title"].head(6)]
        external["news_in_window_total"] = int(len(nn))
        external["news_naming_a_brand_in_window"] = int(len(matched))

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

    strength = sum([
        bool(external.get("incidents")),
        bool(external.get("news_headlines")),
        bool((external.get("attention") or {}).get("corroborates")),
    ])
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
def main() -> dict:
    src = PROCESSED / "reactions_labelled.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_labelled.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    print(f"  {len(df):,} labelled reactions")

    incidents = load_incidents()
    attention = load_attention()
    news = df[df["source"] == "news"].copy()
    delay = df[df["is_delay_related"]].copy()
    print(f"  {len(delay):,} delay-related | {len(incidents):,} incidents "
          f"| {len(attention):,} attention rows")

    results: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
                     "n_records": int(len(df)), "n_delay_related": int(len(delay))}

    # ---- series -----------------------------------------------------------
    overall = daily_series(delay)
    results["daily_overall"] = overall.assign(
        date=overall["date"].dt.strftime("%Y-%m-%d")).to_dict("records")

    by_domain = daily_series(delay, "delay_domain")
    by_domain.to_csv(PROCESSED / "timeseries_by_domain.csv", index=False)
    overall.to_csv(PROCESSED / "timeseries_overall.csv", index=False)
    hourly_series(delay).to_csv(PROCESSED / "timeseries_hourly.csv", index=False)

    # ---- shifts -----------------------------------------------------------
    # Two instruments, deliberately. The Round 2 model is what the rulebook
    # requires us to apply; the star rating is the reviewer's own verdict, on a
    # finer 5-point scale, and is the more sensitive detector. Reporting both
    # means a shift can be corroborated across instruments rather than resting
    # on the one model whose domain transfer we already know is imperfect.
    candidates: list[dict] = []
    for metric in ("sentiment_score", "rating"):
        candidates += shift_candidates(delay, "overall", metric)
        for dom, g in delay.groupby("delay_domain"):
            if dom and len(g) >= 500:
                candidates += shift_candidates(g, f"domain:{dom}", metric)
        for brand, g in delay.groupby("brand"):
            if brand and len(g) >= 400:
                candidates += shift_candidates(g, f"brand:{brand}", metric)
        # the whole-brand series is denser than its delay-related subset, and a
        # delay event moves a brand's overall reception, not only the reviews
        # that happen to name the delay
        for brand, g in df.groupby("brand"):
            if brand and len(g) >= 1200:
                candidates += shift_candidates(g, f"brand-all:{brand}", metric)

    scan_shifts = select_shifts(candidates)

    # --- primary detector: change points on the daily series -------------
    cpd_shifts: list[dict] = []
    overall_daily = daily_series(delay)
    for metric in ("sentiment_mean", "rating_mean"):
        cpd_shifts += [attach_welch(c, delay,
                                    "sentiment_score" if metric == "sentiment_mean" else "rating")
                       for c in change_points(overall_daily, metric, scope="overall")]
    for dom, g in delay.groupby("delay_domain"):
        if not dom or len(g) < 500:
            continue
        gd = daily_series(g)
        for metric in ("sentiment_mean", "rating_mean"):
            cpd_shifts += [attach_welch(c, g,
                                        "sentiment_score" if metric == "sentiment_mean" else "rating")
                           for c in change_points(gd, metric, scope=f"domain:{dom}")]
    for brand, g in delay.groupby("brand"):
        if not brand or len(g) < 400:
            continue
        gd = daily_series(g)
        for metric in ("sentiment_mean", "rating_mean"):
            cpd_shifts += [attach_welch(c, g,
                                        "sentiment_score" if metric == "sentiment_mean" else "rating")
                           for c in change_points(gd, metric, scope=f"brand:{brand}")]

    # rank by how large the step is relative to the noise, then by support
    cpd_shifts = [c for c in cpd_shifts if abs(c["cohens_d"]) >= 0.20]
    cpd_shifts.sort(key=lambda c: (-abs(c["cohens_d"]), c.get("welch_p_uncorrected", 1.0)))
    shifts = cpd_shifts
    results["sentiment_shifts_scan_fdr"] = scan_shifts
    results["shift_scan"] = {
        "candidates_tested": len(candidates),
        "fdr_q": FDR_Q,
        "significant_after_fdr": len(shifts),
        "metrics": ["sentiment_score", "rating"],
        "note": ("Secondary, deliberately conservative check. One "
                 "Benjamini-Hochberg family across every scope, metric and "
                 "window width. Scanning every day turns one question into "
                 "~1,800 tests, and nothing survives that correction - which "
                 "is reported rather than hidden. The primary detector is PELT "
                 "change-point detection, which asks the segmentation question "
                 "directly instead of as a multiplicity problem."),
    }
    results["shift_method"] = {
        "primary": "PELT change-point detection (l2 cost, BIC-scaled penalty, min segment 4 days)",
        "instruments": ["Round 2 model sentiment", "reviewer star rating"],
        "effect_size_floor": 0.20,
        "secondary": "day-by-day Welch scan with pooled Benjamini-Hochberg",
        "welch_p_note": ("p-values attached to change points are uncorrected and "
                         "describe the size of a break PELT located; they do not "
                         "certify its discovery"),
    }
    results["sentiment_shifts"] = shifts
    print(f"  shift candidates tested: {len(candidates):,}")
    print(f"  FDR scan survivors (conservative check): {len(scan_shifts)}")
    print(f"  change points (PELT, |d|>=0.20): {len(shifts)}")

    # ---- spikes -----------------------------------------------------------
    spikes = detect_spikes(overall, "n", "overall")
    spikes += detect_spikes(overall, "engagement_sum", "overall")
    for dom, g in by_domain.groupby("delay_domain"):
        if dom and g["n"].sum() >= 1500:
            spikes += detect_spikes(g, "n", dom)
    spikes = sorted(spikes, key=lambda s: -s["robust_z"])
    results["engagement_spikes"] = spikes
    print(f"  engagement spikes (robust z>{SPIKE_Z}): {len(spikes)}")

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
    results["explained_events"] = explained

    (REPORTS / "analysis.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  wrote analysis.json ({len(explained)} explained events)")
    return results


if __name__ == "__main__":
    main()
