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

MIN_SIDE = 60          # records needed on each side of a candidate change point
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


def detect_shifts(df: pd.DataFrame, label: str = "overall",
                  window_days: int = WINDOW_DAYS, min_side: int = MIN_SIDE) -> list[dict]:
    """Welch t-test at every candidate day, FDR-corrected across the scan."""
    d = df.dropna(subset=["sentiment_score"]).copy()
    if d.empty:
        return []
    d["day"] = pd.to_datetime(d["date"])
    days = sorted(d["day"].unique())
    if len(days) < 2 * window_days + 2:
        return []

    cands = []
    for cut in days[window_days:-window_days]:
        lo = cut - pd.Timedelta(days=window_days)
        pre = d.loc[(d["day"] >= lo) & (d["day"] < cut), "sentiment_score"].to_numpy()
        post = d.loc[(d["day"] >= cut) & (d["day"] < cut + pd.Timedelta(days=window_days)),
                     "sentiment_score"].to_numpy()
        if len(pre) < min_side or len(post) < min_side:
            continue
        t, p = stats.ttest_ind(post, pre, equal_var=False)
        if not np.isfinite(p):
            continue
        sd = math.sqrt((pre.var(ddof=1) + post.var(ddof=1)) / 2) or 1e-9
        cands.append({
            "scope": label,
            "date": cut.strftime("%Y-%m-%d"),
            "mean_before": round(float(pre.mean()), 4),
            "mean_after": round(float(post.mean()), 4),
            "delta": round(float(post.mean() - pre.mean()), 4),
            "cohens_d": round(float((post.mean() - pre.mean()) / sd), 4),
            "n_before": int(len(pre)), "n_after": int(len(post)),
            "t_stat": round(float(t), 3), "p_value": float(p),
            "direction": "deterioration" if post.mean() < pre.mean() else "improvement",
        })
    if not cands:
        return []

    keep = _bh([c["p_value"] for c in cands])
    sig = [c | {"significant_fdr": bool(k)} for c, k in zip(cands, keep) if k]

    # Neighbouring days re-detect the same shift; keep the strongest in each run.
    sig.sort(key=lambda c: c["date"])
    merged: list[dict] = []
    for c in sig:
        if merged and (pd.Timestamp(c["date"]) - pd.Timestamp(merged[-1]["date"])).days <= 2:
            if abs(c["cohens_d"]) > abs(merged[-1]["cohens_d"]):
                merged[-1] = c
        else:
            merged.append(c)
    return sorted(merged, key=lambda c: -abs(c["cohens_d"]))


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


def explain_event(event: dict, df: pd.DataFrame, incidents: pd.DataFrame,
                  attention: pd.DataFrame, news: pd.DataFrame,
                  brand: str | None = None, pad_days: int = 2) -> dict:
    """Assemble every independent piece of evidence around one event."""
    day = pd.Timestamp(event["date"], tz="UTC")
    lo, hi = day - pd.Timedelta(days=pad_days), day + pd.Timedelta(days=pad_days)
    created = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")

    focus = df[(created >= lo) & (created <= hi)]
    background = df[(created < lo) | (created > hi)]
    ev: dict = {"window": [lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")],
                "n_in_window": int(len(focus))}

    # 1. distinctive vocabulary
    ev["distinctive_terms"] = distinctive_terms(
        focus["full_text"].tolist(), background["full_text"].sample(
            min(len(background), 40000), random_state=42).tolist())

    # 2. what kinds of delay and reaction dominate the window
    if len(focus):
        ev["delay_type_mix"] = {k: int(v) for k, v in
                                focus["delay_type"].value_counts().head(6).items()}
        ev["reaction_type_mix"] = {k: int(v) for k, v in
                                   focus["reaction_type"].value_counts().head(6).items()}
        ev["top_brands"] = {k: int(v) for k, v in
                            focus.loc[focus["brand"] != "", "brand"]
                            .value_counts().head(6).items()}

    # 3. documented incidents
    if len(incidents):
        near = incidents[(incidents["started"] >= lo) & (incidents["started"] <= hi)]
        ev["incidents"] = [
            {"entity": r.entity, "kind": r.incident_kind,
             "started": r.started.strftime("%Y-%m-%d %H:%M"),
             "title": str(r.title)[:120], "reason": str(r.impact_reason)[:80]}
            for r in near.head(8).itertuples()
        ]
        ev["n_incidents_in_window"] = int(len(near))

    # 4. news coverage
    if len(news):
        ncreated = pd.to_datetime(news["created_utc"], utc=True, errors="coerce")
        nn = news[(ncreated >= lo) & (ncreated <= hi)]
        if brand:
            hit = nn["full_text"].str.contains(re.escape(brand), case=False, na=False)
            nn = nn[hit] if hit.any() else nn
        ev["news_headlines"] = [str(t)[:130] for t in nn["title"].head(6)]
        ev["n_news_in_window"] = int(len(nn))

    # 5. independent attention signal
    if len(attention) and brand:
        a = attention[attention["brand_hint"].str.lower().str.contains(
            brand.split()[0].lower(), na=False)]
        if len(a):
            a = a.groupby("date")["views"].sum().reset_index()
            base = a[(a["date"] < lo.tz_localize(None)) | (a["date"] > hi.tz_localize(None))]["views"]
            win = a[(a["date"] >= lo.tz_localize(None)) & (a["date"] <= hi.tz_localize(None))]["views"]
            if len(base) >= 5 and len(win):
                ev["attention"] = {
                    "window_mean_views": round(float(win.mean()), 1),
                    "baseline_median_views": round(float(base.median()), 1),
                    "ratio": round(float(win.mean() / max(base.median(), 1)), 2),
                }

    # 6. app releases visible in the window
    if "app_version" in focus.columns:
        vers = focus.loc[focus["app_version"].astype(str).str.len() > 2, "app_version"]
        ev["app_versions_in_window"] = {k: int(v) for k, v in
                                        vers.value_counts().head(5).items()}
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
    shifts = detect_shifts(delay, "overall")
    for dom, g in delay.groupby("delay_domain"):
        if dom and len(g) >= 1500:
            shifts += detect_shifts(g, dom)
    for brand, g in delay.groupby("brand"):
        if brand and len(g) >= 2500:
            shifts += detect_shifts(g, brand)
    shifts = sorted(shifts, key=lambda s: -abs(s["cohens_d"]))
    results["sentiment_shifts"] = shifts
    print(f"  sentiment shifts (FDR<{FDR_Q}): {len(shifts)}")

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
        sub = delay if scope == "overall" else delay[
            (delay["delay_domain"] == scope) | (delay["brand"] == scope)]
        brand = scope if scope in set(delay["brand"]) else None
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
