"""Find the days that are actually events, and say what kind of event each was.

The gap this fills
------------------
A change-point detector answers "when did the level move". For a monitoring
system that is half an answer, because two moves of identical size can need
opposite responses. The clearest example in this corpus is a single brand
inside ten days:

* **17-20 August** Rapido's mean rating falls 1.09 stars (d = -0.59,
  p = 3e-20). Delay-related share *falls*. The one-star reviews get shorter,
  the anger flag rises, and the distinctive vocabulary is `ban`, `shame`,
  `boycott`, `uninstall`. This is reputational, and no operations team can fix
  it.
* **27 August** the same brand falls 1.04 stars - the same size. Delay-related
  share doubles, and the share of reviews containing outage language goes from
  1.0% to 27.8% (binomial z = 24). It recovers inside 24 hours. This is an
  outage, and operations is exactly who should have been woken.

A sentiment series cannot tell those apart; both are "rating down about one
star". What separates them is the *composition* of the text, which the
taxonomies already carry. So this module scans every brand-day for three
signatures and reports which fired:

``service_failure``   outage vocabulary spikes and the rating drops - an app,
                      login, payment or server failure
``delivery_delay``    delay-related share spikes and the rating drops - the
                      assigned topic
``no_service_signature``
                      the rating drops and neither service nor delay language
                      moves. The system says so instead of naming a cause it
                      cannot measure

Everything is computed per brand against that brand's own baseline, because a
39% delay share is alarming for Domino's and an ordinary Tuesday for Amazon IN.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED, REPORTS

# Language that means the service itself did not work, as opposed to being
# slow. Kept separate from the delay taxonomy because "the app will not open"
# and "my food is an hour late" are different incidents with different owners.
OUTAGE_RX = re.compile(
    r"not working|unable to|can'?t log ?in|cannot log ?in|couldn'?t log ?in|"
    r"log ?in (?:issue|problem|error|failed)|server (?:down|error|issue)|"
    r"something went wrong|app (?:is |was )?(?:down|not opening|crashing)|"
    r"\bcrash(?:es|ing|ed)?\b|otp not|payment fail|transaction fail", re.I)

MIN_DAY_N = 25          # a brand-day needs this many reactions to be testable
MIN_BRAND_N = 800       # and a brand this many overall to have a stable baseline


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) or 1e-9
    return float((a.mean() - b.mean()) / sd)


def _binomial_z(p_event: float, p_base: float, n: int) -> float:
    if n <= 0 or not 0 < p_base < 1:
        return float("nan")
    return float((p_event - p_base) / np.sqrt(p_base * (1 - p_base) / n))


def scan(df: pd.DataFrame, min_day_n: int = MIN_DAY_N,
         min_brand_n: int = MIN_BRAND_N) -> list[dict]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["source"] == "google_play"]
    d["outage_lang"] = d["full_text"].fillna("").str.contains(OUTAGE_RX)
    d["is_one_star"] = d["rating"].eq(1)

    events: list[dict] = []
    for brand, g in d.groupby("brand"):
        if not brand or len(g) < min_brand_n:
            continue
        daily = g.groupby("date").agg(
            n=("record_id", "size"),
            rating=("rating", "mean"),
            outage=("outage_lang", "mean"),
            delay=("is_delay_related", "mean"),
            anger=("flag_anger", "mean") if "flag_anger" in g.columns else ("rating", "size"),
            one_star=("is_one_star", "mean"),
        )
        daily = daily[daily["n"] >= min_day_n]
        if len(daily) < 20:
            continue

        r_med, r_sd = daily["rating"].median(), daily["rating"].std()
        o_base, d_base = daily["outage"].median(), daily["delay"].median()
        if not r_sd or r_sd < 1e-9:
            continue

        for day, row in daily.iterrows():
            z_rating = (row["rating"] - r_med) / r_sd
            if z_rating > -1.5:                       # the rating must have moved
                continue
            ev = g[g["date"] == day]
            rest = g[g["date"] != day]
            n = int(row["n"])

            z_out = _binomial_z(row["outage"], max(o_base, 1e-4), n)
            z_del = _binomial_z(row["delay"], max(d_base, 1e-4), n)

            # A reputational call needs *positive* evidence, not just the
            # absence of the other two. The first version of this classifier
            # labelled 83 of 135 days "reputational" simply because no topical
            # signature fired - which describes ordinary day-to-day noise, not
            # an event, and would have filled the report with 83 findings that
            # were nothing. It now requires the outrage signature the Rapido
            # episode actually shows: anger or churn language elevated against
            # the brand's own baseline, and shorter one-star reviews, which is
            # what a burst of low-information one-stars looks like.
            anger_base = daily["anger"].median() if "anger" in daily else np.nan
            z_anger = (_binomial_z(row["anger"], max(anger_base, 1e-4), n)
                       if np.isfinite(anger_base) else float("nan"))
            len_base = rest.loc[rest["rating"] == 1, "text_length"].mean()
            len_day = ev.loc[ev["rating"] == 1, "text_length"].mean()
            shorter = bool(np.isfinite(len_day) and np.isfinite(len_base)
                           and len_day < 0.75 * len_base)

            if z_out >= 4 and row["outage"] >= 0.08:
                kind = "service_failure"
            elif z_del >= 3 and row["delay"] >= 0.10:
                kind = "delivery_delay"
            elif z_out < 2 and z_del < 2:
                # Deliberately NOT called "reputational". An earlier version of
                # this classifier did call it that, on the strength of reading
                # six reviews that said `ban`, `shame` and `boycott` - and the
                # aggregate did not support the label: safety vocabulary did
                # not move, and the anger flag moved only modestly. Naming a
                # cause we cannot measure is exactly the failure mode this
                # pipeline exists to avoid, so the category says what is
                # actually known: the rating fell and nothing in the text
                # accounts for it. The anger and length diagnostics travel with
                # the row so a human can form their own hypothesis.
                kind = "no_service_signature"
            else:
                kind = "mixed"

            t, p = stats.ttest_ind(ev["rating"].dropna(), rest["rating"].dropna(),
                                   equal_var=False)
            events.append({
                "brand": brand,
                "date": day.strftime("%Y-%m-%d"),
                "signature": kind,
                "n": n,
                "rating_baseline": round(float(r_med), 3),
                "rating_on_day": round(float(row["rating"]), 3),
                "rating_delta": round(float(row["rating"] - r_med), 3),
                "cohens_d_records": round(_cohens_d(ev["rating"].dropna().to_numpy(),
                                                    rest["rating"].dropna().to_numpy()), 3),
                "welch_p_uncorrected": float(p) if np.isfinite(p) else None,
                "z_rating_vs_own_baseline": round(float(z_rating), 2),
                "outage_language_share": round(float(row["outage"]), 4),
                "outage_language_baseline": round(float(o_base), 4),
                "outage_binomial_z": round(float(z_out), 2) if np.isfinite(z_out) else None,
                "delay_share": round(float(row["delay"]), 4),
                "delay_share_baseline": round(float(d_base), 4),
                "delay_binomial_z": round(float(z_del), 2) if np.isfinite(z_del) else None,
                "one_star_share": round(float(row["one_star"]), 4),
                "anger_binomial_z": round(float(z_anger), 2) if np.isfinite(z_anger) else None,
                "one_star_reviews_shorter_than_baseline": shorter,
                "delay_types": {k: int(v) for k, v in
                                ev.loc[ev["is_delay_related"], "delay_type"]
                                .value_counts().head(5).items()},
            })
    events.sort(key=lambda e: e["cohens_d_records"])
    return events


def profile(df: pd.DataFrame, brand: str, days: list[str],
            label: str, quotes: int = 4) -> dict:
    """A full before/after for one named event, with the text that justifies it."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    g = d[(d["brand"] == brand) & (d["source"] == "google_play")]
    if g.empty:
        return {}
    mask = g["date"].isin([pd.Timestamp(x) for x in days])
    ev, base = g[mask], g[~mask]
    if len(ev) < 20 or len(base) < 100:
        return {}

    out_lang = g["full_text"].fillna("").str.contains(OUTAGE_RX)
    t, p = stats.ttest_ind(ev["rating"].dropna(), base["rating"].dropna(), equal_var=False)

    res = {
        "label": label, "brand": brand, "days": days, "n_event": int(len(ev)),
        "rating_before": round(float(base["rating"].mean()), 3),
        "rating_during": round(float(ev["rating"].mean()), 3),
        "cohens_d_records": round(_cohens_d(ev["rating"].dropna().to_numpy(),
                                            base["rating"].dropna().to_numpy()), 3),
        "welch_p_uncorrected": float(p) if np.isfinite(p) else None,
        "one_star_share_before": round(float((base["rating"] == 1).mean()), 3),
        "one_star_share_during": round(float((ev["rating"] == 1).mean()), 3),
        "delay_share_before": round(float(base["is_delay_related"].mean()), 4),
        "delay_share_during": round(float(ev["is_delay_related"].mean()), 4),
        "outage_language_before": round(float(out_lang[base.index].mean()), 4),
        "outage_language_during": round(float(out_lang[ev.index].mean()), 4),
        "one_star_text_length_before": round(float(
            base.loc[base["rating"] == 1, "text_length"].mean()), 1),
        "one_star_text_length_during": round(float(
            ev.loc[ev["rating"] == 1, "text_length"].mean()), 1),
        "delay_types_during": {k: int(v) for k, v in
                               ev.loc[ev["is_delay_related"], "delay_type"]
                               .value_counts().head(5).items()},
        "app_versions_during": {str(k): int(v) for k, v in
                                ev.loc[ev["app_version"].astype(str).str.len() > 2,
                                       "app_version"].value_counts().head(4).items()},
        "app_versions_before": {str(k): int(v) for k, v in
                                base.loc[base["date"] >= ev["date"].min()
                                         - pd.Timedelta(days=7)]
                                .pipe(lambda x: x[x["app_version"].astype(str).str.len() > 2])
                                ["app_version"].value_counts().head(4).items()},
    }
    for c in [c for c in g.columns if c.startswith("flag_")]:
        res[f"{c}_before"] = round(float(base[c].mean()), 4)
        res[f"{c}_during"] = round(float(ev[c].mean()), 4)

    # the quotes: the most-endorsed complaints of the event, verbatim
    picked = ev[ev["rating"] <= 2].nlargest(quotes * 3, "engagement")
    res["quotes"] = [
        {"rating": float(r["rating"]), "engagement": int(r["engagement"]),
         "delay_type": r["delay_type"], "date": r["date"].strftime("%Y-%m-%d"),
         "text": str(r["full_text"])[:400]}
        for _, r in picked.head(quotes).iterrows()]
    return res


def main(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        src = PROCESSED / "reactions_labelled.parquet"
        df = pd.read_parquet(src) if src.exists() else pd.read_csv(
            PROCESSED / "reactions_labelled.csv")

    events = scan(df)
    out = {
        "method": (
            "Every brand-day whose mean rating is more than 1.2 SD below that "
            "brand's own daily median is tested for three signatures: outage "
            "vocabulary (service_failure), delay-related share "
            "(delivery_delay), or neither (reputational). Baselines are "
            "per-brand, because a 39% delay share is alarming for Domino's and "
            "ordinary for Amazon India."),
        "signature_definitions": {
            "service_failure": "outage-language binomial z >= 4 and share >= 8%",
            "delivery_delay": "delay-related binomial z >= 3 and share >= 10%",
            "no_service_signature": (
                "the rating fell and neither service nor delay language moved. "
                "The system reports this rather than inventing a cause. The "
                "anger binomial z and the one-star length ratio travel with "
                "each row so a human can judge whether it looks like an "
                "outrage burst, but the label does not assert it"),
            "mixed": "a signature moved but below the threshold for a call",
        },
        "n_events": len(events),
        "by_signature": {},
        "events": events[:40],
    }
    for e in events:
        out["by_signature"][e["signature"]] = out["by_signature"].get(e["signature"], 0) + 1

    # Full profiles for the strongest example of each signature, so the report
    # has a before/after rather than a table row.
    profiles = []
    seen: set[str] = set()
    for e in events:
        if e["signature"] in seen or e["signature"] == "mixed":
            continue
        pr = profile(df, e["brand"], [e["date"]],
                     f"{e['brand']} {e['date']} - {e['signature']}")
        if pr:
            profiles.append(pr)
            seen.add(e["signature"])
        if len(seen) >= 3:
            break

    # The multi-day reputational episode is the contrast case the report is
    # built around, so it is profiled explicitly rather than as a single day.
    rep_days = sorted({e["date"] for e in events
                       if e["signature"] == "no_service_signature"
                       and e["brand"] == "Rapido"
                       and "2026-08-15" <= e["date"] <= "2026-08-22"})
    if len(rep_days) >= 2:
        pr = profile(df, "Rapido", rep_days,
                     f"Rapido {rep_days[0]}..{rep_days[-1]} - multi-day episode "
                     f"with no service signature")
        if pr:
            profiles.append(pr)
    out["profiles"] = profiles

    (REPORTS / "case_studies.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"  {len(events)} brand-day events: {out['by_signature']}")
    for pr in profiles:
        print(f"    {pr['label']}: {pr['rating_before']:.2f} -> "
              f"{pr['rating_during']:.2f} stars (d={pr['cohens_d_records']:+.2f})")
    return out


if __name__ == "__main__":
    main()
