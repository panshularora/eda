"""Generate and execute the Round 3 analysis notebook.

Written by code rather than by hand so it cannot drift from the modules it
calls, and executed before it is saved so every output a judge reads was
produced by the cell above it.
"""
from __future__ import annotations

import sys

import nbformat as nbf
from nbclient import NotebookClient

from config import NOTEBOOKS, ROUND3

OUT = NOTEBOOKS / "Round3_Analysis_Team_SE7EN.ipynb"
MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell


def cells() -> list:
    return [
        MD("""# Reaction to a Major Delivery or Service Delay
## Real-time monitoring for the Social Engine — Round 3

**Team SE7EN** · Tanmay Singh · Panshul Arora
Data Vortex A'26 · Round 3

---

Round 2 restored the Engine's ability to read meaning. It still could not watch
a conversation *move*. This notebook does that: it collects live public reaction
to delivery and service failures, applies the Round 2 model to it, and then asks
the only question that matters for monitoring — **when did the mood change, and
what changed it?**

**Contents**

1. What was collected, and from where
2. The dataset
3. Types of delay, types of reaction
4. Does the Round 2 model still work here? *(validated against star ratings)*
5. Activity over time — engagement spikes
6. Sentiment over time — shifts
7. Trigger explanations
8. Entities and vocabulary
9. What this says about the Social Engine"""),

        CODE("""import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path("round3")
sys.path.insert(0, str((ROOT / "src").resolve()))

import pandas as pd, numpy as np
from config import PROCESSED, REPORTS, TOPIC

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 70)
print("topic:", TOPIC)"""),

        MD("""## 1. What was collected, and from where

Three layers, because "sentiment moved" and "sentiment moved *because of X*" are
very different claims and only the second one is useful to an operator.

| layer | purpose | sources |
|---|---|---|
| **L1 reaction** | what people said, with engagement and a timestamp | Google Play reviews (44 apps), Reddit, Mastodon, Google News, Hacker News |
| **L2 trigger** | documented incidents with start times | FAA airport delay register, 10 public status pages |
| **L3 attention** | an independent volume signal we did not generate | Wikipedia pageviews |

The audit below is what actually answered on the day — including what refused."""),

        CODE("""audit = json.loads((PROCESSED / "collection_audit.json").read_text())
print("collection window :", audit["window_days"], "days requested")
print("started           :", audit["started_utc"])
print("elapsed           :", audit["elapsed_seconds"], "s")
print()
for k, v in audit["records_by_source"].items():
    print(f"  {k:22s} {v:>8,}")
print()
http = audit["http"]
print(f"live HTTP requests {http['total_live_requests']:,} | "
      f"cache hits {http['total_cache_hits']:,} | failures {http['total_failures']:,}")"""),

        MD("""Not every source answered, and that is reported rather than hidden. Reddit
rate-limits unauthenticated readers hard and refused roughly half our requests
even at a 6-second interval; Apple's review RSS is deprecated and returns an
empty feed for every app we tried. Google Play carries the dataset, and the
consequences of that concentration are addressed in section 9."""),

        MD("## 2. The dataset"),

        CODE("""src = PROCESSED / "reactions_labelled.parquet"
if not src.exists(): src = PROCESSED / "reactions_labelled.csv"
df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)

print(f"{len(df):,} reactions x {df.shape[1]} columns")
print(f"window: {df.created_utc.min():%Y-%m-%d %H:%M} .. {df.created_utc.max():%Y-%m-%d %H:%M} UTC")
print(f"        {(df.created_utc.max()-df.created_utc.min()).days} days")
print(f"brands: {df.brand.nunique()}   authors: {df.author_pseudonym.nunique():,}")
df[["source","brand","delay_domain","created_utc","rating","engagement",
    "delay_type","reaction_type","r2_sentiment"]].head(5)"""),

        CODE("""quality = json.loads((REPORTS / "data_quality.json").read_text())
print("by source:      ", quality["by_source"])
print("by domain:      ", quality["by_delay_domain"])
print(f"delay-related:   {quality['delay_related_share']:.1%}")
print(f"thin text:       {quality['thin_text_share']:.1%}  (kept, flagged, excluded from text mining)")
print(f"duplicate ids:   {quality['duplicate_record_ids']}")"""),

        MD("""Every author handle was replaced with a salted hash at the moment of
collection — the dataset studies *what was said and when*, never *who said it*.
No raw identifier is written to disk at any point in the pipeline."""),

        MD("""## 3. Types of delay, types of reaction

The assignment asks about reaction to delay, so the dataset carries two
taxonomies applied as ordered, first-match rules. Rules rather than clustering
because a judge can read the pattern that fired and disagree with it — every row
stores the literal matched span in `delay_type_evidence`."""),

        CODE("""delay = df[df.is_delay_related].copy()
print(f"{len(delay):,} delay-related reactions\\n")
mix = (delay.groupby("delay_type")
       .agg(n=("record_id","count"),
            sentiment=("sentiment_score","mean"),
            stars=("rating","mean"),
            engagement=("engagement","mean"))
       .sort_values("n", ascending=False).round(3))
mix["share"] = (mix.n/mix.n.sum()).round(4)
mix"""),

        CODE("""# the evidence behind the labels - these are auditable, not magic
for dt in delay.delay_type.value_counts().head(4).index:
    g = delay[delay.delay_type==dt]
    print(f"\\n{dt}  (n={len(g):,}, mean sentiment {g.sentiment_score.mean():+.3f})")
    for _, r in g.nlargest(2, "engagement").iterrows():
        print(f"   [{r.brand}] matched {r.delay_type_evidence!r}")
        print(f"      {r.full_text[:150]}")"""),

        CODE("""react = (delay.groupby("reaction_type")
         .agg(n=("record_id","count"),
              sentiment=("sentiment_score","mean"),
              stars=("rating","mean"),
              engagement=("engagement","mean"))
         .sort_values("n", ascending=False).round(3))
react"""),

        MD("""## 4. Does the Round 2 model still work here?

The rulebook says to apply the Round 2 model. It does not say to check whether
the model survives the move from 2015-era tweets to 2026 app-store reviews — but
that is exactly the check an engineer owes the operator, because every sentiment
curve later in this notebook inherits the answer.

We can check, because **Google Play hands us an answer key**. Every review
carries a 1–5 star rating chosen by the same person who wrote the text, about
the same experience, at the same moment. That is an independent sentiment label
on ~100k rows — from a completely different distribution than the Round 2
training data."""),

        CODE("""t = json.loads((REPORTS / "round2_transfer.json").read_text())
print(f"Round 2 model vs {t['n']:,} independent star labels")
print(f"  accuracy {t['accuracy']:.4f}   macro-F1 {t['macro_f1']:.4f}   kappa {t['cohen_kappa']:.4f}")
if "polarity_only" in t:
    po = t["polarity_only"]
    print(f"  dropping Neutral: accuracy {po['accuracy']:.4f} on {po['n']:,} rows (kappa {po['cohen_kappa']:.4f})")
print()
print(t["classification_report"])"""),

        CODE("""print("transfer quality by delay domain (macro-F1 vs stars):")
for dom, v in t["per_domain"].items():
    print(f"  {dom:16s} n={v['n']:>7,}  macroF1={v['macro_f1']:.3f}  "
          f"recall on 1-2 star={v['negative_recall']:.3f}")
print()
print("accuracy by model confidence:")
for b in t["confidence_bands"]:
    print(f"  {b['band']}  n={b['n']:>7,}  accuracy={b['accuracy']:.3f}")"""),

        MD("""This is the licence to believe the rest of the notebook, and it is a
qualified one. Read the confidence bands: accuracy rises with the model's own
confidence, which means the confidence is *informative* and can be used as a
gate. Read the per-domain table: transfer is not uniform, so a per-domain claim
is weaker than an aggregate one. Both facts are carried forward rather than
glossed over."""),

        MD("""## 5. Activity over time — engagement spikes

A spike is detected with a **median/MAD robust z-score**, not mean/SD. A spike
inflates the very mean and standard deviation you would test it against, so the
conventional z-score systematically under-detects exactly the events we are
looking for."""),

        CODE("""analysis = json.loads((REPORTS / "analysis.json").read_text())
spikes = analysis["engagement_spikes"]
print(f"{len(spikes)} spikes at robust z > 3.5\\n")
sp = pd.DataFrame(spikes)[["scope","metric","date","value","baseline_median",
                           "ratio_to_median","robust_z","negative_share"]]
sp.head(12)"""),

        CODE("""from IPython.display import Image, display
display(Image(str(REPORTS / "figures" / "fig01_timeline.png")))"""),

        MD("""## 6. Sentiment over time — shifts

Shifts are **not eyeballed off the chart**. At every candidate day we compare
the individual sentiment scores in the 3 days before against the 3 days after
with Welch's t-test, require at least 60 records on each side, and correct every
p-value across the whole 45-day scan with Benjamini–Hochberg.

That correction matters: scanning 45 days at α=0.05 expects false positives by
construction. What survives is a shift we can defend."""),

        CODE("""shifts = analysis["sentiment_shifts"]
print(f"{len(shifts)} shifts significant after FDR correction\\n")
sh = pd.DataFrame(shifts)[["scope","date","direction","mean_before","mean_after",
                           "delta","cohens_d","n_before","n_after","p_value"]]
sh["p_value"] = sh["p_value"].map(lambda p: f"{p:.2e}")
sh.head(12)"""),

        MD("""## 7. Trigger explanations

For each event we search a ±2-day window for four *independent* kinds of
evidence and report what is found — including when nothing is found. The
vocabulary test uses log-odds with an informative Dirichlet prior (Monroe et
al.), which is comparable across words of very different frequency in a way that
raw counts and simple ratios are not."""),

        CODE("""for ev in analysis["explained_events"][:4]:
    e = ev["evidence"]
    head = (f"{ev['event_type'].replace('_',' ').upper()}  {ev['scope']}  {ev['date']}")
    print("=" * 78); print(head)
    if ev["event_type"] == "sentiment_shift":
        print(f"  {ev['direction']}: {ev['mean_before']:+.3f} -> {ev['mean_after']:+.3f}"
              f"  (d={ev['cohens_d']:+.2f}, n={ev['n_before']}/{ev['n_after']})")
    else:
        print(f"  {ev['value']:.0f} vs median {ev['baseline_median']:.0f}"
              f"  ({ev['ratio_to_median']}x, z={ev['robust_z']})")
    print(f"  window {e['window'][0]} .. {e['window'][1]}   n={e['n_in_window']:,}")
    if e.get("distinctive_terms"):
        print("  words that became distinctive:",
              ", ".join(f"{d['term']}(z={d['z']:.0f})" for d in e["distinctive_terms"][:8]))
    if e.get("delay_type_mix"):   print("  delay mix    :", e["delay_type_mix"])
    if e.get("reaction_type_mix"):print("  reaction mix :", e["reaction_type_mix"])
    if e.get("top_brands"):       print("  brands       :", e["top_brands"])
    if e.get("incidents"):
        print(f"  documented incidents in window ({e.get('n_incidents_in_window',0)}):")
        for i in e["incidents"][:3]:
            print(f"     - {i['started']}  {i['entity']}: {i['title'][:70]}")
    if e.get("news_headlines"):
        print(f"  news in window ({e.get('n_news_in_window',0)}):")
        for h in e["news_headlines"][:3]: print(f"     - {h[:90]}")
    if e.get("attention"):
        a = e["attention"]
        print(f"  Wikipedia attention: {a['window_mean_views']:,.0f} vs baseline "
              f"{a['baseline_median_views']:,.0f}  ({a['ratio']}x)")
    if e.get("app_versions_in_window"):
        print("  app versions :", e["app_versions_in_window"])
    print()"""),

        MD("""## 8. Entities and vocabulary"""),

        CODE("""ent = analysis["entities"]
print("top brands by delay-related volume:")
for b, n in list(ent["by_brand"].items())[:15]:
    print(f"  {b:14s} {n:>7,}")
print()
print("the vocabulary of a delay (log-odds z vs everything else):")
for d in analysis["distinctive_terms_delay_vs_rest"][:15]:
    print(f"  {d['term']:16s} z={d['z']:>7.1f}  ({d['focus_count']:,} vs {d['background_count']:,})")"""),

        CODE("""for name in ["fig02_delay_types.png", "fig07_brand_delay_heatmap.png",
             "fig04_domains.png", "fig05_round2_transfer.png",
             "fig10_attention_corroboration.png"]:
    p = REPORTS / "figures" / name
    if p.exists(): display(Image(str(p)))"""),

        MD("""## 9. What this says about the Social Engine

**The monitoring layer works.** Shifts and spikes are detected statistically,
not by eye, and each one is checked against evidence collected independently of
the text that produced it.

**Three honest limits**, because a monitoring system that oversells itself is
worse than none:

1. **Source concentration.** Google Play supplies the overwhelming majority of
   rows. Reddit rate-limited us and Apple's review feed is dead, so the
   cross-platform triangulation is thinner than designed. A finding that holds
   only in Play reviews is a finding about app-store reviewers.
2. **Reviews are not a live firehose.** People write a review hours or days
   after the delay, so the reaction curve lags the incident. That lag is a
   property of the medium, not of the detector, and it caps how "real-time" this
   can honestly claim to be.
3. **The sentiment model is transferring across domains.** Section 4 measures
   the cost of that rather than assuming it away, and the per-domain table shows
   the cost is uneven.

**What we would do next:** treat the star rating as a training signal and
fine-tune the Round 2 model on in-domain data — section 4 shows there are ~100k
independently labelled examples sitting in the dataset already, which is more
in-domain supervision than Round 2 ever had.

---

**Team SE7EN** — Tanmay Singh · Panshul Arora"""),
    ]


def main():
    nb = nbf.v4.new_notebook(cells=cells())
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"  executing {len(nb.cells)} cells ...", flush=True)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbf.write(nb, OUT)
    print(f"  wrote {OUT.relative_to(ROUND3.parent)} "
          f"({OUT.stat().st_size / 1e6:.2f} MB)")
    return OUT


if __name__ == "__main__":
    main()
