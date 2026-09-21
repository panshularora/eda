"""Figures for the Round 3 analysis.

Charts here have one job each: show the thing the text claims, with the
evidence attached. Shift and spike markers are drawn from the detector output
rather than placed by hand, so a figure cannot disagree with the statistics
that produced it.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, PROCESSED, REPORTS

INK, MUTED, GRID = "#1b1f24", "#6b7280", "#e5e7eb"
ACCENT, WARN, GOOD = "#2563eb", "#ea7317", "#0f9d58"
SERIES = ["#2563eb", "#ea7317", "#0f9d58", "#b23a48", "#7c4dff", "#00897b", "#c2185b"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    return ax


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"    {name}")
    return name


# ---------------------------------------------------------------------------
def fig_timeline(daily: pd.DataFrame, shifts, spikes) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    ax = axes[0]
    ax.fill_between(d["date"], d["n"], color=ACCENT, alpha=0.25)
    ax.plot(d["date"], d["n"], color=ACCENT, lw=1.4)
    ax.set_ylabel("delay-related reactions / day")
    ax.set_title("Activity: delay-related reactions per day, balanced panel")
    # Labels alternate between two rows. The first version stacked every
    # annotation at the same height, so markers on adjacent days overlapped
    # into an unreadable smear.
    picked_spikes = [x for x in spikes
                     if x["scope"] == "overall" and x["metric"] == "n"][:5]
    for i, s in enumerate(picked_spikes):
        x = pd.Timestamp(s["date"])
        ax.axvline(x, color=WARN, ls="--", lw=1.1, alpha=0.9)
        ax.annotate(f"{s['ratio_to_median']:.1f}x median\nz={s['robust_z']:.1f}",
                    xy=(x, d["n"].max()), xytext=(0, -20 - 24 * (i % 2)),
                    textcoords="offset points", ha="center", fontsize=6.6,
                    color=WARN, fontweight="bold")
    despine(ax)

    ax = axes[1]
    ax.plot(d["date"], d["sentiment_mean"], color=INK, lw=1.5, label="mean sentiment")
    ax.axhline(0, color=GRID, lw=1)
    ax.fill_between(d["date"], d["sentiment_mean"], 0,
                    where=d["sentiment_mean"] < 0, color="#b23a48", alpha=0.18)
    ax2 = ax.twinx()
    ax2.plot(d["date"], d["negative_share"], color=WARN, lw=1.1, ls=":",
             label="negative share")
    ax2.set_ylabel("negative share", color=WARN)
    ax2.tick_params(axis="y", colors=WARN)
    ax2.grid(False)
    # The label quotes the RECORD-level d. The first version quoted the
    # daily-series d, which reads "large effect" for a 0.12-star move.
    picked = [x for x in shifts if x["scope"] == "overall"][:6]
    for i, s in enumerate(picked):
        x = pd.Timestamp(s["date"])
        col = "#b23a48" if s["direction"] == "deterioration" else GOOD
        dr = s.get("cohens_d_records")
        ax.axvline(x, color=col, ls="-", lw=1.3, alpha=0.75)
        arrow = "v" if s["direction"] == "deterioration" else "^"
        label = f"{arrow} d={dr:+.2f}" if dr is not None else str(s["date"])
        bp = (s.get("balanced_panel_check") or {}).get("survives")
        if bp is False:
            label += "\n(fails panel control)"
        ax.annotate(label, xy=(x, ax.get_ylim()[1]),
                    xytext=(0, -12 - 20 * (i % 2)),
                    textcoords="offset points", ha="center", fontsize=6.6,
                    color=col, fontweight="bold")
    ax.set_ylabel("mean sentiment  (-1 neg .. +1 pos)")
    ax.set_title("Sentiment on the balanced panel, with change points marked "
                 "(d = effect on reviewers)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    despine(ax)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.suptitle("Reaction to a Major Delivery or Service Delay — collection window",
                 fontsize=11, fontweight="bold", y=1.0)
    return save(fig, "fig01_timeline.png")


def fig_delay_types(analysis) -> str:
    rows = pd.DataFrame(analysis["sentiment_by_delay_type"]).sort_values("n", ascending=True)
    rows = rows[rows["delay_type"] != ""]
    # `unspecified_delay` is excluded from the chart of delay *types* because
    # it is not one - it is the rows that describe a delay reaction without
    # naming a failure mode, overwhelmingly refunds that never came and support
    # that never answered. Leaving it in makes the largest bar in a chart
    # titled "how the delay was described" mean "it wasn't", which reads as a
    # broken taxonomy rather than as an honest residual. Its size and contents
    # are reported in the text and in `analysis.json:unspecified_profile`.
    n_unspec = int(rows.loc[rows["delay_type"] == "unspecified_delay", "n"].sum())
    rows = rows[rows["delay_type"] != "unspecified_delay"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.barh(rows["delay_type"], rows["n"], color=ACCENT, alpha=0.9)
    for i, v in enumerate(rows["n"]):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=6.8, color=INK)
    ax.set_xlabel("reactions")
    ax.set_title("How the delay was described")
    despine(ax)

    ax = axes[1]
    colors = ["#b23a48" if v < 0 else GOOD for v in rows["sentiment_mean"]]
    ax.barh(rows["delay_type"], rows["sentiment_mean"], color=colors, alpha=0.9)
    ax.axvline(0, color=GRID, lw=1)
    for i, (v, r) in enumerate(zip(rows["sentiment_mean"], rows["rating_mean"])):
        lbl = f" {v:+.2f}" + (f"  ({r:.1f}★)" if pd.notna(r) else "")
        ax.text(v, i, lbl, va="center", fontsize=6.8, color=INK)
    ax.set_xlabel("mean sentiment")
    ax.set_title("Which delays anger people most")
    ax.set_yticklabels([])
    despine(ax)
    fig.suptitle("Different delays draw different reactions", fontsize=10.5,
                 fontweight="bold", y=1.03)
    fig.text(0.5, -0.06,
             f"A further {n_unspec:,} delay-related reactions name no failure "
             f"mode (mostly refunds that never came and support that never "
             f"answered) and are excluded from this chart of delay types.",
             ha="center", fontsize=7, color=MUTED)
    return save(fig, "fig02_delay_types.png")


def fig_reactions(analysis) -> str:
    rows = pd.DataFrame(analysis["sentiment_by_reaction_type"]).sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh(rows["reaction_type"], rows["n"], color=SERIES[4], alpha=0.9)
    for b, v, s in zip(bars, rows["n"], rows["sentiment_mean"]):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {int(v):,}   sent {s:+.2f}", va="center", fontsize=6.8, color=INK)
    ax.set_xlabel("reactions")
    ax.set_title("How people reacted")
    despine(ax)
    return save(fig, "fig03_reaction_types.png")


def fig_domains(daily_domain: pd.DataFrame) -> str:
    d = daily_domain.copy()
    d["date"] = pd.to_datetime(d["date"])
    doms = (d.groupby("delay_domain")["n"].sum().sort_values(ascending=False).index[:6])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for i, dom in enumerate(doms):
        g = d[d["delay_domain"] == dom].sort_values("date")
        axes[0].plot(g["date"], g["n"], lw=1.3, color=SERIES[i % len(SERIES)], label=dom)
        roll = g["sentiment_mean"].rolling(3, min_periods=1).mean()
        axes[1].plot(g["date"], roll, lw=1.3, color=SERIES[i % len(SERIES)], label=dom)
    axes[0].set_title("Volume by delay domain")
    axes[0].set_ylabel("reactions / day")
    axes[1].set_title("Sentiment by delay domain (3-day mean)")
    axes[1].set_ylabel("mean sentiment")
    axes[1].axhline(0, color=GRID, lw=1)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        despine(ax)
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.autofmt_xdate(rotation=0, ha="center")
    return save(fig, "fig04_domains.png")


def fig_transfer(transfer) -> str:
    if not transfer.get("n"):
        return ""
    labels = transfer["labels"]
    cm = np.array(transfer["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, mat, title, fmt in ((axes[0], cm, "counts", "{:.0f}"),
                                (axes[1], norm, "row-normalised", "{:.2f}")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() or 1)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Round 2 model prediction"); ax.set_ylabel("star rating label")
        ax.set_title(title); ax.grid(False)
        th = mat.max() * 0.55 if mat.max() else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] > th else INK)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)
    fig.suptitle(f"Round 2 model vs {transfer['n']:,} independent star ratings  "
                 f"(accuracy {transfer['accuracy']:.3f}, κ {transfer['cohen_kappa']:.3f})",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    return save(fig, "fig05_round2_transfer.png")


def fig_transfer_domains(transfer) -> str:
    pd_ = transfer.get("per_domain") or {}
    if not pd_:
        return ""
    doms = list(pd_)[::-1]
    f1 = [pd_[d]["macro_f1"] for d in doms]
    rec = [pd_[d]["negative_recall"] for d in doms]
    y = np.arange(len(doms))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.barh(y - 0.2, f1, 0.4, color=ACCENT, label="macro-F1 vs stars")
    ax.barh(y + 0.2, rec, 0.4, color=WARN, label="recall on 1-2 star reviews")
    ax.set_yticks(y); ax.set_yticklabels(doms, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_title("Where the Round 2 model transfers, and where it does not")
    ax.legend(loc="lower right")
    for i, (a, b) in enumerate(zip(f1, rec)):
        ax.text(a + 0.01, i - 0.2, f"{a:.2f}", va="center", fontsize=6.6)
        ax.text(b + 0.01, i + 0.2, f"{b:.2f}", va="center", fontsize=6.6)
    despine(ax)
    return save(fig, "fig06_transfer_by_domain.png")


def fig_brand_heatmap(df: pd.DataFrame) -> str:
    d = df[df["is_delay_related"] & (df["brand"] != "")]
    top = d["brand"].value_counts().head(14).index
    types = d["delay_type"].value_counts().head(8).index
    mat = (d[d["brand"].isin(top) & d["delay_type"].isin(types)]
           .pivot_table(index="brand", columns="delay_type",
                        values="record_id", aggfunc="count").reindex(top).fillna(0))
    share = mat.div(mat.sum(axis=1).clip(lower=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    im = ax.imshow(share.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(share.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in share.columns], fontsize=7)
    ax.set_yticks(range(len(share.index))); ax.set_yticklabels(share.index, fontsize=7.5)
    ax.grid(False)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.4,
                        color="white" if v > share.values.max() * 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share of that brand's delay reports")
    ax.set_title("Each brand fails in its own way")
    return save(fig, "fig07_brand_delay_heatmap.png")


def fig_hourly(hourly: pd.DataFrame) -> str:
    h = hourly.copy()
    h["hour_utc"] = pd.to_datetime(h["hour_utc"])
    h["hod"] = h["hour_utc"].dt.hour
    g = h.groupby("hod").agg(n=("n", "mean"), s=("sentiment_mean", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.bar(g["hod"], g["n"], color=ACCENT, alpha=0.85, width=0.7)
    ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("mean reactions / hour")
    ax.set_title("When people complain")
    ax2 = ax.twinx()
    ax2.plot(g["hod"], g["s"], color="#b23a48", lw=1.6, marker="o", ms=3)
    ax2.set_ylabel("mean sentiment", color="#b23a48")
    ax2.tick_params(axis="y", colors="#b23a48"); ax2.grid(False)
    despine(ax)
    return save(fig, "fig08_hourly.png")


def fig_terms(analysis) -> str:
    terms = analysis.get("distinctive_terms_delay_vs_rest", [])[:18]
    if not terms:
        return ""
    t = pd.DataFrame(terms).sort_values("z")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.barh(t["term"], t["z"], color=SERIES[3], alpha=0.9)
    ax.set_xlabel("log-odds z (delay-related vs the rest)")
    ax.set_title("The vocabulary of a delay")
    despine(ax)
    return save(fig, "fig09_distinctive_terms.png")


def fig_attention(analysis, attention_path=None) -> str:
    from fetch import read_jsonl
    from config import RAW
    rows = read_jsonl(RAW / "attention.jsonl")
    if not rows:
        return ""
    a = pd.DataFrame(rows)
    a["date"] = pd.to_datetime(a["date"])
    tot = a.groupby("date")["views"].sum().reset_index()
    daily = pd.DataFrame(analysis["daily_overall"])
    daily["date"] = pd.to_datetime(daily["date"])
    m = daily.merge(tot, on="date", how="inner")
    if len(m) < 5:
        return ""
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(m["date"], m["n"] / m["n"].max(), color=ACCENT, lw=1.6,
            label="our reaction volume (scaled)")
    ax.plot(m["date"], m["views"] / m["views"].max(), color=SERIES[5], lw=1.6, ls="--",
            label="Wikipedia pageviews for the same brands (scaled)")
    r = m["n"].corr(m["views"], method="spearman")
    ax.set_title(f"Independent corroboration of the activity signal "
                 f"(Spearman ρ = {r:.2f})")
    ax.set_ylabel("scaled to own maximum")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left")
    despine(ax)
    fig.autofmt_xdate(rotation=0, ha="center")
    return save(fig, "fig10_attention_corroboration.png")


def fig_ratings_vs_sentiment(df: pd.DataFrame) -> str:
    d = df[df["rating"].notna()]
    if d.empty:
        return ""
    g = d.groupby("rating").agg(n=("record_id", "count"),
                                s=("sentiment_score", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    axes[0].bar(g["rating"], g["n"], color=ACCENT, alpha=0.9, width=0.65)
    axes[0].set_xlabel("star rating"); axes[0].set_ylabel("reviews")
    axes[0].set_title("Star ratings in the window")
    for x, v in zip(g["rating"], g["n"]):
        axes[0].text(x, v, f"{int(v):,}", ha="center", va="bottom", fontsize=6.8)
    despine(axes[0])
    axes[1].plot(g["rating"], g["s"], color="#b23a48", lw=1.8, marker="o")
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].set_xlabel("star rating"); axes[1].set_ylabel("mean model sentiment")
    axes[1].set_title("Model sentiment rises monotonically with stars")
    despine(axes[1])
    return save(fig, "fig11_ratings.png")


# ---------------------------------------------------------------------------
def fig_panel_artefact(analysis, df: pd.DataFrame) -> str:
    """The control chart: how much of the trend was the collector.

    This is the first figure in the report because it is the one that decides
    whether any of the others mean anything. Left: brand coverage, one bar per
    brand, showing which were observable for the whole window. Right: daily
    delay-related volume for all brands against the balanced panel. If the two
    lines diverge, the divergence is the sampling artefact, in the units of the
    thing being claimed.
    """
    cov = (analysis.get("coverage") or {}).get("by_brand")
    diag = (analysis.get("coverage") or {}).get("panel_diagnostics")
    if not cov:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    items = sorted(cov.items(), key=lambda kv: kv[1]["coverage_share"])
    names = [k for k, _ in items]
    shares = [v["coverage_share"] for _, v in items]
    inpanel = [v["in_balanced_panel"] for _, v in items]
    colours = [GOOD if p else WARN for p in inpanel]
    ax.barh(range(len(names)), shares, color=colours, height=0.78)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5.6)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of window days on which the brand appears")
    ax.set_title("Brand coverage of the window", loc="left")
    ax.axvline(0.95, color=INK, lw=0.8, ls="--")
    n_in = sum(inpanel)
    ax.text(0.5, -2.4, f"{n_in} of {len(names)} brands cover the window "
                       f"(green) and form the balanced panel",
            fontsize=7, color=MUTED, ha="center")
    despine(ax)

    ax = axes[1]
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    play = d[d["source"] == "google_play"]
    dly = play[play["is_delay_related"]]
    panel = [k for k, v in cov.items() if v["in_balanced_panel"]]
    allv = dly.groupby("date").size()
    balv = dly[dly["brand"].isin(panel)].groupby("date").size().reindex(
        allv.index, fill_value=0)
    ax.plot(allv.index, allv.values, color=WARN, lw=1.8,
            label="all brands (what we first reported)")
    ax.plot(balv.index, balv.values, color=GOOD, lw=1.8,
            label="balanced panel (what the report uses)")
    ax.fill_between(allv.index, balv.values, allv.values,
                    color=WARN, alpha=0.13)
    ax.set_ylabel("delay-related reactions per day")
    ax.set_title("Daily activity: the shaded area is the sampling artefact",
                 loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left")
    if diag:
        ra = diag["all_brands"]["pearson_r_with_day_index"]
        rb = diag["balanced_panel"]["pearson_r_with_day_index"]
        ax.text(0.99, 0.04,
                f"correlation with day index:  all brands r = {ra:+.2f}"
                f"   balanced panel r = {rb:+.2f}",
                transform=ax.transAxes, ha="right", fontsize=7.2, color=INK)
    despine(ax)
    fig.suptitle("Before any finding: is the trend in the public, or in our scraper?",
                 fontsize=10.5, fontweight="bold", x=0.007, ha="left", y=1.02)
    return save(fig, "fig00_panel_artefact.png")


def fig_spike_anatomy(analysis, df: pd.DataFrame) -> str:
    """What an endorsement spike is actually made of.

    A ratio-to-median of 12.35x sounds like a conversation taking off. Drawing
    the day's contributors shows whether it was, and on the largest spike in
    this corpus it was two reviews holding 88% of the endorsement while the
    median review that day got zero.
    """
    end = [s for s in analysis.get("engagement_spikes", [])
           if s.get("kind") == "endorsement"]
    if not end:
        return ""
    top = end[0]
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    day = d[(d["date"] == pd.Timestamp(top["date"])) & d["is_delay_related"]]
    if day.empty:
        return ""
    vals = np.sort(day["engagement"].fillna(0).to_numpy())[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    ax = axes[0]
    k = min(30, len(vals))
    ax.bar(range(k), vals[:k], color=[WARN if i < 2 else ACCENT for i in range(k)])
    ax.set_xlabel(f"the {k} most-endorsed reactions of {top['date']}")
    ax.set_ylabel("thumbs-up")
    ax.set_title(f"{top['date']}: {top['ratio_to_median']}x the median day", loc="left")
    conc = top.get("concentration") or {}
    if conc.get("top1_share") is not None:
        ax.text(0.97, 0.9, f"top 1 = {conc['top1_share']:.0%} of the day\n"
                           f"top 5 = {conc['top5_share']:.0%}\n"
                           f"median reaction = "
                           f"{top.get('median_engagement_that_day', 0):.0f} thumbs",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=INK)
    despine(ax)

    ax = axes[1]
    cum = np.cumsum(vals) / max(vals.sum(), 1)
    ax.plot(np.arange(1, len(cum) + 1), cum, color=ACCENT, lw=1.8)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("reactions, ranked by endorsement (log scale)")
    ax.set_ylabel("cumulative share of the day's endorsement")
    ax.set_title("A spike this concentrated is a post, not a conversation", loc="left")
    despine(ax)
    return save(fig, "fig12_spike_anatomy.png")


def main() -> dict:
    analysis = json.loads((REPORTS / "analysis.json").read_text(encoding="utf-8"))
    transfer_path = REPORTS / "round2_transfer.json"
    transfer = json.loads(transfer_path.read_text(encoding="utf-8")) if transfer_path.exists() else {}

    src = PROCESSED / "reactions_labelled.parquet"
    if not src.exists():
        src = PROCESSED / "reactions_labelled.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    daily = pd.DataFrame(analysis["daily_overall"])
    by_dom = pd.read_csv(PROCESSED / "timeseries_by_domain.csv")
    hourly = pd.read_csv(PROCESSED / "timeseries_hourly.csv")

    made = {}
    # First, because it decides whether the rest means anything.
    made["panel_artefact"] = fig_panel_artefact(analysis, df)
    made["timeline"] = fig_timeline(daily, analysis["sentiment_shifts"],
                                    analysis["engagement_spikes"])
    made["spike_anatomy"] = fig_spike_anatomy(analysis, df)
    made["delay_types"] = fig_delay_types(analysis)
    made["reaction_types"] = fig_reactions(analysis)
    made["domains"] = fig_domains(by_dom)
    if transfer:
        made["transfer"] = fig_transfer(transfer)
        made["transfer_domains"] = fig_transfer_domains(transfer)
    made["brand_heatmap"] = fig_brand_heatmap(df)
    made["hourly"] = fig_hourly(hourly)
    made["terms"] = fig_terms(analysis)
    made["attention"] = fig_attention(analysis)
    made["ratings"] = fig_ratings_vs_sentiment(df)
    made = {k: v for k, v in made.items() if v}
    (REPORTS / "figures.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    print(f"  {len(made)} figures")
    return made


if __name__ == "__main__":
    main()
