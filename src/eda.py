"""
Exploratory data analysis on the cleaned Social Engine tables.

    python src/eda.py

Reads   data/cleaned/posts_cleaned.csv, data/cleaned/users_cleaned.csv, reports/cleaning_audit.json
Writes  reports/figures/fig01..fig10 *.png   (print-ready, light theme)
        reports/eda_stats.json               (every number quoted in the report)

Statistical conventions
  * Every "pattern" is tested before it is called an insight. We report the test and p-value;
    p >= 0.05 is reported as "no evidence of a difference", never as a finding.
  * likes has NULLs (unrecoverable). Likes-based statistics use available cases (n shown);
    a median-imputed variant is computed only to show what imputation would distort.
  * Hour-of-day analysis uses only rows with timestamp_precision = 'second'. Date-only rows
    have no time of day and are excluded rather than assigned midnight.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── visual system (validated categorical slots; ink tokens for all text) ──
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PLATFORMS = ["Facebook", "Instagram", "Reddit", "Twitter", "YouTube"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "legend.frameon": False, "legend.fontsize": 8.5,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
})
thousands = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
STATS: dict = {}


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def p_fmt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def r(x, nd=1):
    return float(round(float(x), nd))


# ───────────────────────────── load ─────────────────────────────
posts = pd.read_csv(ROOT / "data/cleaned/posts_cleaned.csv", dtype={"likes": "Int64"})
users = pd.read_csv(ROOT / "data/cleaned/users_cleaned.csv")
audit = json.loads((ROOT / "reports/cleaning_audit.json").read_text(encoding="utf-8"))
posts["post_date"] = pd.to_datetime(posts.post_date)
posts["post_datetime_utc"] = pd.to_datetime(posts.post_datetime_utc)
posts["likes_f"] = posts.likes.astype("float")
posts["engagement"] = posts.likes_f + posts.shares + posts.comments   # NaN when likes unknown
N = len(posts)

STATS["overview"] = {
    "posts": N, "users": len(users), "platforms": int(posts.platform.nunique()),
    "date_start": str(posts.post_date.min().date()), "date_end": str(posts.post_date.max().date()),
    "days": int((posts.post_date.max() - posts.post_date.min()).days + 1),
    "complete_rows": int(posts[["platform", "text_content", "likes"]].notna().all(axis=1).sum()),
    "platform_counts_known": posts.platform.value_counts().to_dict(),
    "platform_count_chi2_uniform_p": r(stats.chisquare(posts.platform.value_counts().values).pvalue, 3),
}

# ═══════════════ 1. Corruption profile ═══════════════
d, t, pl, lk, tx = audit["duplicates"], audit["timestamps"], audit["platform"], audit["likes"], audit["text_content"]
corr_rows = [
    ("Duplicate rows (exact copies)", d["rows_removed"], "removed", "rows"),
    ("Timestamp: dd-mm-yyyy, time lost", t["format_counts"]["dd-mm-yyyy"], "date kept, time NULL", "posts"),
    ("Timestamp: Unix epoch", t["format_counts"]["unix_epoch"], "converted to UTC", "posts"),
    ("likes missing ('' / 'NULL')", lk["missing_total"], "NULL", "posts"),
    ("platform missing ('' / 'NULL')", pl["missing_total"], "NULL", "posts"),
    ("text missing ('' / 'NULL')", tx["missing_total_after_cleaning"], "NULL", "posts"),
    ("text suffix artifact", tx["rows_with_artifact"], "stripped", "posts"),
    ("text double spaces", tx["whitespace_collapsed_rows"], "collapsed", "posts"),
    ("likes sign-flipped (negative)", lk["negative_values"], "abs()", "posts"),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
labels = [c[0] for c in corr_rows][::-1]
vals = [c[1] for c in corr_rows][::-1]
ax.barh(labels, vals, color=BLUE, height=0.55)
for y, (v, row) in enumerate(zip(vals, corr_rows[::-1])):
    denom = d["raw_rows"] if row[3] == "rows" else N
    ax.text(v + 60, y, f"{v:,}  ({v / denom:.1%})  → {row[2]}", va="center", fontsize=8.2, color=INK2)
ax.set_xlim(0, max(vals) * 1.55)
ax.xaxis.set_major_formatter(thousands)
ax.grid(axis="y", visible=False)
ax.set_title("Corruption found per rule (after de-duplication; duplicates as share of raw rows)")
save(fig, "fig01_corruption_profile")

# Missingness mechanism: rate of each corruption across platforms and source formats (MCAR test)
flags = {
    "likes missing": posts.likes.isna(),
    "text missing": posts.text_content.isna(),
    "platform missing": posts.platform.isna(),
    "likes sign-flipped": posts.dq_likes_sign_corrected == 1,
    "text artifact": posts.dq_text_artifact_removed != "none",
    "duplicated on ingest": posts.dq_duplicate_copies_removed > 0,
}
by_fmt, by_plat, tests = {}, {}, {}
for name, f in flags.items():
    ct = pd.crosstab(posts.timestamp_source_format, f)
    p_fmt_ = stats.chi2_contingency(ct)[1]
    by_fmt[name] = (f.groupby(posts.timestamp_source_format).mean() * 100).round(1).to_dict()
    if name != "platform missing":
        known = posts.platform.notna()
        ct2 = pd.crosstab(posts.platform[known], f[known])
        p_pl = stats.chi2_contingency(ct2)[1]
        by_plat[name] = (f[known].groupby(posts.platform[known]).mean() * 100).round(1).to_dict()
    else:
        p_pl = None
    month = posts.post_date.dt.to_period("M")
    p_month = stats.chi2_contingency(pd.crosstab(month, f))[1]
    tests[name] = {"rate_pct": r(f.mean() * 100, 2), "chi2_p_by_source_format": r(p_fmt_, 3),
                   "chi2_p_by_platform": None if p_pl is None else r(p_pl, 3), "chi2_p_by_month": r(p_month, 3)}
lm, tm, pm = flags["likes missing"], flags["text missing"], flags["platform missing"]
overlap = {
    "likes&platform": [int((lm & pm).sum()), r(lm.mean() * pm.mean() * N, 0)],
    "likes&text": [int((lm & tm).sum()), r(lm.mean() * tm.mean() * N, 0)],
    "platform&text": [int((pm & tm).sum()), r(pm.mean() * tm.mean() * N, 0)],
}
STATS["missingness"] = {"tests": tests, "n_tests": int(sum(v is not None for t_ in tests.values() for k_, v in t_.items() if k_.startswith("chi2"))),
                        "by_source_format_pct": by_fmt, "by_platform_pct": by_plat,
                        "co_missing_observed_vs_expected_if_independent": overlap,
                        "empty_vs_NULL_token_ratio": {
                            "platform": pl["missing_tokens"], "likes": lk["missing_tokens"],
                            "text_content": tx["missing_tokens_raw"]}}

heat_rows = list(by_plat.keys())
mat = np.array([[by_plat[k][p] for p in PLATFORMS] for k in heat_rows])
fig, ax = plt.subplots(figsize=(7.2, 2.9))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
ax.imshow(mat, cmap=cmap, vmin=0, vmax=30, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(PLATFORMS)), PLATFORMS)
ax.set_yticks(range(len(heat_rows)),
              [f"{k}   ({p_fmt(tests[k]['chi2_p_by_platform'])})" for k in heat_rows])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Corruption rate by platform is flat: every χ² test is non-significant (MCAR)")
save(fig, "fig02_missingness_mcar")

# ═══════════════ 2. The midnight artifact ═══════════════
timed = posts[posts.timestamp_precision == "second"]
hour_true = timed.post_datetime_utc.dt.hour.value_counts().reindex(range(24), fill_value=0)
naive = hour_true.copy()
naive[0] += int((posts.timestamp_precision == "day").sum())     # what padding with 00:00:00 does
chi_true = stats.chisquare(hour_true.values)
STATS["hour_of_day"] = {
    "timed_rows": int(len(timed)), "date_only_rows": int(N - len(timed)),
    "true_hour0": int(hour_true[0]), "true_mean_per_hour": r(hour_true.mean()),
    "true_min_hour": [int(hour_true.idxmin()), int(hour_true.min())],
    "true_max_hour": [int(hour_true.idxmax()), int(hour_true.max())],
    "chi2_uniform_p": r(chi_true.pvalue, 3),
    "naive_hour0_if_padded_midnight": int(naive[0]),
    "naive_hour0_multiple_of_mean": r(naive[0] / naive.drop(0).mean(), 1),
    "naive_hour0_share_pct": r(naive[0] / N * 100, 1),
}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for ax, series, title, color in [
        (axes[0], naive, "Naive: date-only posts padded to 00:00", ORANGE),
        (axes[1], hour_true, "Correct: only posts with a real time", BLUE)]:
    ax.bar(series.index, series.values, color=[color if (h == 0 and series is naive) else BLUE for h in series.index],
           width=0.72)
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Hour of day (UTC)")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
axes[0].annotate(f"{naive[0]:,} posts\n= {STATS['hour_of_day']['naive_hour0_multiple_of_mean']}× other hours",
                 xy=(0.4, naive[0]), xytext=(4, naive[0] * 0.78), fontsize=8.2, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axes[1].text(11.5, hour_true.max() * 1.35, f"flat: {hour_true.min()}–{hour_true.max()} posts/hour\nχ² uniform {p_fmt(chi_true.pvalue)}",
             ha="center", fontsize=8.2, color=INK2)
axes[0].set_ylabel("Posts")
fig.suptitle("The “midnight spike” is created by the cleaning step, not by users", x=0.07, ha="left",
             fontsize=10.5, fontweight="semibold", y=1.04)
save(fig, "fig03_midnight_artifact")

# ═══════════════ 3. Sign-flip evidence and why likes are not imputed ═══════════════
raw = pd.read_csv(ROOT / "data/raw/Social_Engine_Posts_Corrupted.csv", dtype=str, keep_default_na=False).drop_duplicates("post_id")
raw_likes = pd.to_numeric(raw.likes.replace({"NULL": ""}), errors="coerce")
pos, neg_abs = raw_likes[raw_likes >= 0], -raw_likes[raw_likes < 0]
ks = stats.ks_2samp(pos, neg_abs)
ks_u = stats.kstest(posts.likes_f.dropna(), "uniform", args=(0, 5000))
median_fill = float(posts.likes_f.median())
imputed = posts.likes_f.fillna(median_fill)
STATS["likes"] = {
    "n_available": int(posts.likes.notna().sum()), "n_missing": int(posts.likes.isna().sum()),
    "negatives_all_float_formatted": lk["negative_values_float_formatted"] == lk["negative_values"],
    "mean_positive": r(pos.mean()), "mean_abs_negative": r(neg_abs.mean()),
    "ks_abs_negative_vs_positive_p": r(ks.pvalue, 3), "ks_uniform_0_5000_p": r(ks_u.pvalue, 3),
    "median_available": median_fill,
    "if_median_imputed": {"rows_at_single_value": int((imputed == median_fill).sum()),
                          "std_available": r(posts.likes_f.std()), "std_imputed": r(imputed.std()),
                          "std_shrink_pct": r((1 - imputed.std() / posts.likes_f.std()) * 100)},
}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
xs = np.linspace(0, 5000, 200)
for series, color, label in [(pos, BLUE, f"likes ≥ 0 (n={len(pos):,})"), (neg_abs, ORANGE, f"|likes| where negative (n={len(neg_abs):,})")]:
    axes[0].plot(xs, [(series <= x).mean() for x in xs], color=color, label=label)
axes[0].set_title("Negatives are mirrored positives", fontsize=9.5)
axes[0].set_xlabel("likes")
axes[0].set_ylabel("Cumulative share")
axes[0].legend(loc="upper left")
axes[0].text(4950, 0.06, f"KS test {p_fmt(ks.pvalue)}\n→ same distribution", ha="right", fontsize=8.2, color=INK2)
axes[0].xaxis.set_major_formatter(thousands)
bins = np.arange(0, 5001, 100)
axes[1].hist(imputed, bins=bins, color=ORANGE, label="median-imputed")
axes[1].hist(posts.likes_f.dropna(), bins=bins, color=BLUE, label="observed only")
axes[1].set_title("Median imputation would add a false spike", fontsize=9.5)
axes[1].set_xlabel("likes")
axes[1].set_ylabel("Posts per 100-like bin")
axes[1].legend(loc="upper right")
axes[1].annotate(f"{STATS['likes']['if_median_imputed']['rows_at_single_value']:,} posts\nat {median_fill:,.0f}",
                 xy=(median_fill + 60, 1250), xytext=(3300, 1150), fontsize=8.2, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axes[1].xaxis.set_major_formatter(thousands)
axes[1].yaxis.set_major_formatter(thousands)
axes[1].grid(axis="x", visible=False)
save(fig, "fig04_likes_signflip_and_imputation")

# ═══════════════ 4. Volume over time ═══════════════
daily = posts.groupby("post_date").size()
daily = daily.reindex(pd.date_range(posts.post_date.min(), posts.post_date.max()), fill_value=0)
monthly = posts.groupby(posts.post_date.dt.to_period("M")).size()
days_in_month = np.asarray(monthly.index.days_in_month, dtype=float)
per_day = monthly / days_in_month
dow = posts.post_date.dt.day_name().value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
chi_month_rate = stats.chisquare(monthly.values, f_exp=days_in_month / days_in_month.sum() * N)
chi_dow = stats.chisquare(dow.values)
# Anomalous days: one-sided Poisson tail per day, Bonferroni-corrected across all 365 days.
lam_day = daily.mean()
tail_hi = pd.Series(stats.poisson.sf(daily.values - 1, lam_day), index=daily.index)
tail_lo = pd.Series(stats.poisson.cdf(daily.values, lam_day), index=daily.index)
alpha_day = 0.05 / len(daily)
z_day = (daily - lam_day) / np.sqrt(lam_day)
STATS["daily_anomalies"] = {
    "method": "Poisson tail with lambda = mean daily posts; Bonferroni alpha = 0.05/365",
    "lambda": r(lam_day, 2), "bonferroni_alpha": alpha_day,
    "high_days": {str(k.date()): {"posts": int(daily[k]), "z": r(z_day[k], 2), "p": float(f"{tail_hi[k]:.2e}")}
                  for k in tail_hi[tail_hi < alpha_day].index},
    "low_days": {str(k.date()): {"posts": int(daily[k]), "z": r(z_day[k], 2), "p": float(f"{tail_lo[k]:.2e}")}
                 for k in tail_lo[tail_lo < alpha_day].index},
    "days_z_above_3": int((z_day > 3).sum()), "days_z_below_minus_3": int((z_day < -3).sum()),
    "expected_days_z_above_3_if_poisson": r(len(daily) * stats.poisson.sf(np.ceil(lam_day + 3 * np.sqrt(lam_day)) - 1, lam_day), 2),
}
STATS["time"] = {
    "daily_mean": r(daily.mean()), "daily_std": r(daily.std()), "daily_min": int(daily.min()), "daily_max": int(daily.max()),
    "poisson_dispersion_var_over_mean": r(daily.var() / daily.mean(), 2),
    "monthly_counts": {str(k): int(v) for k, v in monthly.items()},
    "monthly_posts_per_day": {str(k): r(v) for k, v in per_day.items()},
    "busiest_month_raw": [str(monthly.idxmax()), int(monthly.max())],
    "busiest_month_per_day": [str(per_day.idxmax()), r(per_day.max())],
    "quietest_month_raw": [str(monthly.idxmin()), int(monthly.min())],
    "chi2_month_counts_vs_calendar_days_p": r(chi_month_rate.pvalue, 3),
    "day_of_week": {k: int(v) for k, v in dow.items()}, "chi2_dow_uniform_p": r(chi_dow.pvalue, 3),
    "top_days": {str(k.date()): int(v) for k, v in daily.nlargest(3).items()},
}
fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.4), gridspec_kw={"height_ratios": [1.5, 1]})
axes[0].plot(daily.index, daily.values, color=AXIS, lw=0.9, label="daily posts")
axes[0].plot(daily.index, daily.rolling(28, center=True).mean(), color=BLUE, lw=2, label="28-day mean")
axes[0].set_title(f"Daily volume is stable at ~{daily.mean():.0f} posts/day (variance/mean = {daily.var() / daily.mean():.2f}, Poisson-like)")
axes[0].legend(loc="lower left", ncol=2)
axes[0].set_ylabel("Posts")
axes[0].set_ylim(0, daily.max() * 1.15)
axes[1].bar([str(p)[2:] for p in per_day.index], per_day.values, color=BLUE, width=0.6)
for i, (p, v) in enumerate(per_day.items()):
    axes[1].text(i, v + 0.6, f"{monthly[p]:,}", ha="center", fontsize=7.4, color=MUTED)
axes[1].set_ylim(0, per_day.max() * 1.25)
axes[1].set_ylabel("Posts per day")
axes[1].set_title(f"Month totals (labels) differ only by calendar length: χ² vs days-in-month {p_fmt(chi_month_rate.pvalue)}", fontsize=9.5)
axes[1].grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig05_volume_over_time")

# ═══════════════ 5. Engagement structure ═══════════════
eng = posts[["likes_f", "shares", "comments"]]
spear = eng.corr(method="spearman")
ks_sh = stats.kstest(posts.shares, "uniform", args=(0, 2000))
ks_cm = stats.kstest(posts.comments, "uniform", args=(0, 1000))
kw = {}
BONFERRONI_ALPHA = 0.05 / 4   # four engagement metrics tested against platform
for m in ["likes_f", "shares", "comments", "engagement"]:
    groups = [posts.loc[(posts.platform == p) & posts[m].notna(), m] for p in PLATFORMS]
    h = stats.kruskal(*groups)
    n_m = sum(len(g) for g in groups)
    kw[m] = {"kruskal_p": r(h.pvalue, 3), "significant_after_bonferroni": bool(h.pvalue < BONFERRONI_ALPHA),
             "epsilon_squared": r(h.statistic / (n_m - 1), 4),
             "spread_pct_of_mean": r((max(g.mean() for g in groups) - min(g.mean() for g in groups)) / posts[m].mean() * 100, 1),
             "means": {p: r(g.mean()) for p, g in zip(PLATFORMS, groups)},
             "ci95": {p: r(1.96 * g.std() / np.sqrt(len(g))) for p, g in zip(PLATFORMS, groups)},
             "spread_max_minus_min": r(max(g.mean() for g in groups) - min(g.mean() for g in groups))}
STATS["engagement"] = {
    "likes": {"mean": r(posts.likes_f.mean()), "median": r(posts.likes_f.median()), "min": int(posts.likes.min()), "max": int(posts.likes.max())},
    "shares": {"mean": r(posts.shares.mean()), "median": r(posts.shares.median()), "min": int(posts.shares.min()), "max": int(posts.shares.max()), "ks_uniform_p": r(ks_sh.pvalue, 3)},
    "comments": {"mean": r(posts.comments.mean()), "median": r(posts.comments.median()), "min": int(posts.comments.min()), "max": int(posts.comments.max()), "ks_uniform_p": r(ks_cm.pvalue, 3)},
    "spearman": {"likes_shares": r(spear.loc["likes_f", "shares"], 3), "likes_comments": r(spear.loc["likes_f", "comments"], 3),
                 "shares_comments": r(spear.loc["shares", "comments"], 3)},
    "by_platform": kw,
    "engagement_mean": r(posts.engagement.mean()),
}
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=False)
for ax, m, label in zip(axes, ["likes_f", "shares", "comments"], ["likes", "shares", "comments"]):
    means = [kw[m]["means"][p] for p in PLATFORMS]
    ci = [kw[m]["ci95"][p] for p in PLATFORMS]
    ax.errorbar(means, range(5), xerr=ci, fmt="o", color=BLUE, ecolor=AXIS, elinewidth=2, ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    overall = posts[m].mean()
    ax.axvline(overall, color=MUTED, lw=0.8)
    ax.set_yticks(range(5), PLATFORMS if ax is axes[0] else [""] * 5)
    ax.set_title(f"mean {label}\nKruskal–Wallis {p_fmt(kw[m]['kruskal_p'])}", fontsize=9)
    lo = min(a - b for a, b in zip(means, ci)); hi = max(a + b for a, b in zip(means, ci))
    pad = (hi - lo) * 0.25
    ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.set_major_formatter(thousands)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
STATS["engagement"]["bonferroni_alpha"] = BONFERRONI_ALPHA
worst_gap = max(kw[m]["spread_pct_of_mean"] for m in ["likes_f", "shares", "comments"])
fig.suptitle(f"Platform gaps are small (≤ {worst_gap:.1f}% of the mean) and none survives a Bonferroni "
             f"correction (α = {BONFERRONI_ALPHA:.4f})", x=0.02, ha="left", fontsize=10.2, fontweight="semibold", y=1.12)
fig.text(0.02, -0.04, "Dots = mean, bars = 95% CI, vertical line = overall mean. x-axes are zoomed and differ per panel.",
         fontsize=7.8, color=MUTED)
save(fig, "fig06_platform_engagement")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
for ax, col, hi in zip(axes, ["likes_f", "shares", "comments"], [5000, 2000, 1000]):
    vals_ = posts[col].dropna()
    ax.hist(vals_, bins=25, range=(0, hi), color=BLUE, rwidth=0.88)
    ax.axhline(len(vals_) / 25, color=ORANGE, lw=1.5)
    ax.set_title(col.replace("_f", ""), fontsize=9.5)
    ax.xaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Posts per bin")
axes[2].text(990, len(posts) / 25 * 1.08, "uniform\nexpectation", ha="right", va="bottom", fontsize=7.8, color=INK2)
fig.suptitle(f"Engagement metrics are flat (uniform) and independent: |Spearman ρ| ≤ {max(abs(v) for v in STATS['engagement']['spearman'].values()):.3f}",
             x=0.02, ha="left", fontsize=10.5, fontweight="semibold", y=1.1)
save(fig, "fig07_engagement_distributions")

# ═══════════════ 6. Users: activity, reach and engagement ═══════════════
ug = posts.groupby("user_id").agg(
    posts=("post_id", "size"), platforms=("platform", "nunique"),
    engagement_total=("engagement", "sum"), engagement_mean=("engagement", "mean"),
    shares_total=("shares", "sum"), comments_total=("comments", "sum"),
    likes_known=("likes", "count")).join(users.set_index("user_id"))
known_k = posts[posts.platform.notna()].groupby("user_id").size().reindex(ug.index, fill_value=0).values.astype(float)
lam = ug.posts.mean()
pois = stats.poisson(lam)
k = np.arange(ug.posts.min(), ug.posts.max() + 1)
observed = ug.posts.value_counts().reindex(k, fill_value=0)
rho_posts_total = stats.spearmanr(ug.posts, ug.engagement_total)
rho_follow_mean = stats.spearmanr(ug.follower_count, ug.engagement_mean, nan_policy="omit")
rho_follow_posts = stats.spearmanr(ug.follower_count, ug.posts)
top_total = ug.sort_values("engagement_total", ascending=False).head(10)
eligible = ug[ug.likes_known >= 5]
top_mean = eligible.sort_values("engagement_mean", ascending=False).head(10)
overlap_top = len(set(top_total.index) & set(top_mean.index))
# how much of a top-10 total ranking is explained by volume: posts of top10 vs overall
STATS["users"] = {
    "posts_per_user": {"mean": r(lam, 2), "median": r(ug.posts.median()), "min": int(ug.posts.min()), "max": int(ug.posts.max()),
                       "variance_over_mean": r(ug.posts.var() / lam, 2)},
    "users_on_2plus_platforms": int((ug.platforms >= 2).sum()), "mean_platforms_per_user": r(ug.platforms.mean(), 2),
    # Null model: each post's platform drawn uniformly from 5, independently. For a user with k posts of known
    # platform, E[distinct] = 5(1 - 0.8^k) and P(>= 2 distinct) = 1 - 0.2^(k-1).
    "expected_mean_platforms_if_random": r((5 * (1 - 0.8 ** known_k)).mean(), 2),
    "expected_users_2plus_if_random": r(np.where(known_k > 0, 1 - 0.2 ** (known_k - 1), 0).sum(), 0),
    "spearman_posts_vs_total_engagement": r(rho_posts_total.statistic, 3),
    "spearman_followers_vs_mean_engagement": [r(rho_follow_mean.statistic, 3), r(rho_follow_mean.pvalue, 3)],
    "spearman_followers_vs_posts": [r(rho_follow_posts.statistic, 3), r(rho_follow_posts.pvalue, 3)],
    "top10_by_total": [{"user_id": i, "posts": int(x.posts), "engagement_total": int(x.engagement_total),
                        "engagement_mean": r(x.engagement_mean), "followers": int(x.follower_count)} for i, x in top_total.iterrows()],
    "top10_by_mean_min5_known": [{"user_id": i, "posts": int(x.posts), "engagement_mean": r(x.engagement_mean),
                                  "followers": int(x.follower_count), "location": x.location} for i, x in top_mean.iterrows()],
    "top10_total_mean_posts": r(top_total.posts.mean(), 1),
    "top10_overlap_total_vs_mean": overlap_top,
    "followers": {"min": int(users.follower_count.min()), "max": int(users.follower_count.max()),
                  "mean": r(users.follower_count.mean()), "median": r(users.follower_count.median()),
                  "ks_uniform_p": r(stats.kstest(users.follower_count, "uniform", args=(0, 50000)).pvalue, 3)},
    "account_created_by_quarter": users.assign(q=pd.to_datetime(users.account_created).dt.quarter).q.value_counts().sort_index().to_dict(),
}
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
axes[0].bar(k, observed.values, color=BLUE, width=0.72, label="observed")
axes[0].plot(k, pois.pmf(k) * len(ug), color=ORANGE, marker="o", ms=3.5, lw=1.5, label=f"Poisson(λ={lam:.0f})")
axes[0].set_title("Posts per user", fontsize=9.5)
axes[0].set_xlabel("posts")
axes[0].set_ylabel("Users")
axes[0].legend(loc="upper right")
axes[0].grid(axis="x", visible=False)
axes[1].scatter(ug.posts, ug.engagement_total, s=6, color=BLUE, alpha=0.35, linewidths=0)
axes[1].scatter(top_total.posts, top_total.engagement_total, s=22, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="top-10 by total")
axes[1].set_title(f"Total engagement ≈ post count\nSpearman ρ = {rho_posts_total.statistic:.2f}", fontsize=9)
axes[1].set_xlabel("posts")
kfmt = matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}K")
axes[1].yaxis.set_major_formatter(kfmt)
axes[1].legend(loc="upper left")
axes[2].scatter(ug.follower_count, ug.engagement_mean, s=6, color=BLUE, alpha=0.35, linewidths=0)
axes[2].set_title(f"Followers don't predict engagement\nSpearman ρ = {rho_follow_mean.statistic:.3f} ({p_fmt(rho_follow_mean.pvalue)})", fontsize=9)
axes[2].set_xlabel("followers")
axes[2].set_ylabel("mean engagement / post")
axes[2].xaxis.set_major_formatter(kfmt)
fig.tight_layout()
save(fig, "fig08_user_activity")

# ═══════════════ 7. Language vs location ═══════════════
# Official / majority languages available among the 10 codes in the data (Italy and South Korea have none).
national = {"China": {"zh"}, "Japan": {"ja"}, "India": {"hi", "en"}, "Germany": {"de"}, "France": {"fr"},
            "Spain": {"es"}, "Mexico": {"es"}, "Brazil": {"pt"}, "Egypt": {"ar"}, "UAE": {"ar", "en"},
            "USA": {"en", "es"}, "UK": {"en"}, "Canada": {"en", "fr"}, "Australia": {"en"}, "Nigeria": {"en"},
            "South Africa": {"en"}, "Singapore": {"en", "zh"}, "Italy": set(), "South Korea": set()}
ct = pd.crosstab(users.country, users.language)
chi_lang = stats.chi2_contingency(ct)
cramers_v = np.sqrt(chi_lang[0] / (ct.values.sum() * (min(ct.shape) - 1)))
is_match = pd.Series([lang in national[c] for c, lang in zip(users.country, users.language)], index=users.index)
lang_share = users.language.value_counts(normalize=True)
expected_match = float(np.mean([sum(lang_share.get(l, 0) for l in national[c]) for c in users.country]))
lang_counts = users.language.value_counts()
chi_lang_uniform = stats.chisquare(lang_counts.values)
binom = stats.binomtest(int(is_match.sum()), len(users), expected_match)
unambiguous = users[~is_match & users.country.isin(["Germany", "Japan", "China", "Brazil", "France"])]
STATS["language"] = {
    "counts": lang_counts.to_dict(), "chi2_uniform_p": r(chi_lang_uniform.pvalue, 3),
    "country_language_chi2_p": r(chi_lang[1], 3), "cramers_v": r(cramers_v, 3),
    "national_language_match_pct": r(is_match.mean() * 100, 1),
    "expected_match_pct_if_random": r(expected_match * 100, 1), "binomial_p": r(binom.pvalue, 3),
    "users_in_countries_without_language_code": int(users.country.isin(["Italy", "South Korea"]).sum()),
    "example_mismatches": unambiguous.head(4)[["user_id", "location", "language"]].to_dict("records"),
    "top_country": [users.country.value_counts().idxmax(), int(users.country.value_counts().max())],
    "countries": int(users.country.nunique()), "cities": int(users.city.nunique()),
}
order_c = users.country.value_counts().index
ctm = ct.loc[order_c]
row_pct = ctm.div(ctm.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(7.2, 4.9))
im = ax.imshow(row_pct.values, cmap=cmap, vmin=0, vmax=30, aspect="auto")
for i, c in enumerate(ctm.index):
    for j, lcode in enumerate(ctm.columns):
        is_native = lcode in national[c]
        ax.text(j, i, ctm.iloc[i, j], ha="center", va="center", fontsize=7.4,
                color=SURFACE if row_pct.iloc[i, j] >= 20 else INK, fontweight="bold" if is_native else "normal")
        if is_native:
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.44), 0.92, 0.88, fill=False, ec=ORANGE, lw=1.5))
ax.set_xticks(range(len(ctm.columns)), [f"{c}\n{n}" for c, n in zip(ctm.columns, ctm.sum())], fontsize=8)
ax.set_yticks(range(len(ctm.index)), [f"{c} ({n})" for c, n in zip(ctm.index, ctm.sum(axis=1))], fontsize=8)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("% of the country's users", color=INK2, fontsize=8)
cb.outline.set_visible(False)
ax.set_title(f"Language is independent of country: χ² {p_fmt(chi_lang[1])}, Cramér's V = {cramers_v:.2f}\n"
             f"Outlined = a national language of that country. Match rate {is_match.mean():.1%} vs "
             f"{expected_match:.1%} expected by pure chance", fontsize=9.2)
fig.text(0.02, 0.0, "Cell text = users; colour = share of the country's users. Italy and South Korea have no matching code.",
         fontsize=7.8, color=MUTED)
save(fig, "fig09_language_vs_country")

# ═══════════════ 8. Text content ═══════════════
text = posts.text_content.dropna()
pos_verdict = ["Absolutely loving it", "Worth every penny", "Exceeded my expectations", "Highly recommend", "Best purchase ever"]
neg_verdict = ["Disappointed with the quality", "Returning it ASAP", "Not worth the money", "Had issues with it", "Wouldn't recommend"]
neu_verdict = ["It's okay", "Mixed feelings about it", "Not bad", "Does the job", "As expected"]
pos_emotion = ["Can't contain my excitement", "Loving it", "So happy", "Super excited", "Thrilled", "Delighted"]
neg_emotion = ["Feeling let down", "Sad to report", "Bummed out", "Fed up", "Frustrated"]


def has_any(s: pd.Series, words: list[str]) -> pd.Series:
    return s.str.contains("|".join(re.escape(w) for w in words), regex=True)


pv, nv, uv = has_any(text, pos_verdict), has_any(text, neg_verdict), has_any(text, neu_verdict)
# emotion words must not be counted inside the verdict phrase "Absolutely loving it"
emo_text = text.str.replace("Absolutely loving it", "", regex=False)
pe, ne = has_any(emo_text, pos_emotion), has_any(emo_text, neg_emotion)
verdict = np.select([pv & ~nv, nv & ~pv, uv], ["positive", "negative", "neutral"], "none")
contradiction = (pe & nv) | (ne & pv)
posts.loc[text.index, "verdict"] = verdict
v_eng = posts.dropna(subset=["engagement"]).groupby("verdict").engagement
kw_verdict = stats.kruskal(*[g.values for _, g in v_eng if len(g) > 30])

brands = ["Apple", "Google", "Samsung", "Nike", "Adidas", "Amazon", "Microsoft", "Toyota", "Coca-Cola", "Pepsi"]
brand_counts = {b: int(text.str.contains(re.escape(b)).sum()) for b in brands}
hashtags = text.str.findall(r"#\w+").explode().value_counts()
mentions = text.str.findall(r"@\w+").explode().value_counts()
opener = text.str.extract(r"^(Just unboxed|Just saw an ad|Has anyone else|Just tried|Comparing|Attended the|My \w+ \w+ review|"
                          r"Should I upgrade|How do I fix|Any advice|Anyone have tips|What|Loving it with)")[0]

# Calendar month numbers in which each campaign would plausibly be discussed.
seasonal = {"BlackFriday": [11], "CyberMonday": [11, 12], "ValentinesDeals": [2], "BackToSchool": [8, 9],
            "SummerSale": [6, 7, 8], "HolidaySpecial": [12], "NewYearNewYou": [1], "EarthDay": [4],
            "SpringBlast2025": [3, 4], "WinterWonders": [12, 1, 2], "FallCollection": [9, 10, 11]}
month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
months = posts.post_date.dt.to_period("M")
mentions_by_camp = {c: posts.text_content.fillna("").str.contains(c) for c in seasonal}
camp = pd.DataFrame({c: m.groupby(months).sum() for c, m in mentions_by_camp.items()}).T
camp_share = camp.div(camp.sum(axis=1), axis=0) * 100
camp_tests = {c: r(stats.chisquare(camp.loc[c].values, f_exp=monthly.values / monthly.sum() * camp.loc[c].sum()).pvalue, 3) for c in seasonal}
in_season_obs = sum(int(camp.loc[c, [p for p in camp.columns if p.month in mm]].sum()) for c, mm in seasonal.items())
in_season_exp = sum(camp.loc[c].sum() * monthly[[p for p in monthly.index if p.month in mm]].sum() / N for c, mm in seasonal.items())
spring_2024 = int((mentions_by_camp["SpringBlast2025"] & (posts.post_date < "2025-01-01")).sum())

STATS["text"] = {
    "posts_with_text": int(len(text)), "distinct_texts": int(text.nunique()),
    "mean_chars": r(text.str.len().mean()), "mean_words": r(text.str.split().str.len().mean()),
    "brand_mentions": brand_counts, "distinct_hashtags": int(len(hashtags)), "top_hashtags": hashtags.head(5).to_dict(),
    "hashtags_per_post_mean": r(text.str.count(r"#\w+").mean(), 2),
    "posts_with_mentions_pct": r(text.str.contains("@").mean() * 100, 1), "distinct_mentions": int(len(mentions)),
    "template_openers": opener.value_counts().head(8).to_dict(),
    "verdict_counts": pd.Series(verdict).value_counts().to_dict(),
    "contradictory_sentiment_posts": int(contradiction.sum()),
    "contradiction_pct_of_emotional_posts": r(contradiction.sum() / (pe | ne).sum() * 100, 1),
    "emotional_posts": int((pe | ne).sum()),
    "contradiction_examples": text[contradiction].head(3).tolist(),
    "engagement_mean_by_verdict": {k: r(v) for k, v in v_eng.mean().items()},
    "kruskal_engagement_by_verdict_p": r(kw_verdict.pvalue, 3),
    "seasonal_campaign_month_chi2_p": camp_tests, "springblast2025_mentions_in_2024": spring_2024,
    "seasonal_mentions_total": int(camp.values.sum()), "seasonal_in_season_observed": in_season_obs,
    "seasonal_in_season_expected_if_random": r(in_season_exp, 0),
    "seasonal_in_season_pct": r(in_season_obs / camp.values.sum() * 100, 1),
    "seasonal_in_season_expected_pct": r(in_season_exp / camp.values.sum() * 100, 1),
    "blackfriday_share_in_nov_pct": r(camp_share.loc["BlackFriday"].get(pd.Period("2024-11", "M"), 0), 1),
    "valentines_share_in_feb_pct": r(camp_share.loc["ValentinesDeals"].get(pd.Period("2025-02", "M"), 0), 1),
}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
bc = pd.Series(brand_counts).sort_values()
axes[0].barh(bc.index, bc.values, color=BLUE, height=0.55)
axes[0].set_title("Brand mentions (≈ equal)", fontsize=9.5)
axes[0].xaxis.set_major_formatter(thousands)
axes[0].grid(axis="y", visible=False)
cm = camp_share.loc[list(seasonal)]
im = axes[1].imshow(cm.values, cmap=cmap, vmin=0, vmax=20, aspect="auto")
for i, c in enumerate(cm.index):
    for j, per in enumerate(cm.columns):
        if per.month in seasonal[c]:
            axes[1].add_patch(plt.Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, fill=False, ec=ORANGE, lw=1.3))
axes[1].set_xticks(range(12), [p.strftime("%b") for p in cm.columns], fontsize=7.5)
axes[1].set_yticks(range(len(cm.index)), [f"{c} ({camp.loc[c].sum()})" for c in cm.index], fontsize=7.5)
axes[1].grid(False)
for s in axes[1].spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)
cb.set_label("% of the campaign's mentions", color=INK2, fontsize=7.5)
cb.outline.set_visible(False)
axes[1].set_title(f"Campaign mentions by month (outlined = in season)\n{in_season_obs / camp.values.sum():.1%} in season vs "
                  f"{in_season_exp / camp.values.sum():.1%} expected by chance", fontsize=9)
fig.tight_layout()
save(fig, "fig10_text_brands_campaigns")

(ROOT / "reports" / "eda_stats.json").write_text(json.dumps(STATS, indent=2, default=str), encoding="utf-8")
print(f"wrote {len(list(FIG.glob('*.png')))} figures to reports/figures and reports/eda_stats.json")
