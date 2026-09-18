"""Held-out evaluation, figures and error analysis.

Everything the two PDFs quote is computed here and written to
``round2/reports`` as JSON/CSV first, so the reports never contain a number
that was typed by hand.

The held-out slice is the 20% of *unique posts* that the model search in
``train.py`` never touched, reconstructed from the same seed.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import learning_curve

from audit_labels import DEFAULT_CLASS, PRIORITY, RECOVERED_RULE, apply_rule
from config import CV_FOLDS, REPORTS, SEED, TASKS, TEXT_COL
from dataio import grouped_cv, load, make_split
from plotting import ACCENT, GRID, INK, MUTED, SEQ, SERIES, despine, plt, save

PRETTY = {"sentiment": "Sentiment", "topic": "Topic"}


# ---------------------------------------------------------------------------
# core metrics
# ---------------------------------------------------------------------------
def corpus_profile(df) -> dict:
    """Counts the preprocessing and problem-definition sections quote.

    Computed here rather than typed into the PDFs, so that pointing the
    pipeline at a different file updates the prose as well as the tables.
    """
    text = df[TEXT_COL]
    low = text.str.lower()
    chars = text.str.len()
    words = text.str.split().str.len()
    backslash = chr(92)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1] - 1),        # text_group is ours, not theirs
        "n_nulls": int(df.isna().sum().sum()),
        "chars": {"min": int(chars.min()), "median": int(chars.median()),
                  "max": int(chars.max())},
        "words": {"min": int(words.min()), "median": int(words.median()),
                  "max": int(words.max())},
        "unicode_escapes": int(text.str.contains(re.escape(backslash + "u00"), regex=True).sum()),
        "escaped_quotes": int(text.str.contains(re.escape(backslash + '"'), regex=True).sum()),
        "quote_wrapped": int((text.str.startswith('"') & text.str.endswith('"')).sum()),
        "mentions": int(text.str.contains("@", regex=False).sum()),
        "hashtags": int(text.str.contains("#", regex=False).sum()),
        "retweets": int(text.str.contains(r"\bRT\b", regex=True).sum()),
        "truncated": int(text.str.rstrip().str.endswith("...").sum()),
        "urls": int(low.str.contains("http", regex=False).sum()),
        "class_counts": {
            task: df[col].value_counts().sort_index().to_dict()
            for task, col in TASKS.items()
        },
        "majority_share": {
            task: float(df[col].value_counts(normalize=True).max())
            for task, col in TASKS.items()
        },
        "cue_word_polarity": {
            kw: df.loc[low.str.contains(kw, regex=False), TASKS["sentiment"]]
                  .value_counts(normalize=True).round(3).to_dict()
            for kw in ["love", "hate", "happy", "worst"]
        },
    }


def metric_block(y_true, y_pred, proba=None, classes=None) -> dict:
    """Every headline number for one task on one slice of data."""
    classes = classes or sorted(set(y_true))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    block = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "labels": list(classes),
    }
    if proba is not None and len(classes) > 1:
        try:
            block["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr",
                              average="macro", labels=classes))
        except ValueError:
            pass
    return block


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_class_distribution(df) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (task, col) in zip(axes, TASKS.items()):
        counts = df[col].value_counts().sort_index()
        bars = ax.bar(range(len(counts)), counts.values,
                      color=SERIES[: len(counts)], width=0.62)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([c.replace("_", "\n") for c in counts.index], fontsize=7)
        ax.set_title(f"{PRETTY[task]} - {col}")
        ax.set_ylabel("posts")
        ax.set_ylim(0, counts.max() * 1.18)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                    ha="center", va="bottom", fontsize=7, color=INK)
        despine(ax)
    fig.suptitle("Dataset 2: one balanced target, one severely skewed target",
                 fontsize=9.5, fontweight="bold", y=1.04)
    return save(fig, "fig01_class_distribution.png")


def fig_length_profile(df) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    words = df[TEXT_COL].str.split().str.len()
    axes[0].hist(words, bins=30, color=ACCENT, alpha=0.85)
    axes[0].set_title("Post length")
    axes[0].set_xlabel("words")
    axes[0].set_ylabel("posts")
    despine(axes[0])
    data = [words[df.sentiment_label == c] for c in sorted(df.sentiment_label.unique())]
    bp = axes[1].boxplot(data, patch_artist=True, widths=0.55,
                         medianprops=dict(color=INK), flierprops=dict(
                             marker=".", markersize=2, markerfacecolor=MUTED,
                             markeredgecolor="none", alpha=0.4))
    for patch, colour in zip(bp["boxes"], SERIES):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
        patch.set_edgecolor(colour)
    axes[1].set_xticklabels(sorted(df.sentiment_label.unique()))
    axes[1].set_title("Length carries almost no polarity signal")
    axes[1].set_ylabel("words")
    despine(axes[1])
    return save(fig, "fig02_length_profile.png")


def fig_model_comparison(tables: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for ax, (task, table) in zip(axes, tables.items()):
        t = table.sort_values("cv_f1_macro")
        names = [n.replace(" + ", "\n+ ") for n in t["model"]]
        ax.barh(range(len(t)), t["cv_f1_macro"],
                xerr=t["cv_f1_macro_std"], color=ACCENT, alpha=0.85,
                error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels(names, fontsize=6.2)
        ax.set_xlabel("grouped 5-fold CV macro-F1")
        ax.set_title(f"{PRETTY[task]}")
        ax.set_xlim(0, 1.02)
        for i, v in enumerate(t["cv_f1_macro"]):
            ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=6.4, color=INK)
        despine(ax)
    fig.suptitle("Candidate comparison - feature ablation first, then model family",
                 fontsize=9.5, fontweight="bold", y=1.02)
    return save(fig, "fig03_model_comparison.png")


def fig_confusion(block: dict, task: str, idx: int) -> str:
    labels = block["labels"]
    cm = np.array(block["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, mat, title, fmt in (
        (axes[0], cm, "counts", "{:.0f}"),
        (axes[1], norm, "row-normalised (recall)", "{:.2f}"),
    ):
        im = ax.imshow(mat, cmap=SEQ, vmin=0,
                       vmax=mat.max() if mat.max() else 1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_yticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(title)
        ax.grid(False)
        thresh = mat.max() * 0.55 if mat.max() else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] > thresh else INK)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)
    fig.suptitle(f"{PRETTY[task]} - held-out confusion matrix "
                 f"(n = {block['n']:,})", fontsize=9.5, fontweight="bold", y=1.03)
    return save(fig, f"fig0{idx}_confusion_{task}.png")


def fig_per_class(blocks: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for ax, (task, block) in zip(axes, blocks.items()):
        labels = block["labels"]
        metrics = ["precision", "recall", "f1"]
        width = 0.26
        x = np.arange(len(labels))
        for k, m in enumerate(metrics):
            vals = [block["per_class"][c][m] for c in labels]
            ax.bar(x + (k - 1) * width, vals, width, label=m, color=SERIES[k], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title(f"{PRETTY[task]} - per class (held-out)")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        despine(ax)
    return save(fig, "fig06_per_class_metrics.png")


def fig_learning_curve(pipe, X, y, groups, task: str) -> tuple[str, dict]:
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, groups=groups, cv=grouped_cv(CV_FOLDS),
        train_sizes=np.linspace(0.15, 1.0, 6), scoring="f1_macro", n_jobs=1,
    )
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot(sizes, train_scores.mean(1), "o-", color=MUTED, label="training folds")
    ax.fill_between(sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), color=ACCENT, alpha=0.15)
    ax.plot(sizes, val_scores.mean(1), "o-", color=ACCENT, label="validation folds")
    ax.set_xlabel("training posts")
    ax.set_ylabel("macro-F1")
    ax.set_title(f"{PRETTY[task]} - learning curve")
    ax.legend()
    despine(ax)
    name = save(fig, f"fig07_learning_curve_{task}.png")
    return name, {
        "train_sizes": [int(s) for s in sizes],
        "train_f1_macro": [float(v) for v in train_scores.mean(1)],
        "val_f1_macro": [float(v) for v in val_scores.mean(1)],
        "final_gap": float(train_scores.mean(1)[-1] - val_scores.mean(1)[-1]),
    }


def fig_leakage(leaks: dict) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    tasks = list(leaks)
    x = np.arange(len(tasks))
    grouped = [leaks[t]["grouped_by_text_f1_macro"] for t in tasks]
    naive = [leaks[t]["random_row_split_f1_macro"] for t in tasks]
    ax.bar(x - 0.18, grouped, 0.34, label="split by unique post (reported)", color=ACCENT)
    ax.bar(x + 0.18, naive, 0.34, label="split by row (leaky)", color=SERIES[1])
    for i, (a, b) in enumerate(zip(grouped, naive)):
        ax.text(i - 0.18, a + 0.012, f"{a:.3f}", ha="center", fontsize=7, color=INK)
        ax.text(i + 0.18, b + 0.012, f"{b:.3f}", ha="center", fontsize=7, color=INK)
        ax.annotate(f"+{b - a:.3f}", xy=(i, max(a, b) + 0.07), ha="center",
                    fontsize=7.5, color=SERIES[3], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[t] for t in tasks])
    ax.set_ylabel("CV macro-F1")
    ax.set_ylim(0, 1.15)
    ax.set_title("Score a duplicate-blind split would have invented")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=1)
    despine(ax)
    return save(fig, "fig08_split_leakage.png")


def fig_reliability(conf, correct, task: str) -> tuple[str, dict]:
    bins = np.linspace(0.3, 1.0, 9)
    idx = np.digitize(conf, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 15:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot([0.3, 1], [0.3, 1], "--", color=GRID, lw=1.2, label="perfect calibration")
    ax.plot(xs, ys, "o-", color=ACCENT, label="model")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=6, color=MUTED)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"{PRETTY[task]} - is the confidence honest?")
    ax.legend(loc="upper left")
    despine(ax)
    ece = float(np.sum(np.array(ns) * np.abs(np.array(xs) - np.array(ys))) / max(sum(ns), 1))
    return save(fig, f"fig09_reliability_{task}.png"), {"expected_calibration_error": ece,
                                                        "bins": len(xs)}


def fig_top_features(pipe, classes, task: str) -> str:
    """Highest-weight word features per class from the uncalibrated linear model."""
    union = pipe.named_steps["f"]
    clf = pipe.named_steps["c"]
    word = dict(union.transformer_list).get("word")
    if word is None or not hasattr(clf, "coef_"):
        return ""
    names = np.asarray(word.get_feature_names_out())
    n_word = len(names)
    fig, axes = plt.subplots(1, len(classes), figsize=(2.55 * len(classes), 3.0))
    axes = np.atleast_1d(axes)
    for ax, i, cls in zip(axes, range(len(classes)), classes):
        coef = clf.coef_[i][:n_word] if clf.coef_.shape[0] > 1 else clf.coef_[0][:n_word]
        top = np.argsort(coef)[-12:]
        ax.barh(range(len(top)), coef[top], color=SERIES[i % len(SERIES)], alpha=0.88)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names[top], fontsize=6.4)
        ax.set_title(cls.replace("_", " "), fontsize=8.5)
        ax.set_xlabel("weight")
        despine(ax)
    fig.suptitle(f"{PRETTY[task]} - what the model actually keys on (word block)",
                 fontsize=9.5, fontweight="bold", y=1.03)
    return save(fig, f"fig10_top_features_{task}.png")


def fig_error_profile(test_df, wrong, conf, task: str) -> tuple[str, dict]:
    words = test_df[TEXT_COL].str.split().str.len().to_numpy()
    edges = [0, 12, 16, 20, 24, 40]
    rates, centres, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (words >= lo) & (words < hi)
        if m.sum() >= 20:
            rates.append(wrong[m].mean())
            centres.append(f"{lo}-{hi}")
            counts.append(int(m.sum()))
    cues = {
        "has emoticon": test_df[TEXT_COL].str.contains(r"[:;=]-?[\)\(\[\]dpDP]|<3", regex=True).to_numpy(),
        "has negation": test_df[TEXT_COL].str.lower().str.contains(
            r"\b(?:not|no|never|don't|dont|can't|cant|isn't|won't)\b", regex=True).to_numpy(),
        "has hashtag": test_df[TEXT_COL].str.contains("#", regex=False).to_numpy(),
        "has @mention": test_df[TEXT_COL].str.contains("@", regex=False).to_numpy(),
        "ALL CAPS word": test_df[TEXT_COL].str.contains(r"\b[A-Z]{3,}\b", regex=True).to_numpy(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7))
    axes[0].bar(range(len(rates)), rates, color=ACCENT, alpha=0.88, width=0.62)
    axes[0].set_xticks(range(len(rates)))
    axes[0].set_xticklabels(centres, fontsize=7)
    axes[0].set_xlabel("post length (words)")
    axes[0].set_ylabel("error rate")
    axes[0].set_title("Error rate by length")
    despine(axes[0])

    names, with_, without = [], [], []
    for cue, mask in cues.items():
        if mask.sum() >= 20 and (~mask).sum() >= 20:
            names.append(cue)
            with_.append(wrong[mask].mean())
            without.append(wrong[~mask].mean())
    x = np.arange(len(names))
    axes[1].barh(x - 0.19, with_, 0.36, label="cue present", color=SERIES[1])
    axes[1].barh(x + 0.19, without, 0.36, label="cue absent", color=ACCENT)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("error rate")
    axes[1].set_title("Error rate by surface cue")
    axes[1].legend(loc="lower right")
    despine(axes[1])

    axes[2].hist([conf[~wrong.astype(bool)], conf[wrong.astype(bool)]], bins=12,
                 color=[ACCENT, SERIES[3]], label=["correct", "wrong"], alpha=0.9)
    axes[2].set_xlabel("confidence")
    axes[2].set_ylabel("posts")
    axes[2].set_title("Are the mistakes confident?")
    axes[2].legend()
    despine(axes[2])
    fig.suptitle(f"{PRETTY[task]} - where the errors live", fontsize=9.5,
                 fontweight="bold", y=1.04)
    stats = {
        "error_rate_by_length": dict(zip(centres, [float(r) for r in rates])),
        "error_rate_by_cue": {n: {"present": float(a), "absent": float(b)}
                              for n, a, b in zip(names, with_, without)},
        "mean_conf_correct": float(conf[~wrong.astype(bool)].mean()),
        "mean_conf_wrong": float(conf[wrong.astype(bool)].mean()),
        "share_of_errors_above_0_8_conf": float((conf[wrong.astype(bool)] > 0.8).mean()),
    }
    return save(fig, f"fig11_error_profile_{task}.png"), stats


def fig_topic_rule(df, test_df, y_pred) -> tuple[str, dict]:
    """The topic model's remaining errors, seen through the recovered rule."""
    low = test_df[TEXT_COL].str.lower()
    rule_pred = apply_rule(test_df[TEXT_COL])
    wrong = (y_pred != test_df["topic_category"].to_numpy())
    train_low = df[TEXT_COL].str.lower()

    # how often does the deciding trigger appear in the corpus at all?
    def deciding_trigger(text: str) -> str | None:
        for cls in PRIORITY:
            for kw in RECOVERED_RULE[cls]:
                if kw in text:
                    return kw
        return None

    trig = low.map(deciding_trigger)
    freq = {kw: int(train_low.str.contains(kw, regex=False).sum())
            for cls in PRIORITY for kw in RECOVERED_RULE[cls]}
    rare = trig.map(lambda k: freq.get(k, 0) < 60 if k else False).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    groups = {
        "no trigger\n(Community)": trig.isna().to_numpy(),
        "common trigger\n(>=60 posts)": (~trig.isna() & ~rare).to_numpy(),
        "rare trigger\n(<60 posts)": rare,
    }
    rates = [wrong[m].mean() if m.sum() else 0 for m in groups.values()]
    ns = [int(m.sum()) for m in groups.values()]
    bars = axes[0].bar(range(3), rates, color=[ACCENT, SERIES[2], SERIES[3]], width=0.6)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(list(groups), fontsize=7)
    axes[0].set_ylabel("model error rate")
    axes[0].set_title("Topic errors concentrate on rare triggers")
    for b, r, n in zip(bars, rates, ns):
        axes[0].text(b.get_x() + b.get_width() / 2, r, f"{r:.1%}\nn={n}",
                     ha="center", va="bottom", fontsize=6.8, color=INK)
    axes[0].set_ylim(0, max(rates) * 1.45 if max(rates) else 1)
    despine(axes[0])

    scores = {"learned model\n(char+word TF-IDF)": float((~wrong).mean()),
              "recovered substring rule": float((rule_pred.to_numpy() ==
                                                 test_df["topic_category"].to_numpy()).mean())}
    bars = axes[1].bar(range(2), list(scores.values()),
                       color=[ACCENT, SERIES[4]], width=0.55)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(list(scores), fontsize=7)
    axes[1].set_ylabel("held-out accuracy")
    axes[1].set_ylim(0, 1.12)
    for b, v in zip(bars, scores.values()):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                     ha="center", fontsize=7.5, color=INK)
    axes[1].set_title("The ceiling is a rule, not a model")
    despine(axes[1])
    fig.suptitle("Topic task - diagnosing the gap", fontsize=9.5,
                 fontweight="bold", y=1.04)
    stats = {
        "error_rate_no_trigger": float(wrong[groups["no trigger\n(Community)"]].mean()),
        "error_rate_common_trigger": float(wrong[groups["common trigger\n(>=60 posts)"]].mean()),
        "error_rate_rare_trigger": float(wrong[groups["rare trigger\n(<60 posts)"]].mean()),
        "trigger_corpus_frequency": freq,
        "model_accuracy": scores["learned model\n(char+word TF-IDF)"],
        "rule_accuracy": scores["recovered substring rule"],
    }
    return save(fig, "fig12_topic_rule_gap.png"), stats


# ---------------------------------------------------------------------------
# error tables
# ---------------------------------------------------------------------------
def error_table(test_df, y_true, y_pred, conf, task, limit=14) -> pd.DataFrame:
    wrong = y_true != y_pred
    out = pd.DataFrame({
        "text_id": test_df["text_id"].to_numpy()[wrong],
        "post_text": test_df[TEXT_COL].to_numpy()[wrong],
        "actual": y_true[wrong],
        "predicted": y_pred[wrong],
        "confidence": conf[wrong].round(3),
    }).sort_values("confidence", ascending=False)
    out.to_csv(REPORTS / f"errors_{task}.csv", index=False, encoding="utf-8")
    return out.head(limit)


def confusion_pairs(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and cm[i, j]:
                rows.append({"actual": a, "predicted": b, "n": int(cm[i, j]),
                             "share_of_actual": float(cm[i, j] / cm[i].sum())})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> dict:
    import joblib
    from config import MODEL_BUNDLE
    from train import candidates

    bundle = joblib.load(MODEL_BUNDLE)
    df = load()
    results: dict = {"dataset_sha256": bundle["dataset_sha256"],
                     "seed": SEED,
                     "duplicates": bundle["duplicates"],
                     "corpus": corpus_profile(df),
                     "figures": {},
                     "tasks": {}}

    print("figures: corpus")
    results["figures"]["class_distribution"] = fig_class_distribution(df)
    results["figures"]["length_profile"] = fig_length_profile(df)

    tables = {t: pd.read_csv(REPORTS / f"model_comparison_{t}.csv") for t in TASKS}
    results["figures"]["model_comparison"] = fig_model_comparison(tables)
    results["figures"]["split_leakage"] = fig_leakage(bundle["leakage"])

    blocks = {}
    for i, (task, col) in enumerate(TASKS.items()):
        print(f"\nevaluating: {task}")
        split = make_split(df, task)
        test = split.test
        model = bundle["models"][task]
        classes = list(model.classes_)
        y_true = test[col].to_numpy()
        proba = model.predict_proba(test[TEXT_COL].to_numpy())
        y_pred = np.asarray(classes)[proba.argmax(1)]
        conf = proba.max(1)

        block = metric_block(y_true, y_pred, proba=proba, classes=classes)
        block["selected_model"] = bundle["selected"][task]
        block["cv"] = {
            "f1_macro": float(tables[task].iloc[0]["cv_f1_macro"]),
            "f1_macro_std": float(tables[task].iloc[0]["cv_f1_macro_std"]),
            "folds": CV_FOLDS,
        }
        block["leakage"] = bundle["leakage"][task]
        block["classification_report"] = classification_report(
            y_true, y_pred, labels=classes, zero_division=0)

        results["figures"][f"confusion_{task}"] = fig_confusion(block, task, 4 + i)

        wrong = (y_true != y_pred).astype(float)
        name, err_stats = fig_error_profile(test, wrong, conf, task)
        results["figures"][f"error_profile_{task}"] = name
        block["error_profile"] = err_stats

        name, cal = fig_reliability(conf, (y_true == y_pred).astype(float), task)
        results["figures"][f"reliability_{task}"] = name
        block["calibration"] = cal

        # uncalibrated refit: needed for coefficients and the learning curve
        pipe = candidates(task)[bundle["selected"][task]]
        lc_name, lc = fig_learning_curve(
            pipe, split.train[TEXT_COL].to_numpy(), split.train[col].to_numpy(),
            split.groups_train, task)
        results["figures"][f"learning_curve_{task}"] = lc_name
        block["learning_curve"] = lc

        pipe.fit(split.train[TEXT_COL].to_numpy(), split.train[col].to_numpy())
        feat_name = fig_top_features(pipe, list(pipe.named_steps["c"].classes_), task)
        if feat_name:
            results["figures"][f"top_features_{task}"] = feat_name

        pairs = confusion_pairs(y_true, y_pred, classes)
        pairs.to_csv(REPORTS / f"confusion_pairs_{task}.csv", index=False)
        block["top_confusions"] = pairs.head(4).to_dict("records")
        block["worst_errors"] = error_table(
            test, y_true, y_pred, conf, task).to_dict("records")

        if task == "topic":
            name, topic_stats = fig_topic_rule(df, test, y_pred)
            results["figures"]["topic_rule_gap"] = name
            block["rule_diagnosis"] = topic_stats

        blocks[task] = block
        results["tasks"][task] = block
        print(f"  held-out macro-F1 {block['macro_f1']:.4f} | "
              f"accuracy {block['accuracy']:.4f} | kappa {block['cohen_kappa']:.4f}")

    results["figures"]["per_class"] = fig_per_class(blocks)

    (REPORTS / "metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: metrics.json and {len(results['figures'])} figures")
    return results


if __name__ == "__main__":
    main()
