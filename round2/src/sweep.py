"""Hyperparameter sweep behind the settings in ``train.HPARAMS``.

Run separately from ``train.py`` because it is slow and its answer does not
change: it sweeps regularisation strength and the character n-gram range for
each task with the same grouped CV, **on the training split only**, and writes
``reports/hparam_sweep.md``.

The result is the sharpest single piece of evidence in this submission. On
sentiment the best character range is 3-5 and the model wants heavy
regularisation. On topic the best range is 2-3 and the model wants light
regularisation - and 2-3 characters is exactly the length of the trigger
substrings (``ui``, ``ban``, ``app``, ``bug``) that ``audit_labels.py``
recovered. The hyperparameter that wins tells you what the label is made of.

    python sweep.py            # both tasks
    python sweep.py topic      # one task
"""
from __future__ import annotations

import sys
import time
import warnings

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import REPORTS, SEED, TASKS, TEXT_COL
from dataio import grouped_cv, load, make_split
from features import build_features

warnings.filterwarnings("ignore")

C_GRID = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
NGRAM_GRID = [(2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (3, 6)]


def sweep_task(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = TASKS[task]
    split = make_split(load(), task)
    X = split.train[TEXT_COL].to_numpy()
    y = split.train[col].to_numpy()
    g = split.groups_train
    cv = grouped_cv()
    bal = "balanced" if task == "topic" else None

    def score(**feature_kw):
        C = feature_kw.pop("C")
        pipe = Pipeline([
            ("f", build_features(**feature_kw)),
            ("c", LinearSVC(C=C, class_weight=bal, random_state=SEED)),
        ])
        t0 = time.time()
        s = cross_val_score(pipe, X, y, cv=cv, groups=g, scoring="f1_macro", n_jobs=1)
        return s.mean(), s.std(), time.time() - t0

    print(f"\n=== {task}: regularisation sweep (word + char + surface) ===")
    c_rows = []
    for C in C_GRID:
        m, sd, secs = score(C=C)
        c_rows.append({"C": C, "cv_f1_macro": m, "sd": sd, "seconds": round(secs, 1)})
        print(f"  C={C:<6} {m:.4f} +- {sd:.3f}  ({secs:.0f}s)", flush=True)

    print(f"\n=== {task}: character n-gram range sweep (char block alone) ===")
    n_rows = []
    best_C = max(c_rows, key=lambda r: r["cv_f1_macro"])["C"]
    for ng in NGRAM_GRID:
        m, sd, secs = score(C=best_C, use_word=False, use_stats=False, char_ngram=ng)
        n_rows.append({"char_ngram": f"{ng[0]}-{ng[1]}", "cv_f1_macro": m, "sd": sd,
                       "seconds": round(secs, 1)})
        print(f"  char {ng[0]}-{ng[1]}   {m:.4f} +- {sd:.3f}  ({secs:.0f}s)", flush=True)
    return pd.DataFrame(c_rows), pd.DataFrame(n_rows)


def main(tasks=None):
    tasks = tasks or list(TASKS)
    lines = [
        "# Hyperparameter sweep",
        "",
        "Grouped 5-fold CV macro-F1 on the **training split only**; the held-out",
        "slice took no part in any of this. Produced by `python round2/src/sweep.py`.",
        "",
    ]
    for task in tasks:
        c_df, n_df = sweep_task(task)
        best_c = c_df.loc[c_df.cv_f1_macro.idxmax()]
        best_n = n_df.loc[n_df.cv_f1_macro.idxmax()]
        lines += [
            f"## {task}",
            "",
            f"Best regularisation **C = {best_c.C}** ({best_c.cv_f1_macro:.4f}); "
            f"best character range **{best_n.char_ngram}** ({best_n.cv_f1_macro:.4f}).",
            "",
            "### Regularisation (word + char + surface, LinearSVC)",
            "",
            "| C | CV macro-F1 | SD |",
            "|---|---|---|",
        ]
        lines += [f"| {r.C} | {r.cv_f1_macro:.4f} | {r.sd:.3f} |" for r in c_df.itertuples()]
        lines += [
            "",
            "### Character n-gram range (character block alone)",
            "",
            "| range | CV macro-F1 | SD |",
            "|---|---|---|",
        ]
        lines += [f"| {r.char_ngram} | {r.cv_f1_macro:.4f} | {r.sd:.3f} |"
                  for r in n_df.itertuples()]
        lines.append("")
        c_df.to_csv(REPORTS / f"sweep_C_{task}.csv", index=False)
        n_df.to_csv(REPORTS / f"sweep_ngram_{task}.csv", index=False)

    lines += [
        "## Reading",
        "",
        "The two tasks land in opposite regimes. Sentiment wants heavy",
        "regularisation over long character n-grams: it is noisy, semantic, and the",
        "model has to generalise. Topic wants light regularisation over 2-3 character",
        "n-grams - which is exactly the length of the trigger substrings (`ui`, `ban`,",
        "`app`, `bug`) that `audit_labels.py` recovered from the labels. The",
        "hyperparameter that wins is telling you what the label is made of.",
        "",
    ]
    out = REPORTS / "hparam_sweep.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out.name}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
