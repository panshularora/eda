"""Model selection and training for both Social Engine tasks.

What this script does, in order:

1. Split each task on the *unique post* (``dataio.make_split``), holding out
   20% that the model search never sees.
2. Score a fixed candidate list with 5-fold **grouped** stratified CV on the
   training part only. The list is arranged as an ablation - each feature
   block is added on its own before any model family is varied - so the
   comparison table answers "what did this choice buy?" rather than just
   "which number is biggest".
3. Quantify how much score a row-level split would have handed us for free.
4. Refit the winner on the whole training part, calibrate it so the system can
   report a confidence with every prediction, and score it once on the
   held-out 20%.
5. Save one bundle containing both task models, their metadata and the
   recovered topic rule.

Run:  ``python train.py``           (both tasks)
      ``python train.py sentiment`` (one task)
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from audit_labels import RECOVERED_RULE, PRIORITY, DEFAULT_CLASS
from config import CV_FOLDS, MODEL_BUNDLE, REPORTS, SEED, TASKS, TEXT_COL
from dataio import duplicate_report, grouped_cv, load, make_split, sha256
from features import build_features

# ---------------------------------------------------------------------------
# candidate list - ablation first, then model families
# ---------------------------------------------------------------------------
def candidates(task: str) -> dict[str, Pipeline]:
    """Ordered candidate registry. Identical for both tasks so the two
    comparison tables can be read side by side."""
    balanced = "balanced" if task == "topic" else None
    return {
        # --- floor ---------------------------------------------------------
        "baseline: stratified guess": Pipeline([
            ("f", build_features(use_char=False, use_stats=False, word_max_features=1)),
            ("c", DummyClassifier(strategy="stratified", random_state=SEED)),
        ]),
        # --- feature ablation, model held fixed at LinearSVC ----------------
        "word 1-2gram + LinearSVC": Pipeline([
            ("f", build_features(use_char=False, use_stats=False)),
            ("c", LinearSVC(C=1.0, class_weight=balanced, random_state=SEED)),
        ]),
        "char 3-5gram + LinearSVC": Pipeline([
            ("f", build_features(use_word=False, use_stats=False)),
            ("c", LinearSVC(C=1.0, class_weight=balanced, random_state=SEED)),
        ]),
        "word + char + LinearSVC": Pipeline([
            ("f", build_features(use_stats=False)),
            ("c", LinearSVC(C=1.0, class_weight=balanced, random_state=SEED)),
        ]),
        "word + char + surface + LinearSVC": Pipeline([
            ("f", build_features()),
            ("c", LinearSVC(C=1.0, class_weight=balanced, random_state=SEED)),
        ]),
        # --- model families, features held fixed ----------------------------
        "word + char + surface + LogisticRegression": Pipeline([
            ("f", build_features()),
            ("c", LogisticRegression(C=10.0, max_iter=3000,
                                     class_weight=balanced, random_state=SEED)),
        ]),
        "word + char + surface + ComplementNB": Pipeline([
            ("f", build_features(use_stats=False)),
            ("c", ComplementNB(alpha=0.3)),
        ]),
        "word + char + surface + SGD (modified huber)": Pipeline([
            ("f", build_features()),
            ("c", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=3000,
                                class_weight=balanced, random_state=SEED)),
        ]),
    }


def score_candidates(X, y, groups, task: str) -> pd.DataFrame:
    """Grouped 5-fold CV for every candidate; returns a tidy comparison table."""
    cv = grouped_cv(CV_FOLDS)
    rows = []
    for name, pipe in candidates(task).items():
        t0 = time.time()
        cvres = cross_validate(
            pipe, X, y, cv=cv, groups=groups,
            scoring=("f1_macro", "accuracy", "f1_weighted"), n_jobs=1,
        )
        rows.append({
            "model": name,
            "cv_f1_macro": cvres["test_f1_macro"].mean(),
            "cv_f1_macro_std": cvres["test_f1_macro"].std(),
            "cv_f1_weighted": cvres["test_f1_weighted"].mean(),
            "cv_accuracy": cvres["test_accuracy"].mean(),
            "fit_seconds": round(time.time() - t0, 1),
        })
        print(f"    {name:46s} macro-F1 {rows[-1]['cv_f1_macro']:.4f} "
              f"+- {rows[-1]['cv_f1_macro_std']:.3f}", flush=True)
    return pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)


def leakage_audit(pipe, X, y, groups) -> dict:
    """Same pipeline, same folds count, two splitting rules.

    The gap is the score a row-level split would have invented, and it is the
    reason every number in this submission uses the grouped split.
    """
    grouped = cross_val_score(pipe, X, y, cv=grouped_cv(CV_FOLDS), groups=groups,
                              scoring="f1_macro", n_jobs=1)
    naive = cross_val_score(pipe, X, y,
                            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                            scoring="f1_macro", n_jobs=1)
    return {
        "grouped_by_text_f1_macro": float(grouped.mean()),
        "random_row_split_f1_macro": float(naive.mean()),
        "inflation": float(naive.mean() - grouped.mean()),
    }


def fit_final(pipe, X, y):
    """Refit the winner and wrap it so it can report calibrated confidence.

    `LinearSVC` has no `predict_proba`; Platt scaling over 5 internal folds
    supplies one without changing the decision rule materially. Error analysis
    in `evaluate.py` needs that confidence to separate "wrong and sure" from
    "wrong and hesitant".
    """
    model = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
    model.fit(X, y)
    return model


def train_task(df: pd.DataFrame, task: str) -> dict:
    col = TASKS[task]
    print(f"\n=== {task}  ({col}) ===")
    split = make_split(df, task)
    Xtr, ytr = split.train[TEXT_COL].to_numpy(), split.train[col].to_numpy()
    Xte, yte = split.test[TEXT_COL].to_numpy(), split.test[col].to_numpy()
    print(f"  train {len(Xtr)} rows / {split.train['text_group'].nunique()} unique posts")
    print(f"  test  {len(Xte)} rows / {split.test['text_group'].nunique()} unique posts")

    table = score_candidates(Xtr, ytr, split.groups_train, task)
    best_name = table.iloc[0]["model"]
    best_pipe = candidates(task)[best_name]
    print(f"  winner: {best_name}")

    print("  leakage audit (grouped vs random row split) ...", flush=True)
    leak = leakage_audit(candidates(task)[best_name], Xtr, ytr, split.groups_train)
    print(f"    grouped {leak['grouped_by_text_f1_macro']:.4f} | "
          f"random rows {leak['random_row_split_f1_macro']:.4f} | "
          f"inflation +{leak['inflation']:.4f}")

    print("  refitting on full training split and calibrating ...", flush=True)
    model = fit_final(best_pipe, Xtr, ytr)

    table.to_csv(REPORTS / f"model_comparison_{task}.csv", index=False)
    return {
        "task": task,
        "target_column": col,
        "model": model,
        "selected": best_name,
        "comparison": table,
        "leakage": leak,
        "split": split,
        "classes": sorted(pd.unique(df[col]).tolist()),
    }


def main(tasks=None) -> dict:
    tasks = tasks or list(TASKS)
    df = load()
    print(f"corpus: {len(df)} rows, sha256 {sha256()[:16]}...")
    dup = duplicate_report(df)
    print(f"  unique posts {dup['n_unique_texts']}, duplicate rows "
          f"{dup['n_duplicate_rows']}, label conflicts "
          f"{dup['label_conflicts_among_repeats']}")

    trained = {t: train_task(df, t) for t in tasks}

    bundle = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "team": "Team SE7EN (Tanmay Singh, Panshul Arora)",
        "event": "Data Vortex A'26 Round 2",
        "dataset_sha256": sha256(),
        "seed": SEED,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "duplicates": dup,
        "models": {t: trained[t]["model"] for t in trained},
        "selected": {t: trained[t]["selected"] for t in trained},
        "classes": {t: trained[t]["classes"] for t in trained},
        "leakage": {t: trained[t]["leakage"] for t in trained},
        "topic_label_rule": {
            "priority": PRIORITY, "default": DEFAULT_CLASS,
            "keywords": RECOVERED_RULE, "fidelity_on_training_file": 1.0,
        },
    }
    joblib.dump(bundle, MODEL_BUNDLE, compress=3)
    size_mb = MODEL_BUNDLE.stat().st_size / 1e6
    print(f"\nsaved {MODEL_BUNDLE.name} ({size_mb:.2f} MB)")

    summary = {t: {
        "selected": trained[t]["selected"],
        "cv_f1_macro": float(trained[t]["comparison"].iloc[0]["cv_f1_macro"]),
        "leakage": trained[t]["leakage"],
    } for t in trained}
    (REPORTS / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return trained


if __name__ == "__main__":
    main(sys.argv[1:] or None)
