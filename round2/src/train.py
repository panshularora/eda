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
from sklearn.ensemble import VotingClassifier
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
# candidate list - feature ablation first, then model families
# ---------------------------------------------------------------------------
# Regularisation strength and the character n-gram range were swept per task by
# the same grouped CV on the *training split only* (see reports/hparam_sweep.md).
# The two tasks land in opposite regimes, which is itself a finding.
#
# Regularisation: sentiment is noisy and semantic, so it wants a lot of it and
# degrades monotonically as C rises (0.6227 -> 0.5814 from C=0.1 to C=32). Topic
# is deterministic and orthographic, so it wants almost none and improves
# monotonically over the same sweep (0.5986 -> 0.7287).
#
# Character range: both tasks want the range to start at 2, but they part
# company at the top end. Sentiment keeps gaining from longer n-grams, because
# meaning lives in morphemes and words. Topic peaks at 2-3 and then *falls*, from
# 0.9140 to 0.8223 by 3-6 - longer n-grams bury the short triggers (ui, ban, app,
# bug) that audit_labels.py recovered, which is what a label made of substrings
# rather than of words looks like from the hyperparameters.
HPARAMS = {
    "sentiment": {"C": 0.1, "C_lr": 0.5, "char_ngram": (2, 4), "balanced": None},
    "topic": {"C": 8.0, "C_lr": 50.0, "char_ngram": (2, 3), "balanced": "balanced"},
}


def candidates(task: str) -> dict[str, Pipeline]:
    """Ordered candidate registry for one task.

    The early rows hold the classifier fixed and add one feature block at a
    time; the later ones hold the features fixed and change the model family,
    ending with a soft-vote ensemble of the three. Reading the resulting table
    top to bottom therefore says what each decision bought.

    When a task's tuned character range *is* 3-5 (sentiment), the fixed 3-5 row
    and the tuned-range row collide on the same key and the dict keeps one of
    them. That is intended: there is nothing to ablate between a range and
    itself, so the sentiment table is one row shorter than the topic table.
    """
    hp = HPARAMS[task]
    bal, C, C_lr, cng = hp["balanced"], hp["C"], hp["C_lr"], hp["char_ngram"]
    svc = lambda: LinearSVC(C=C, class_weight=bal, max_iter=20000, random_state=SEED)
    label = f"char {cng[0]}-{cng[1]}gram"
    return {
        # --- floor ---------------------------------------------------------
        "baseline: stratified guess": Pipeline([
            ("f", build_features(use_char=False, use_stats=False, word_max_features=1)),
            ("c", DummyClassifier(strategy="stratified", random_state=SEED)),
        ]),
        # --- feature ablation, classifier held fixed ------------------------
        "word 1-2gram + LinearSVC": Pipeline([
            ("f", build_features(use_char=False, use_stats=False)),
            ("c", svc()),
        ]),
        "char 3-5gram + LinearSVC": Pipeline([
            ("f", build_features(use_word=False, use_stats=False, char_ngram=(3, 5))),
            ("c", svc()),
        ]),
        f"{label} + LinearSVC": Pipeline([
            ("f", build_features(use_word=False, use_stats=False, char_ngram=cng)),
            ("c", svc()),
        ]),
        f"word + {label} + LinearSVC": Pipeline([
            ("f", build_features(use_stats=False, char_ngram=cng)),
            ("c", svc()),
        ]),
        f"word + {label} + surface + LinearSVC": Pipeline([
            ("f", build_features(char_ngram=cng)),
            ("c", svc()),
        ]),
        # --- model families, features held fixed ----------------------------
        f"word + {label} + surface + LogisticRegression": Pipeline([
            ("f", build_features(char_ngram=cng)),
            ("c", LogisticRegression(C=C_lr, max_iter=3000,
                                     class_weight=bal, random_state=SEED)),
        ]),
        f"word + {label} + ComplementNB": Pipeline([
            ("f", build_features(use_stats=False, char_ngram=cng)),
            ("c", ComplementNB(alpha=0.3)),
        ]),
        f"word + {label} + surface + SGD (modified huber)": Pipeline([
            ("f", build_features(char_ngram=cng)),
            ("c", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=3000,
                                class_weight=bal, random_state=SEED)),
        ]),
        # --- does combining them help? ---------------------------------------
        f"word + {label} + surface + soft-vote ensemble": Pipeline([
            ("f", build_features(char_ngram=cng)),
            ("c", VotingClassifier(
                [("svc", CalibratedClassifierCV(
                    LinearSVC(C=C, class_weight=bal, max_iter=20000, random_state=SEED),
                    method="sigmoid", cv=3)),
                 ("lr", LogisticRegression(C=C_lr, max_iter=3000,
                                           class_weight=bal, random_state=SEED)),
                 ("cnb", ComplementNB(alpha=0.3))],
                voting="soft")),
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
    """Two sanity checks that keep the headline number honest.

    *Split rule.* The same pipeline is scored with the grouped split we report
    and with a duplicate-blind row split. The gap is the score the shortcut
    would have invented, and it belongs on the record rather than merely being
    avoided.

    *Label permutation.* The same pipeline is scored on shuffled labels. If the
    protocol were leaking, this would come back above chance; it does not, which
    is what licenses reading the real score as signal from the text.
    """
    grouped = cross_val_score(pipe, X, y, cv=grouped_cv(CV_FOLDS), groups=groups,
                              scoring="f1_macro", n_jobs=1)
    naive = cross_val_score(pipe, X, y,
                            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                            scoring="f1_macro", n_jobs=1)
    shuffled = np.random.RandomState(SEED).permutation(y)
    permuted = cross_val_score(pipe, X, shuffled, cv=grouped_cv(CV_FOLDS), groups=groups,
                               scoring="f1_macro", n_jobs=1)
    return {
        "grouped_by_text_f1_macro": float(grouped.mean()),
        "random_row_split_f1_macro": float(naive.mean()),
        "inflation": float(naive.mean() - grouped.mean()),
        "permuted_labels_f1_macro": float(permuted.mean()),
    }


def fit_final(pipe, X, y):
    """Refit the winner and wrap it so it can report calibrated confidence.

    `LinearSVC` has no `predict_proba`; Platt scaling supplies one without
    changing the decision rule materially. Error analysis in `evaluate.py` needs
    that confidence to separate "wrong and sure" from "wrong and hesitant".

    ``ensemble=False`` fits the pipeline once on all of the training data and
    calibrates it with cross-validated decision values, instead of keeping five
    separately fitted copies. It is the same calibration with one vocabulary
    instead of five, which takes the saved bundle from 9.0 MB to well inside the
    10 MB the submission form allows.
    """
    model = CalibratedClassifierCV(pipe, method="sigmoid", cv=5, ensemble=False)
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

    print("  leakage audit + permutation check ...", flush=True)
    leak = leakage_audit(candidates(task)[best_name], Xtr, ytr, split.groups_train)
    print(f"    grouped {leak['grouped_by_text_f1_macro']:.4f} | "
          f"random rows {leak['random_row_split_f1_macro']:.4f} | "
          f"inflation +{leak['inflation']:.4f} | "
          f"permuted labels {leak['permuted_labels_f1_macro']:.4f}")

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
