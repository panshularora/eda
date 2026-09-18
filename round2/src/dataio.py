"""Loading and splitting - with the one guard that decides whether every
number downstream is honest.

1,100 of the 9,000 rows are exact repeats of text that already appears
elsewhere in the file (7,900 unique posts). Every repeat carries an identical
label, so a random row-level split puts the *same post* in train and test and
the model is graded partly on memorised strings. Splitting on the unique text
instead removes that channel. ``audit_split_leakage`` quantifies exactly how
much score the shortcut is worth, and the technical report quotes it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from config import ID_COL, RAW_CSV, SEED, TASKS, TEST_SIZE, TEXT_COL


@dataclass
class Split:
    """A train/test partition plus the group ids needed for grouped CV."""

    train: pd.DataFrame
    test: pd.DataFrame
    groups_train: np.ndarray

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.test)


def load(path=RAW_CSV) -> pd.DataFrame:
    """Read the labelled corpus and attach a ``text_group`` key."""
    df = pd.read_csv(path)
    missing = {ID_COL, TEXT_COL, *TASKS.values()} - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    # group = identity of the underlying post, so duplicates share a group
    df["text_group"] = df[TEXT_COL].astype("category").cat.codes
    return df


def sha256(path=RAW_CSV) -> str:
    """Checksum of the input file, recorded in every report for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duplicate_report(df: pd.DataFrame) -> dict:
    """Facts about repeated posts, including whether any repeat disagrees on a label."""
    grouped = df.groupby(TEXT_COL)
    sizes = grouped.size()
    conflicts = {
        task: int((grouped[col].nunique()[sizes > 1] > 1).sum())
        for task, col in TASKS.items()
    }
    return {
        "n_rows": int(len(df)),
        "n_unique_texts": int(df[TEXT_COL].nunique()),
        "n_duplicate_rows": int(len(df) - df[TEXT_COL].nunique()),
        "n_texts_repeated": int((sizes > 1).sum()),
        "max_repeats": int(sizes.max()),
        "label_conflicts_among_repeats": conflicts,
    }


def make_split(df: pd.DataFrame, task: str, test_size=TEST_SIZE, seed=SEED) -> Split:
    """Stratified split that keeps every copy of a post on the same side.

    Implemented by splitting the *unique texts* (stratified on their label,
    which is unambiguous because no repeated post disagrees with itself) and
    then pulling every row of the chosen texts.
    """
    col = TASKS[task]
    uniq = df.drop_duplicates(TEXT_COL)[[TEXT_COL, "text_group", col]]
    tr_groups, te_groups = train_test_split(
        uniq["text_group"].to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=uniq[col].to_numpy(),
    )
    te = set(te_groups.tolist())
    mask = df["text_group"].isin(te)
    train, test = df[~mask].copy(), df[mask].copy()
    assert not set(train["text_group"]) & set(test["text_group"]), "group leak"
    return Split(train, test, train["text_group"].to_numpy())


def grouped_cv(n_splits=5, seed=SEED) -> StratifiedGroupKFold:
    """Stratified k-fold that never splits a post across folds."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
