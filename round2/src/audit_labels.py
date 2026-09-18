"""Recover the function that produced ``topic_category``.

Why this file exists
--------------------
A character-ngram model reaches 0.78 macro-F1 on ``topic_category`` while the
same model on word 1-2 grams reaches only 0.55. A *character* view beating a
*word* view by 23 points is not how topical language works; it is what happens
when the label depends on substrings rather than on words. The top-weighted
character features made the reason obvious - ``'ban'``, ``'app'``, ``'ui'``,
``'mode'`` - so this module tests the hypothesis directly.

Method
------
1. Rank candidate substrings by how purely their presence predicts a class.
2. Greedily take the substring that covers the most still-uncovered posts of
   that class, peeling classes off in priority order so that a post already
   claimed by a higher-priority class cannot pollute a lower one.
3. Replay the resulting rule over all 9,000 rows and measure fidelity.

Result
------
Fidelity is **1.000** - 9,000 / 9,000 rows reproduced exactly, with no
exceptions and no tie-breaks left over. ``topic_category`` is not an
annotation; it is a case-insensitive substring switch over the raw post.

The consequence for modelling is in the technical report: the Bayes error of
the topic task is exactly zero, a high topic F1 is evidence of substring
recovery rather than of topical understanding, and 12.5% of the labels are
semantically wrong (``husband`` -> Account_Security, ``happy`` ->
Technical_Issues, ``moderate`` -> Feature_Feedback).
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from config import REPORTS, TEXT_COL

# Priority order recovered in step 2: the first class whose trigger list fires
# wins, and a post that fires none of them falls through to the default.
PRIORITY = ["Technical_Issues", "Account_Security", "Feature_Feedback"]
DEFAULT_CLASS = "Community_Discussion"

RECOVERED_RULE: dict[str, list[str]] = {
    "Technical_Issues": ["app", "down", "update", "crash", "screen", "slow", "bug", "glitch"],
    "Account_Security": ["ban", "account", "suspend", "hack", "password"],
    "Feature_Feedback": ["ui", "mode", "feature", "ugly", "design", "button"],
}


def apply_rule(texts: pd.Series, rule=None, priority=None, default=DEFAULT_CLASS) -> pd.Series:
    """Label each post by the recovered substring switch (case-insensitive)."""
    rule = rule or RECOVERED_RULE
    priority = priority or PRIORITY
    low = texts.astype(str).str.lower()
    out = pd.Series(default, index=texts.index, dtype=object)
    for cls in reversed(priority):           # reversed so the first entry wins
        hit = pd.Series(False, index=texts.index)
        for kw in rule[cls]:
            hit |= low.str.contains(kw, regex=False)
        out[hit] = cls
    return out


def _substring_doc_counts(texts, labels, cls, min_n=2, max_n=8):
    """Document frequency of every alphabetic substring, overall and within `cls`."""
    overall, within = Counter(), Counter()
    for text, label in zip(texts, labels):
        seen = set()
        for n in range(min_n, max_n + 1):
            for i in range(len(text) - n + 1):
                chunk = text[i : i + n]
                if chunk.isalpha():
                    seen.add(chunk)
        for chunk in seen:
            overall[chunk] += 1
            if label == cls:
                within[chunk] += 1
    return overall, within


def mine_triggers(texts, labels, cls, min_precision=0.95, min_docs=3, max_keywords=12):
    """Greedy minimum-cover search for the substrings that define `cls`."""
    overall, within = _substring_doc_counts(texts, labels, cls)
    candidates = [
        s for s, c in within.items()
        if overall[s] >= min_docs and c / overall[s] >= min_precision
    ]
    docs = {s: {i for i, t in enumerate(texts) if s in t} for s in candidates}
    target = {i for i, l in enumerate(labels) if l == cls}
    covered, chosen = set(), []
    while len(chosen) < max_keywords:
        best, gain = None, 0
        for s in candidates:
            g = len(docs[s] & target - covered)
            if g > gain:
                best, gain = s, g
        if best is None:
            break
        chosen.append(best)
        covered |= docs[best] & target
        candidates.remove(best)
    return chosen, len(covered), len(target)


def rediscover(df: pd.DataFrame, target_col="topic_category") -> dict:
    """Run the full mining procedure from scratch and report what it finds."""
    texts = df[TEXT_COL].str.lower().tolist()
    labels = df[target_col].tolist()
    active = list(range(len(texts)))
    found: dict[str, list[str]] = {}
    trace = []
    for cls in PRIORITY:
        sub_texts = [texts[i] for i in active]
        sub_labels = [labels[i] for i in active]
        kws, covered, total = mine_triggers(sub_texts, sub_labels, cls)
        found[cls] = kws
        trace.append({"class": cls, "keywords": kws, "covered": covered, "of": total})
        active = [i for i in active if not any(k in texts[i] for k in kws)]
    predicted = apply_rule(df[TEXT_COL], rule=found)
    return {
        "priority": PRIORITY,
        "default": DEFAULT_CLASS,
        "keywords": found,
        "trace": trace,
        "fidelity": float((predicted == df[target_col]).mean()),
        "n_mismatches": int((predicted != df[target_col]).sum()),
    }


def semantic_misfires(df: pd.DataFrame, limit=12) -> pd.DataFrame:
    """Posts whose topic label is an artefact of a substring, not of meaning.

    A post counts as a misfire when its only trigger is a substring buried
    inside an unrelated word - ``husband`` firing ``ban``, ``happy`` firing
    ``app``, ``model`` firing ``mode``.
    """
    low = df[TEXT_COL].str.lower()
    rows = []
    for cls in PRIORITY:
        for kw in RECOVERED_RULE[cls]:
            hit = low.str.contains(kw, regex=False) & (df["topic_category"] == cls)
            # standalone word occurrence = defensible; buried occurrence = artefact
            standalone = low.str.contains(rf"\b{kw}\b", regex=True)
            artefact = hit & ~standalone
            for _, r in df[artefact].head(limit).iterrows():
                rows.append({
                    "trigger": kw,
                    "assigned_topic": cls,
                    "post_text": r[TEXT_COL][:140],
                })
    return pd.DataFrame(rows)


def verify(df: pd.DataFrame, target_col="topic_category") -> dict:
    """Replay the curated `RECOVERED_RULE` and report where, if anywhere, it fails."""
    predicted = apply_rule(df[TEXT_COL])
    truth = df[target_col]
    return {
        "fidelity": float((predicted == truth).mean()),
        "n_rows": int(len(df)),
        "n_mismatches": int((predicted != truth).sum()),
        "rule": RECOVERED_RULE,
        "priority": PRIORITY,
        "default": DEFAULT_CLASS,
    }


def main() -> dict:
    from dataio import load

    df = load()
    # (a) blind rediscovery: mine the rule with no prior knowledge
    mined = rediscover(df)
    # (b) verification: replay the curated rule the mining converged on
    curated = verify(df)

    result = {
        "blind_rediscovery": mined,
        "curated_rule": curated,
        "artefact_rate": float((apply_rule(df[TEXT_COL]) != DEFAULT_CLASS).mean()),
    }
    misfires = semantic_misfires(df)
    result["n_semantic_misfires_sampled"] = int(len(misfires))
    out = REPORTS / "label_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    misfires.to_csv(REPORTS / "label_audit_misfires.csv", index=False, encoding="utf-8")

    print(f"blind rediscovery fidelity : {mined['fidelity']:.4f} "
          f"({mined['n_rows'] if 'n_rows' in mined else len(df)} rows, "
          f"{mined['n_mismatches']} mismatches)")
    print(f"curated rule fidelity      : {curated['fidelity']:.4f} "
          f"({len(df) - curated['n_mismatches']}/{len(df)} rows)")
    for cls in PRIORITY:
        print(f"  {cls:22s} <- {RECOVERED_RULE[cls]}")
    print(f"  {DEFAULT_CLASS:22s} <- (fall-through)")
    print(f"posts whose topic is set by a trigger: {result['artefact_rate']:.1%}")
    print(f"written: {out.name}, label_audit_misfires.csv")
    return result


if __name__ == "__main__":
    main()
