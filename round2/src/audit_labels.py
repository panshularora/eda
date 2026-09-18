"""Recover the function that produced ``topic_category``.

Why this file exists
--------------------
On ``topic_category`` a character-ngram model beats the same classifier on word
1-2 grams by more than 25 macro-F1 points (see
``reports/model_comparison_topic.csv``). A *character* view beating a *word*
view by that margin is not how topical language works; it is what happens when
the label depends on substrings rather than on words. The top-weighted
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
the topic task is exactly zero, and a high topic F1 is evidence of substring
recovery rather than of topical understanding. 13.9% of posts get a non-default
topic from a trigger, and three quarters of those triggers are buried inside an
unrelated word - ``happy`` -> Technical_Issues, ``band`` -> Account_Security,
``model`` -> Feature_Feedback - so about one post in ten carries a topic a human
reader would call wrong.
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
    # Python randomises string hashing per process, so Counter iteration order
    # is not stable across runs. Sort explicitly - purest first, then most
    # supported, then alphabetically - so the greedy search below breaks ties
    # the same way every time and the mined rule is reproducible.
    candidates.sort(key=lambda s: (-within[s] / overall[s], -overall[s], s))
    docs = {s: {i for i, t in enumerate(texts) if s in t} for s in candidates}
    target = {i for i, l in enumerate(labels) if l == cls}
    covered, chosen = set(), []
    while len(chosen) < max_keywords:
        best, gain = None, 0
        for s in candidates:
            g = len(docs[s] & target - covered)
            if g > gain:                     # strict >, so the sort decides ties
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


def _misfire_mask(df: pd.DataFrame) -> pd.Series:
    """True where a post's topic is decided *only* by buried substrings.

    A trigger that appears as a standalone word (``the app crashed``) is a
    defensible label. A trigger that appears only inside an unrelated word
    (``happy`` firing ``app``, ``band`` firing ``ban``, ``model`` firing
    ``mode``) is an artefact, and the post's topic is wrong to a human reader.
    """
    low = df[TEXT_COL].str.lower()
    assigned = apply_rule(df[TEXT_COL])
    mask = pd.Series(False, index=df.index)
    for cls in PRIORITY:
        in_class = assigned == cls
        if not in_class.any():
            continue
        buried_only = pd.Series(True, index=df.index)
        fired = pd.Series(False, index=df.index)
        for kw in RECOVERED_RULE[cls]:
            present = low.str.contains(kw, regex=False)
            standalone = low.str.contains(rf"\b{kw}\b", regex=True)
            fired |= present
            # any standalone occurrence of any trigger makes the label defensible
            buried_only &= ~(present & standalone)
        mask |= in_class & fired & buried_only
    return mask


def semantic_misfires(df: pd.DataFrame, limit=12) -> pd.DataFrame:
    """A sample of posts whose topic label is an artefact rather than a meaning."""
    low = df[TEXT_COL].str.lower()
    assigned = apply_rule(df[TEXT_COL])
    misfire = _misfire_mask(df)
    rows = []
    for cls in PRIORITY:
        sub = df[misfire & (assigned == cls)]
        for _, r in sub.head(limit).iterrows():
            text = r[TEXT_COL].lower()
            trigger = next((k for k in RECOVERED_RULE[cls] if k in text), "")
            rows.append({
                "trigger": trigger,
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

    assigned = apply_rule(df[TEXT_COL])
    misfire = _misfire_mask(df)
    result = {
        "blind_rediscovery": mined,
        "curated_rule": curated,
        # share of the corpus whose topic is decided by a trigger at all
        "trigger_rate": float((assigned != DEFAULT_CLASS).mean()),
        # share of the corpus whose topic is decided only by a buried substring
        "artefact_rate": float(misfire.mean()),
        # ... and the same as a share of the posts that got a non-default topic
        "artefact_share_of_triggered": float(
            misfire.sum() / max((assigned != DEFAULT_CLASS).sum(), 1)),
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
    print(f"posts whose topic is set by a trigger      : {result['trigger_rate']:.1%}")
    print(f"posts whose topic is a buried-substring artefact: "
          f"{result['artefact_rate']:.1%} of the corpus, "
          f"{result['artefact_share_of_triggered']:.1%} of the triggered posts")
    print(f"written: {out.name}, label_audit_misfires.csv")
    return result


if __name__ == "__main__":
    main()
