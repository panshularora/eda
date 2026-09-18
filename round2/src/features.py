"""Feature blocks for the Social Engine NLP models.

Three complementary views of a post are concatenated into one sparse matrix:

``word``    TF-IDF over 1-2 grams of the normalised, negation-marked text.
            Carries topical and lexical polarity evidence.
``char``    TF-IDF over character 3-5 grams inside word boundaries. Robust to
            the misspellings, elongations and hashtag compounds that make
            social text hostile to a fixed word vocabulary - and, as the audit
            in ``audit_labels.py`` shows, it is the only view that can see the
            substring triggers behind ``topic_category``.
``stats``   Eleven hand-built surface counts taken from the *raw* string,
            before normalisation destroys them (capitalisation, punctuation
            runs, emoticon polarity, truncation).

All three are fitted inside the cross-validation loop, never before it.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from preprocess import prepare

_POS_EMOTE = re.compile(r"(:-?\)|:-?\]|:-?d|=\)|;-?\)|<3|:-?p)", re.I)
_NEG_EMOTE = re.compile(r"(:-?\(|:-?\[|=\(|</3|:-?/|:-?\|)", re.I)
_ELONG = re.compile(r"(\w)\1{2,}")
_ALLCAPS = re.compile(r"\b[A-Z]{3,}\b")

STAT_NAMES = [
    "n_chars", "n_words", "n_exclaim", "n_question", "n_punct_run",
    "n_allcaps", "n_elong", "n_hashtag", "n_mention", "emote_pos", "emote_neg",
]


class SurfaceStats(BaseEstimator, TransformerMixin):
    """Eleven surface statistics per post, read off the raw (un-normalised) text.

    These survive as a separate block because normalisation deliberately
    collapses them: ``!!!!!`` becomes ``!!`` and ``SHOUTING`` becomes lower
    case, yet both are cues an annotator used.
    """

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X):  # noqa: D102
        rows = np.empty((len(X), len(STAT_NAMES)), dtype=np.float64)
        for i, raw in enumerate(X):
            s = raw if isinstance(raw, str) else ""
            words = s.split()
            rows[i] = (
                len(s),
                len(words),
                s.count("!"),
                s.count("?"),
                len(re.findall(r"([!?])\1", s)),
                len(_ALLCAPS.findall(s)),
                len(_ELONG.findall(s)),
                s.count("#"),
                s.count("@"),
                len(_POS_EMOTE.findall(s)),
                len(_NEG_EMOTE.findall(s)),
            )
        # length features are unbounded; log1p keeps them on the same scale as
        # the counts so MaxAbsScaler does not squash everything else to zero.
        rows[:, 0] = np.log1p(rows[:, 0])
        rows[:, 1] = np.log1p(rows[:, 1])
        return rows

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return np.asarray(STAT_NAMES, dtype=object)


def word_block(ngram=(1, 2), min_df=2, max_features=None) -> TfidfVectorizer:
    """TF-IDF over normalised, negation-marked words."""
    return TfidfVectorizer(
        preprocessor=prepare,
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w'#]+\b",
        ngram_range=ngram,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def char_block(ngram=(3, 5), min_df=3, max_features=None) -> TfidfVectorizer:
    """TF-IDF over character n-grams bounded by word edges."""
    return TfidfVectorizer(
        preprocessor=prepare,
        analyzer="char_wb",
        ngram_range=ngram,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def stats_block() -> Pipeline:
    """Surface statistics, scaled into ``[-1, 1]`` to match the TF-IDF blocks."""
    return Pipeline([("stats", SurfaceStats()), ("scale", MaxAbsScaler())])


def build_features(
    use_word=True,
    use_char=True,
    use_stats=True,
    word_ngram=(1, 2),
    char_ngram=(3, 5),
    word_min_df=2,
    char_min_df=3,
    word_max_features=None,
    char_max_features=None,
    weights=None,
) -> FeatureUnion:
    """Assemble the requested blocks into a single `FeatureUnion`."""
    blocks = []
    if use_word:
        blocks.append(("word", word_block(word_ngram, word_min_df, word_max_features)))
    if use_char:
        blocks.append(("char", char_block(char_ngram, char_min_df, char_max_features)))
    if use_stats:
        blocks.append(("stats", stats_block()))
    if not blocks:
        raise ValueError("at least one feature block must be enabled")
    return FeatureUnion(blocks, transformer_weights=weights)
