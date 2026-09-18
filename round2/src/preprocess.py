r"""Text normalisation for the Social Engine corpus.

The corpus is social-media text that has been through at least one bad
serialisation round-trip, so the raw strings carry three kinds of damage that
have nothing to do with meaning:

  1. literal JSON escapes that survived as text  (``,`` for a comma,
     ``’`` for a curly apostrophe, ``\"`` for a quote);
  2. CSV quote wrapping that leaked into the value (``"...."``);
  3. platform noise - ``@user`` placeholders, URLs, ``RT``, and the ``...``
     left behind where the crawler truncated a post.

`normalise` repairs 1-3 and then canonicalises the parts of the surface form
that are noise for a bag-of-ngrams model (digits, repeated characters) while
*preserving* the parts that carry sentiment (emoticons, ``!``/``?`` runs,
hashtag bodies).

Nothing is deleted that a human annotator would have used to pick the label.
"""
from __future__ import annotations

import codecs
import html
import re
import unicodedata

# --------------------------------------------------------------------------
# 1. de-serialisation damage
# --------------------------------------------------------------------------
_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")
_ESCAPED_QUOTE = re.compile(r'\\+"')
_REPEATED_QUOTE = re.compile(r'"{2,}')

# --------------------------------------------------------------------------
# 2. platform noise
# --------------------------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_MENTION = re.compile(r"@\w+")
_RT = re.compile(r"^\s*RT\b[: ]*", re.I)
_TRUNCATION = re.compile(r"\s*\.{3,}\s*$")
_NUMBER = re.compile(r"\b\d[\d,.:/]*\b")
_ELONGATION = re.compile(r"(\w)\1{2,}")          # sooooo -> soo
_PUNCT_RUN = re.compile(r"([!?.])\1{1,}")        # !!!! -> !!  (run kept, length capped)
_SPACE = re.compile(r"\s+")

# Emoticons are the highest-precision sentiment cue in the corpus (":(" is 85%
# Negative), so they are mapped to opaque single tokens before punctuation is
# touched - otherwise the tokeniser throws them away.
_EMOTICONS = [
    (re.compile(r"</3"), " emotebroken "),
    (re.compile(r"<3"), " emoteheart "),
    (re.compile(r";-?\)"), " emotewink "),
    (re.compile(r"(?::|=)-?(?:\)|\]|>|d|3)", re.I), " emotesmile "),
    (re.compile(r"(?::|=)-?(?:\(|\[|<|c)", re.I), " emotefrown "),
    (re.compile(r"(?::|=)-?(?:p|b)", re.I), " emotetongue "),
    (re.compile(r"(?::|=)-?(?:\||/)"), " emotemeh "),
]

_HASHTAG = re.compile(r"#(\w+)")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _unescape(text: str) -> str:
    """Undo the serialisation damage described in the module docstring."""
    if "\\u" in text:
        text = _UNICODE_ESCAPE.sub(
            lambda m: codecs.decode(m.group(0), "unicode_escape"), text
        )
    text = _ESCAPED_QUOTE.sub('"', text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    # CSV quote wrapping that leaked into the value
    while len(text) > 1 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    text = text.lstrip('"').rstrip('"')
    return _REPEATED_QUOTE.sub('"', text)


def split_hashtag(tag: str) -> str:
    """``#GoodFriday`` -> ``# good friday``; the body often carries the polarity."""
    return "# " + _CAMEL.sub(" ", tag).replace("_", " ").lower()


def normalise(text: str) -> str:
    """Full normalisation pipeline. Returns lower-cased, whitespace-collapsed text."""
    if not isinstance(text, str):
        return ""
    text = _unescape(text)
    text = _RT.sub("", text)
    text = _TRUNCATION.sub(" trunctoken ", text)
    text = _URL.sub(" urltoken ", text)
    text = _MENTION.sub(" usertoken ", text)
    for pattern, token in _EMOTICONS:
        text = pattern.sub(token, text)
    text = _HASHTAG.sub(lambda m: " " + split_hashtag(m.group(1)) + " ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = _PUNCT_RUN.sub(r"\1\1", text)
    text = _ELONGATION.sub(r"\1\1", text)
    text = text.lower()
    return _SPACE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# 3. negation scope
# --------------------------------------------------------------------------
_NEGATORS = {
    "not", "no", "never", "none", "cannot", "cant", "can't", "dont", "don't",
    "doesnt", "doesn't", "didnt", "didn't", "isnt", "isn't", "arent", "aren't",
    "wasnt", "wasn't", "werent", "weren't", "wont", "won't", "wouldnt",
    "wouldn't", "shouldnt", "shouldn't", "couldnt", "couldn't", "aint",
    "ain't", "nor", "neither", "nothing", "nowhere", "hardly", "barely",
    "without",
}
_CLAUSE_END = set(".,;:!?")


def mark_negation(text: str, window: int = 4) -> str:
    """Append ``_neg`` to up to ``window`` tokens after a negator.

    ``not a good day`` and ``a good day`` share every unigram; without this the
    linear model sees identical evidence for opposite labels. Clause-ending
    punctuation closes the scope, which stops the marker leaking across
    sentence boundaries.
    """
    out, countdown = [], 0
    for token in text.split():
        if countdown > 0 and token not in _NEGATORS:
            out.append(token + "_neg")
            countdown -= 1
        else:
            out.append(token)
        if token in _NEGATORS:
            countdown = window
        elif token and token[-1] in _CLAUSE_END:
            countdown = 0
    return " ".join(out)


def prepare(text: str) -> str:
    """`normalise` + negation marking - the single entry point used by the models."""
    return mark_negation(normalise(text))
