"""
text_utils.py
=============
Lightweight, dependency-free text processing: normalisation, tokenisation,
stop-word removal and n-gram phrase extraction. Everything here works on plain
Python strings so the whole project runs on the standard library alone.
"""

from __future__ import annotations

import re

# A compact English stop-word list — enough to keep TF-IDF focused on the
# content words that actually distinguish one document from another.
STOPWORDS: frozenset[str] = frozenset("""
a an the and or but if then else for to of in on at by with from as is are was
were be been being this that these those it its we you they he she i our your
their his her my me us them will would can could should shall may might must
do does did done have has had not no nor so than too very just about into over
under again further once here there all any both each few more most other some
such only own same up down out off above below who whom which what when where why
how am also per via etc using use used work working ability able strong excellent
good great new role position candidate team company including include includes
""".split())

# Keep tokens like c++, c#, ci/cd, node.js, react.js, rest-api intact.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\+\#\./\-]*[a-z0-9\+\#]|[a-z0-9]")


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    """Split text into lower-cased content tokens (symbols like + # / kept)."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stop-words and 1-char noise removed (except c, r, x)."""
    keep_single = {"c", "r"}
    out = []
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        if len(tok) == 1 and tok not in keep_single:
            continue
        out.append(tok)
    return out


def ngrams(tokens: list[str], n: int) -> list[str]:
    """All contiguous n-word phrases, space-joined."""
    if n <= 1:
        return list(tokens)
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def phrase_set(text: str, max_n: int = 3) -> set[str]:
    """Set of all 1..max_n-grams over the *raw* token stream.

    Built from the full token stream (stop-words included) so multi-word skill
    phrases like "natural language processing" survive intact.
    """
    toks = tokenize(text)
    phrases: set[str] = set()
    for n in range(1, max_n + 1):
        phrases.update(ngrams(toks, n))
    return phrases
