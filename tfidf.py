"""
tfidf.py
========
A from-scratch TF-IDF vectoriser and cosine-similarity function — no numpy,
no scikit-learn. This is the "semantic-ish" overlap score: how much of the job
description's important vocabulary actually appears in the resume.

TF-IDF intuition
----------------
* TF (term frequency): how often a word appears in a document, length-normalised.
* IDF (inverse document frequency): rare words across the corpus carry more
  signal than words that appear everywhere. idf(t) = ln((1 + N) / (1 + df)) + 1
  (the smoothed form, so a term present in every document still has idf > 0).
* A document becomes a vector of tf * idf weights; cosine similarity measures
  the angle between two such vectors, ignoring document length.
"""

from __future__ import annotations

import math
from collections import Counter

from text_utils import content_tokens


def term_frequencies(tokens: list[str]) -> dict[str, float]:
    """Length-normalised term frequencies for one document."""
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {term: c / total for term, c in counts.items()}


def inverse_document_frequencies(docs_tokens: list[list[str]]) -> dict[str, float]:
    """Smoothed IDF computed across a corpus of tokenised documents."""
    n_docs = len(docs_tokens)
    doc_freq: Counter[str] = Counter()
    for tokens in docs_tokens:
        for term in set(tokens):
            doc_freq[term] += 1
    return {
        term: math.log((1 + n_docs) / (1 + df)) + 1.0
        for term, df in doc_freq.items()
    }


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF weight vector (sparse dict) for one document."""
    tf = term_frequencies(tokens)
    return {term: freq * idf.get(term, 0.0) for term, freq in tf.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse weight vectors, in [0, 1]."""
    if not vec_a or not vec_b:
        return 0.0
    # Iterate over the smaller vector for the dot product.
    small, large = (vec_a, vec_b) if len(vec_a) <= len(vec_b) else (vec_b, vec_a)
    dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def text_similarity(resume_text: str, jd_text: str) -> float:
    """Convenience: TF-IDF cosine similarity between two raw texts, in [0, 1]."""
    resume_tokens = content_tokens(resume_text)
    jd_tokens = content_tokens(jd_text)
    idf = inverse_document_frequencies([resume_tokens, jd_tokens])
    v_resume = tfidf_vector(resume_tokens, idf)
    v_jd = tfidf_vector(jd_tokens, idf)
    return cosine_similarity(v_resume, v_jd)
