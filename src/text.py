"""Lightweight, dependency-free TF-IDF cosine similarity.

We avoid heavy ML libraries on purpose: a sparse TF-IDF over the candidate's
free text against a fixed JD query vector is fast (the whole 100K pool scores
in a few seconds on one CPU core), fully offline, and trivially reproducible.
It supplies the "semantic-ish" retrieval signal in the hybrid scorer without
needing a GPU, a model download, or any network call during ranking.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
_STOP = frozenset(
    """a an the and or of to in for on with at by from as is are was were be been
    this that these those it its i we you they our your their my his her using used
    use work worked working built build building team teams across most some few
    into out over under more less very also etc""".split()
)


def tokenize(text):
    """Lowercase word tokens, keeping tech tokens like c++, .net, a/b intact-ish."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def candidate_text(candidate):
    """Assemble the free text used for lexical matching against the JD."""
    profile = candidate.get("profile", {})
    parts = [profile.get("headline", ""), profile.get("summary", ""), profile.get("current_title", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    parts.extend(skill.get("name", "") for skill in candidate.get("skills", []))
    return " ".join(p for p in parts if p)


class TfidfSimilarity:
    """Two-phase TF-IDF: accumulate document frequencies, then score docs."""

    def __init__(self):
        self._df = Counter()
        self._n_docs = 0
        self._idf = None

    def add_document(self, tokens):
        self._n_docs += 1
        for term in set(tokens):
            self._df[term] += 1

    def finalize(self):
        n = self._n_docs
        self._idf = {
            term: math.log((n + 1) / (df + 1)) + 1.0
            for term, df in self._df.items()
        }
        return self

    def _vector(self, tokens):
        counts = Counter(tokens)
        vec = {}
        for term, count in counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            vec[term] = (1.0 + math.log(count)) * idf
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm > 0:
            for term in vec:
                vec[term] /= norm
        return vec

    def build_query(self, tokens):
        return self._vector(tokens)

    def cosine(self, query_vec, tokens):
        if not query_vec:
            return 0.0
        doc_vec = self._vector(tokens)
        if not doc_vec:
            return 0.0
        # iterate the smaller dict
        if len(query_vec) < len(doc_vec):
            small, large = query_vec, doc_vec
        else:
            small, large = doc_vec, query_vec
        return sum(weight * large.get(term, 0.0) for term, weight in small.items())
