#!/usr/bin/env python3
"""Intelligent Candidate Discovery & Ranking Engine — entry point.

Reproduce the submission with:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs CPU-only, with no network calls and no GPU, using only the Python
standard library. The candidate pool is read once; each profile is tokenized a
single time and cached so the TF-IDF index and the scoring pass share the work.
A bounded heap keeps only the best `top` candidates in the ranking phase.
"""

import argparse
import heapq
import sys
import time

from src import reasoning, scoring
from src.io_utils import iter_candidates, write_submission
from src.jd_profile import JD_QUERY_TEXT
from src.text import TfidfSimilarity, candidate_text, tokenize
from src.validation import validity_multiplier


def _id_number(candidate_id):
    try:
        return int(candidate_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def rank(candidates_path, top=100, verbose=True):
    start = time.time()
    if verbose:
        print("Loading + indexing candidates...", file=sys.stderr)

    # Read once; tokenize once. Cache (candidate, tokens) so the index build and
    # the scoring pass reuse the same tokenization instead of repeating it.
    cached = []
    tfidf = TfidfSimilarity()
    for candidate in iter_candidates(candidates_path):
        text_lower = candidate_text(candidate).lower()
        tokens = tokenize(text_lower)
        tfidf.add_document(tokens)
        cached.append((candidate, tokens, text_lower))
    tfidf.finalize()
    n = len(cached)
    query_vec = tfidf.build_query(tokenize(JD_QUERY_TEXT))

    if verbose:
        print(f"  indexed {n} candidates in {time.time() - start:.1f}s", file=sys.stderr)
        print("Scoring candidates...", file=sys.stderr)

    heap = []  # min-heap of (score, -id_number, candidate_id, candidate, detail)
    for candidate, tokens, text_lower in cached:
        cid = candidate.get("candidate_id")
        cosine = tfidf.cosine(query_vec, tokens)
        validity = validity_multiplier(candidate)
        score, detail = scoring.score_candidate(candidate, cosine, validity, text_lower)

        entry = (score, -_id_number(cid), cid, candidate, detail)
        if len(heap) < top:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    # Best first; ties broken by candidate_id ascending, rounded to match output.
    ranked = sorted(heap, key=lambda e: (-round(e[0], 4), e[2]))

    rows = []
    for rank_pos, (score, _neg_id, cid, candidate, detail) in enumerate(ranked, start=1):
        rows.append((cid, rank_pos, score, reasoning.build_reasoning(candidate, detail)))

    if verbose:
        print(f"Done: ranked top {len(rows)} of {n} in {time.time() - start:.1f}s", file=sys.stderr)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path (default: submission.csv)")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to rank (default: 100)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args(argv)

    rows = rank(args.candidates, top=args.top, verbose=not args.quiet)
    write_submission(rows, args.out)
    if not args.quiet:
        print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
