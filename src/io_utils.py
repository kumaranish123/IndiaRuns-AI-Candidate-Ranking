"""Streaming readers for the candidate pool and the submission writer."""

import csv
import gzip
import json
from pathlib import Path


def open_candidates(path):
    """Open a candidates file, transparently handling .jsonl and .jsonl.gz."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path):
    """Yield one candidate dict at a time.

    Supports both JSON Lines (one object per line, as in candidates.jsonl) and a
    plain JSON array (as in sample_candidates.json). JSON Lines is streamed so
    the 100K pool stays comfortably inside the 16 GB budget.
    """
    with open_candidates(path) as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return
        if first == "[":
            # Small JSON-array sample (e.g. sample_candidates.json): load whole.
            f.seek(0)
            for candidate in json.load(f):
                yield candidate
            return
        # JSON Lines: rewind and stream one object per line.
        f.seek(0)
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no}: {exc}") from exc


def write_submission(rows, out_path):
    """Write the ranked rows to a spec-compliant CSV.

    `rows` is an iterable of (candidate_id, rank, score, reasoning).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for candidate_id, rank, score, reasoning in rows:
            writer.writerow([candidate_id, rank, f"{score:.4f}", reasoning])
