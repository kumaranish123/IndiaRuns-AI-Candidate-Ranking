"""Streamlit sandbox for the Intelligent Candidate Discovery & Ranking Engine.

Required by the submission spec (Section 10.5): a hosted environment where the
ranker can be run on a small candidate sample and produce a ranked CSV. It runs
the exact same `rank()` code path used to generate the full submission.

    streamlit run app.py
"""

import csv
import io
import os
import tempfile
import time

import streamlit as st

from rank import rank
from src.io_utils import iter_candidates

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="\U0001F50D",
    layout="wide",
)

st.title("\U0001F50D Intelligent Candidate Discovery & Ranking Engine")
st.caption(
    "Track 1 — Redrob / INDIA RUNS. Ranks candidates for the *Senior AI Engineer "
    "(Founding Team)* JD. CPU-only, no network, no LLM calls at ranking time."
)

with st.expander("What this JD actually wants (and the traps in the data)"):
    st.markdown(
        "**Looking for:** 5-9 yrs (6-8 ideal), 4-5 of them **applied ML at product "
        "companies**; hands-on **retrieval / vector search / ranking / recommendation**; "
        "strong evaluation instinct (NDCG / MRR); Pune / Noida or Tier-1 India.\n\n"
        "**Explicit negatives:** AI keywords with the wrong job title, title-chasing "
        "job-hoppers, consulting-only careers, pure-research-no-production, and "
        "computer-vision/speech profiles with no NLP/IR. The pool also hides ~80 "
        "**honeypots** - internally impossible profiles - which we detect and bury."
    )

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "- **Role gate**: is this person actually an ML/AI/SWE practitioner, "
        "or a keyword-stuffer with the wrong job title?\n"
        "- **Domain match**: retrieval / ranking / search / recsys / NLP evidence "
        "from career text + a TF-IDF similarity to the JD.\n"
        "- **Experience, company type, location, education.**\n"
        "- **Behavioral multiplier**: recency, recruiter response, notice period.\n"
        "- **Validity check**: internally impossible profiles (honeypots) are "
        "pushed to the bottom."
    )
    top_n = st.number_input("How many to rank", min_value=1, max_value=100, value=20)
    st.divider()
    st.markdown(
        "[GitHub repo](https://github.com/kumaranish123/IndiaRuns-AI-Candidate-Ranking)"
    )

st.subheader("1. Provide a candidate sample (<= 100)")
uploaded = st.file_uploader("Upload candidates (.json array or .jsonl)", type=["json", "jsonl"])

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_candidates.json")
use_sample = st.checkbox("Use the bundled sample_candidates.json", value=not uploaded)


def _ranker_input():
    if uploaded is not None:
        suffix = ".jsonl" if uploaded.name.endswith(".jsonl") else ".json"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded.getbuffer())
        return path, True
    if use_sample and os.path.exists(SAMPLE):
        return SAMPLE, False
    return None, False


if st.button("Rank candidates", type="primary"):
    path, is_temp = _ranker_input()
    if not path:
        st.error("Upload a file or enable the bundled sample.")
    else:
        try:
            n_available = sum(1 for _ in iter_candidates(path))
            with st.spinner(f"Scoring {n_available} candidates..."):
                started = time.perf_counter()
                rows = rank(path, top=min(int(top_n), n_available), verbose=False)
                elapsed = time.perf_counter() - started
        finally:
            if is_temp:
                os.remove(path)

        st.subheader("2. Ranked candidates")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates scored", f"{n_available:,}")
        c2.metric("Shortlisted", len(rows))
        c3.metric("Top fit score", f"{rows[0][2]:.3f}" if rows else "-")
        c4.metric("Scoring time", f"{elapsed:.2f}s")

        st.dataframe(
            [
                {"Rank": r[1], "Candidate": r[0], "Fit score": round(r[2], 4), "Why this rank": r[3]}
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Fit score": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.4f"
                ),
                "Why this rank": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption(
            "Every line in **Why this rank** is generated only from facts in the "
            "profile - no skill is named that the candidate does not list."
        )

        buf = io.StringIO()
        buf.write("candidate_id,rank,score,reasoning\n")
        writer = csv.writer(buf)
        for cid, rnk, score, reasoning in rows:
            writer.writerow([cid, rnk, f"{score:.4f}", reasoning])
        st.download_button(
            "Download ranked CSV", buf.getvalue(), file_name="ranking.csv", mime="text/csv"
        )
