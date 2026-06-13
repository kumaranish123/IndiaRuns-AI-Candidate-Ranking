"""Streamlit sandbox for the Intelligent Candidate Discovery & Ranking Engine.

Required by the submission spec (Section 10.5): a hosted environment where the
ranker can be run on a small candidate sample and produce a ranked CSV. It runs
the exact same `rank()` code path used to generate the full submission.

    streamlit run app.py
"""

import io
import os
import tempfile

import streamlit as st

from rank import rank
from src.io_utils import iter_candidates

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="*", layout="wide")

st.title("Intelligent Candidate Discovery & Ranking Engine")
st.caption(
    "Track 1 — Redrob / INDIA RUNS. Ranks candidates for the *Senior AI Engineer "
    "(Founding Team)* JD. CPU-only, no network, no LLM calls at ranking time."
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
            rows = rank(path, top=min(int(top_n), n_available), verbose=False)
        finally:
            if is_temp:
                os.remove(path)

        st.subheader("2. Ranked candidates")
        st.dataframe(
            [
                {"rank": r[1], "candidate_id": r[0], "score": round(r[2], 4), "reasoning": r[3]}
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )

        buf = io.StringIO()
        buf.write("candidate_id,rank,score,reasoning\n")
        import csv

        writer = csv.writer(buf)
        for cid, rnk, score, reasoning in rows:
            writer.writerow([cid, rnk, f"{score:.4f}", reasoning])
        st.download_button("Download ranked CSV", buf.getvalue(), file_name="ranking.csv", mime="text/csv")
