# Intelligent Candidate Discovery & Ranking Engine

**INDIA RUNS — Track 1 (Data & AI Challenge), Redrob × Hack2skill.**

**Live demo:** https://indiaruns-ai-candidate-ranking-1.streamlit.app/

**Video walkthrough:** https://drive.google.com/file/d/1brIbPLqXPTJVZui1jWBqWmZ3JqQgbmxX/view?usp=sharing

Given the released *Senior AI Engineer (Founding Team)* job description and a pool of
100,000 candidate profiles, this system produces an explainable, ranked list of the
top 100 best-fit candidates.

It runs **CPU-only, with no network access and no LLM calls at ranking time**, using
**only the Python standard library**. The full 100K pool ranks in about **3 minutes**
on a 16 GB laptop — well inside the challenge's 5-minute budget.

---

## TL;DR — reproduce the submission

```bash
# 1. Put the dataset where the command expects it
#    (download candidates.jsonl from the hackathon bundle)
mkdir -p data
cp /path/to/candidates.jsonl data/candidates.jsonl

# 2. Produce the ranked top-100 CSV
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 3. Validate against the official format checker shipped in the challenge bundle
python validate_submission.py submission.csv   # -> "Submission is valid."
```

The core ranker needs **no third-party packages**. The only dependency
(`requirements.txt`) is `streamlit`, used purely for the optional demo UI.

---

## The problem, and why naive matching fails

The JD is written in prose and is explicit that the obvious approach is a trap:

> *"The 'right answer' is not 'find candidates whose skills section contains the most
> AI keywords.' That's a trap we've explicitly built into the dataset."*

The dataset reflects this. Of 100K candidates, most have **decoy job titles**
(Marketing Manager, HR Manager, Mechanical Engineer, …) — many with AI keywords stuffed
into their skills list. Genuine fits (ML Engineer, Recommendation Systems Engineer, NLP
Engineer, …) are a small minority. The pool also contains:

- **Keyword stuffers** — AI skills, wrong job. (Down-ranked by the role gate.)
- **Plain-language Tier-5s** — built a real recsys/search system at a product company but
  never wrote "RAG" or "Pinecone". (Surfaced by reading career text, not just skills.)
- **~80 honeypots** — subtly impossible profiles (e.g. "expert in 10 skills with 0 years
  used"). Ranking these in the top 10 is a tell; >10% in the top 100 is an auto-DQ.

So the system has to reason about **what the JD means**, weigh **behavioral availability**,
and **validate profiles** for internal consistency.

---

## How it scores a candidate

```
final_score = content_fit  x  profile_validity  x  behavioral_availability
```

**`content_fit`** — weighted blend (weights in `src/scoring.py`):

| Component | Weight | What it captures |
|---|---|---|
| Role / title fit | 0.28 | Is this an actual ML/AI/SWE practitioner? Decisive gate vs keyword stuffers. |
| Domain fit | 0.24 | Retrieval / ranking / search / recsys / NLP evidence from career text **+ TF-IDF similarity to the JD**. |
| Skill fit | 0.14 | Relevant skills, trust-weighted by proficiency × endorsements × duration, **validated against platform assessment scores**. |
| Experience fit | 0.14 | 6-8 yrs ideal, 5-9 accepted; juniors discounted. |
| Company / career fit | 0.11 | Product vs services; anti-job-hopping; consulting-only penalty; open-source bonus. |
| Location fit | 0.05 | Pune/Noida preferred, Tier-1 India welcome, relocation considered. |
| Education fit | 0.04 | Institution tier (minor). |

**`profile_validity`** — internal-consistency checks (`src/validation.py`). Impossible
profiles (single role longer than total career, multiple "expert" skills with 0 months of
use, education running backwards, …) get a near-zero multiplier, which keeps honeypots out.

**`behavioral_availability`** — multiplier in `[0.5, 1.0]` from recency of last activity,
recruiter response rate, open-to-work, interview completion, notice period and recruiter
demand. A perfect-on-paper candidate who's been inactive for months and never replies is
not actually hireable, so their score is halved.

Every clause in the CSV `reasoning` column is generated from facts already extracted for
scoring — no skill is named that isn't in the profile, and the stated concern is always
consistent with the rank.

---

## Project layout

```
rank.py                  # entry point: python rank.py --candidates ... --out ...
app.py                   # Streamlit sandbox (demo on a small sample)
sample_candidates.json   # small sample so the sandbox works out of the box
submission_metadata.yaml # portal metadata (fill in placeholders)
requirements.txt
src/
  io_utils.py            # streaming reader (.jsonl / .jsonl.gz / .json array) + CSV writer
  jd_profile.py          # structured interpretation of the JD (the "what it means")
  text.py                # dependency-free TF-IDF cosine similarity
  features.py            # per-component feature extraction
  validation.py          # honeypot / impossible-profile detection
  scoring.py             # combines components into the final score
  reasoning.py           # grounded reasoning-string builder
tests/
  test_ranker.py         # unit + end-to-end tests (no dataset needed)
docs/
  IDEA_SUBMISSION.md     # filled idea-submission template
```

---

## Sandbox / demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a small candidate sample (`.json` array or `.jsonl`, ≤ 100) or use the bundled
`sample_candidates.json`, and the app runs the **same `rank()` code path** and lets you
download the ranked CSV.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers honeypot detection, the keyword-stuffer role gate, skill trust weighting, and an
end-to-end ranking-order check — all on inline fixtures, so no dataset download is needed.

---

## Compute constraints (met)

- **CPU only**, no GPU.
- **No network** during ranking — no hosted LLM/API calls.
- **~3 min** for 100K candidates, **< 16 GB RAM** (the pool is read once, tokenized once,
  and only the running top-K is held in a bounded heap).
- **Deterministic** — same input always yields the same ranking; score ties break by
  `candidate_id` ascending, matching the validator.
