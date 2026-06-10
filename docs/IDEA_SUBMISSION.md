# Idea Submission — Intelligent Candidate Discovery & Ranking Engine

> Follows the official *Idea Submission Template* (Redrob / INDIA RUNS, Track 1).

**Team Name:** `<your-registered-participant-id>`
**Problem Statement:** Intelligent Candidate Discovery & Ranking — given the *Senior AI
Engineer (Founding Team)* JD, return an explainable ranked top-100 from a 100K-profile pool.
**Team Leader Name:** `<your-name>`

---

## Solution Overview

**What is your proposed solution?**
An explainable hybrid ranking engine that scores each candidate as
`content_fit × profile_validity × behavioral_availability`. `content_fit` blends seven
interpretable components (role, domain, skills, experience, company/career, location,
education); a TF-IDF similarity to the JD feeds the domain component. Behavioral signals
and an internal-consistency (honeypot) check act as multipliers. It runs CPU-only with no
LLM calls at ranking time and uses only the Python standard library.

**What differentiates your approach from traditional candidate matching systems?**
Traditional ATS/keyword matchers reward profiles that *contain* the right words — exactly
the trap this dataset is built around. Our system instead:
- gates on **what the candidate actually does** (job title/career), not what they list;
- reads **career-history prose** so a "plain-language" expert who built a recsys without
  buzzwords still surfaces;
- **validates** the claimed skills against platform assessment scores and the profile
  against internal consistency (catching honeypots);
- weighs **behavioral availability**, because a perfect-on-paper but unreachable candidate
  is not hireable.

---

## JD Understanding & Candidate Evaluation

**Key requirements extracted from the JD:**
- 5-9 years (6-8 ideal), 4-5 of them applied ML/AI **at product companies, not services**.
- Hands-on **embeddings retrieval + vector search / hybrid search**, **ranking / search /
  recommendation** systems shipped to real users.
- Strong **evaluation** instinct (NDCG / MRR / MAP, offline↔online).
- **Explicit negatives:** keyword/title-only AI, title-chasers (job-hoppers), consulting-only
  careers, pure-research-no-production, CV/speech/robotics without NLP/IR.
- **Location:** Pune/Noida preferred; Tier-1 India welcome; relocation considered.

**Which candidate signals matter most / how do you evaluate fit beyond keyword matching?**
The decisive signal is the **role/title gate** combined with **career-history evidence** —
a Marketing Manager with a perfect AI skills list scores near zero. On top of that, domain
relevance (retrieval/ranking/NLP), trust-weighted (assessment-validated) skills, experience
band, product-vs-services career, and behavioral availability determine the ordering.

---

## Ranking Methodology

**How does the system retrieve, score, and rank?**
One streaming pass reads and tokenizes the pool and builds a TF-IDF index. Each candidate
is then scored with the formula above; a bounded heap keeps the running top-100, which is
sorted by `(-score, candidate_id)` into final ranks.

**Models / algorithms / heuristics:**
- **Pure-Python TF-IDF cosine** to the JD query vector (semantic-ish retrieval signal).
- **Interpretable rule-based scoring** per component, with calibrated weights.
- **Trust weighting** of skills (proficiency × endorsements × duration × assessment).
- **Consistency rules** for honeypot detection.

**How are multiple signals combined?**
`content_fit` is a fixed-weight linear blend; **availability** and **validity** are
**multiplicative** so they can collapse (not merely nudge) the score.

---

## Explainability & Data Validation

**How are ranking decisions explained?**
Every row carries a `reasoning` string built only from extracted facts: title + years,
the specific domain terms matched, vector DBs found, number of relevant skills, recruiter
responsiveness, location, and the single most relevant concern (e.g. long notice period,
inactivity, consulting-only career).

**How do you prevent hallucinations / unsupported justifications?**
Reasoning is templated **from the same evidence used to score** — it can never name a skill
the candidate doesn't have, and the stated concern is always consistent with the rank.

**How do you handle inconsistent / low-quality / suspicious profiles?**
`src/validation.py` flags hard impossibilities (a single role longer than the whole career,
several "expert" skills with 0 months of use, backwards education timelines, current-role
with an end date). Flagged profiles get a near-zero multiplier, pushing honeypots out of the
top ranks. The rule set was calibrated against the real data to avoid penalizing ordinary
noise.

---

## End-to-End Workflow

JD (interpreted into `src/jd_profile.py`) → read pool once + tokenize → build TF-IDF index
→ score each candidate (content × validity × behavioral) → keep top-100 in a heap → sort and
assign ranks → write `submission.csv` with grounded reasoning → validate with
`validate_submission.py`.

---

## Results & Performance

**Ranking quality (released pool):** Top 100 contains **0 keyword-stuffers** and **0
honeypot-flagged** profiles (DQ threshold 10%), is dominated by Recommendation Systems / ML
/ AI / NLP / Search Engineers, and is ~93% India-based — consistent with the JD.

**Compute:** ~3 minutes for 100K candidates, CPU-only, < 16 GB RAM, no network — comfortably
within the 5-minute budget. Output passes the official validator.

---

## Technologies Used

- **Python standard library only** for the ranker (json, csv, re, math, heapq, datetime) —
  zero ML dependencies, so reproduction inside the Stage-3 sandbox is friction-free.
- **Custom TF-IDF** (no scikit-learn) for offline, fast lexical similarity.
- **Streamlit** for the required hosted sandbox/demo (`app.py`).
- **unittest** for the test suite.

Chosen for **reproducibility, speed, and explainability** — the three things the spec
filters on (Stage-3 compute reproduction, Stage-4 reasoning review, Stage-5 defense).

---

## Submission Assets

- **GitHub:** `https://github.com/<your-username>/IndiaRuns-AI-Candidate-Ranking`
- **Sandbox:** `https://huggingface.co/spaces/<your-username>/redrob-ranker` (or Streamlit Cloud)
- **Video demo:** `<link to screencast>`
- **Ranked CSV:** `submission.csv` (top-100)
