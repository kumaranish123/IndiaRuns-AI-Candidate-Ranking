# System Architecture

## End-to-end pipeline

```
candidates.jsonl
      |
      v
[ read once + tokenize once ]  ---->  cached (candidate, tokens, lower_text)
      |                                      |
      v                                      |
[ TF-IDF document-frequency index ]          |
      |                                      |
      v                                      v
[ JD query vector ]  ------------>  [ score each candidate ]
                                            |
                  +-------------------------+--------------------------+
                  |                         |                          |
            content_fit            profile_validity         behavioral_availability
       (role, domain, skill,     (honeypot / impossible    (recency, response rate,
        experience, company,      profile detection)        notice period, demand)
        location, education)
                  |
                  v
        final = content_fit x validity x behavioral
                  |
                  v
        [ bounded min-heap, size = top ]
                  |
                  v
   sort by (-round(score,4), candidate_id) -> ranks 1..N
                  |
                  v
            submission.csv  (candidate_id, rank, score, reasoning)
```

The pool is read and tokenized exactly once. Document frequencies for TF-IDF are
accumulated during that pass; scoring then reuses the cached tokens and lowercased text.
Only the current best `top` candidates are retained, in a bounded heap, so memory stays
flat regardless of pool size.

## Why this design

**No LLM at ranking time.** The spec is explicit that a system calling a hosted model per
candidate cannot scale to a 200K production pool and won't fit the 5-minute CPU budget. A
transparent feature-based scorer is fast, reproducible inside a sandboxed container, and —
crucially — **defensible at the Stage-5 interview**, because every number has a reason.

**Hybrid lexical + structured scoring.** Pure keyword counting is exactly the trap the JD
warns about. Pure embeddings would also chase keywords and rank honeypots highly. So the
decisive signal is **structured** (role gate, experience band, company type, validity),
and the **TF-IDF similarity** is one input into the domain component — enough to surface
"plain-language Tier-5" candidates whose career text describes a recommendation system
without using fashionable buzzwords.

**Multiplicative behavioral and validity modifiers.** Availability and consistency are not
additive niceties; an unreachable candidate or an impossible profile should collapse the
score, not nudge it. Modeling them as multipliers expresses that directly.

## Component notes

- **Role gate (`features.role_fit`)** — classifies the current title (and history) as
  `core` / `adjacent` / `unknown` / `non_technical`. A non-technical current title with no
  technical history scores ~0.05, which is what neutralizes keyword stuffers regardless of
  how many AI skills they list.
- **Domain fit (`features.domain_fit`)** — counts evidence across six concept groups
  (core IR/ranking terms, vector DBs, evaluation, LLM, general ML, off-domain) and blends
  it with the JD TF-IDF cosine. Candidates whose only ML signal is computer-vision/speech
  with no NLP/IR are discounted, per the JD.
- **Skill trust (`features.skill_fit`)** — a skill's contribution is
  `proficiency × (endorsement + duration factors)`, then scaled by the platform assessment
  score when present. An "expert" tag with a 30/100 assessment contributes little — this is
  the anti-keyword-stuffing lever on the skills list itself.
- **Validity (`validation.py`)** — flags only **rare, hard impossibilities** (verified
  against the real data so we don't penalize ordinary noise such as a skill whose
  months-used slightly exceeds tenure, which occurs for ~9% of normal candidates). One hard
  flag drops the multiplier to ~0.12; two or more to ~0.02.

## Determinism and tie-breaks

Scores are rounded to 4 decimals for output and the top-K is sorted by
`(-rounded_score, candidate_id)`. This exactly matches the validator's requirements: score
non-increasing with rank, and equal scores broken by `candidate_id` ascending.

## Complexity / performance

- One pass over the file: O(total tokens) for parsing + tokenizing + DF accumulation.
- Scoring: O(N × average profile length) for the cosine plus O(N) feature work.
- Memory: cached candidates + tokens for the single pass, and a heap of size `top`.
- Measured: ~85 s to index + ~90 s to score 100K candidates ≈ 3 min total, CPU-only.

## Validated outcome on the released pool

- **0 keyword-stuffers** (non-technical titles) in the top 100.
- **0 honeypot-flagged** profiles in the top 100 (DQ threshold is 10%).
- Top 100 dominated by Recommendation Systems / ML / AI / NLP / Search Engineers.
- ~93% India-based, consistent with the JD's location preference.
- Scores strictly non-increasing; submission passes `validate_submission.py`.
