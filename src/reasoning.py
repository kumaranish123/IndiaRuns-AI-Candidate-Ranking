"""Build grounded, per-candidate reasoning strings for the submission CSV.

Every clause is sourced from facts already extracted into `detail` (titles,
years, matched domain terms, behavioral signals). We never name a skill the
candidate doesn't have, and the stated concern is consistent with the score —
the two failure modes the spec penalizes at manual review.
"""


def _fmt_terms(terms, limit=3):
    # Drop a term if a longer matched term already contains it (e.g. keep
    # "ranking", drop the redundant "rank").
    kept = []
    for term in terms:
        if any(term != other and term in other for other in terms):
            continue
        kept.append(term)
    return "/".join(kept[:limit])


def build_reasoning(candidate, detail):
    profile = candidate.get("profile", {})
    ev = detail["evidence"]
    title = profile.get("current_title", "Candidate")
    yoe = profile.get("years_of_experience") or 0

    positives = [f"{title} with {yoe:.1f} yrs"]

    domain = ev["domain"]
    if domain["core_terms"]:
        positives.append(f"shows {_fmt_terms(domain['core_terms'])} experience")
    elif ev["role"]["current_class"] in ("core", "adjacent"):
        positives.append("engineering background")

    if domain["vecdb_terms"]:
        positives.append(f"{_fmt_terms(domain['vecdb_terms'], 2)}")

    n_skills = ev["skill"]["n_relevant_skills"]
    if n_skills:
        positives.append(f"{n_skills} relevant ML/AI skills")

    behav = ev["behavioral"]
    if behav["response_rate"] >= 0.4:
        positives.append(f"responsive to recruiters ({behav['response_rate']:.2f})")

    loc = ev["location"]
    if loc["country"] and loc["country"].lower() == "india" and loc["location"]:
        positives.append(f"based in {loc['location']}")
    elif loc["relocate"]:
        positives.append("open to relocation")

    # Pick the single most important caveat, consistent with the score.
    concern = None
    if detail["validity_mult"] < 0.5:
        concern = "profile has internal inconsistencies (treated as low-trust)"
    elif ev["role"]["current_class"] == "non_technical":
        concern = f"current role is {title}, not an ML/AI practitioner despite listed skills"
    elif domain["off_domain_only"]:
        concern = "background skews CV/speech with little NLP/IR signal"
    elif ev["company"]["consulting_only"]:
        concern = "career entirely at services/consulting firms"
    elif ev["company"]["job_hopper"]:
        concern = "short average tenure (job-hop pattern)"
    elif behav["days_inactive"] >= 180:
        concern = f"inactive ~{behav['days_inactive'] // 30} months"
    elif behav["response_rate"] < 0.2:
        concern = f"low recruiter response rate ({behav['response_rate']:.2f})"
    elif behav["notice_period_days"] and behav["notice_period_days"] > 90:
        concern = f"long notice period ({behav['notice_period_days']}d)"

    text = "; ".join(positives) + "."
    if concern:
        text += f" Concern: {concern}."
    return text
