"""Per-candidate feature extraction.

Each function turns one slice of a candidate profile into a sub-score in [0, 1]
plus a little evidence we reuse when writing the reasoning column. The scorer
in `scoring.py` combines these; keeping them separate makes the system
auditable and easy to defend in an interview.
"""

from datetime import date

from . import jd_profile as jd

# Dataset era. Candidate `last_active_date` values run up to mid-2026, so we
# anchor recency to the start of June 2026 rather than the real wall clock.
REFERENCE_DATE = date(2026, 6, 1)

PROFICIENCY_WEIGHT = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85, "expert": 1.0}


def _contains_any(text, terms):
    return [t for t in terms if t in text]


# --- role / title ----------------------------------------------------------
def _classify_title(title):
    t = (title or "").lower()
    if any(term in t for term in jd.CORE_ROLE_TERMS):
        return "core"
    if any(term in t for term in jd.NON_TECHNICAL_ROLE_TERMS):
        return "non_technical"
    if any(term in t for term in jd.ADJACENT_ROLE_TERMS):
        return "adjacent"
    return "unknown"


def role_fit(candidate):
    profile = candidate.get("profile", {})
    current = profile.get("current_title", "")
    current_class = _classify_title(current)

    history_classes = {_classify_title(j.get("title", "")) for j in candidate.get("career_history", [])}
    has_core_history = "core" in history_classes
    has_adjacent_history = "adjacent" in history_classes

    if current_class == "core":
        score = 1.0
    elif current_class == "adjacent":
        score = 0.78 if has_core_history else 0.62
    elif current_class == "unknown":
        score = 0.5 if (has_core_history or has_adjacent_history) else 0.32
    else:  # non_technical current title
        # keyword-stuffer trap: AI skills but a Marketing/HR/etc. job.
        score = 0.32 if has_core_history else 0.05

    return score, {"current_title": current, "current_class": current_class}


# --- domain ----------------------------------------------------------------
def domain_fit(candidate, tfidf_cosine, text):
    core = _contains_any(text, jd.DOMAIN_CORE_TERMS)
    vecdb = _contains_any(text, jd.VECTOR_DB_TERMS)
    evals = _contains_any(text, jd.EVAL_TERMS)
    llm = _contains_any(text, jd.LLM_TERMS)
    genml = _contains_any(text, jd.GENERAL_ML_TERMS)
    off = _contains_any(text, jd.OFF_DOMAIN_TERMS)

    rule = (
        0.50 * min(len(core), 4) / 4
        + 0.15 * min(len(vecdb), 2) / 2
        + 0.10 * min(len(evals), 2) / 2
        + 0.10 * min(len(llm), 3) / 3
        + 0.15 * min(len(genml), 4) / 4
    )

    # CV / speech / robotics without any NLP/IR signal: JD says they'd be
    # re-learning fundamentals here, so discount the domain match.
    off_domain_only = bool(off) and not core
    if off_domain_only:
        rule *= 0.4

    tfidf_norm = min(tfidf_cosine / 0.35, 1.0)
    score = min(0.6 * rule + 0.4 * tfidf_norm, 1.0)

    evidence = {
        "core_terms": core,
        "vecdb_terms": vecdb,
        "eval_terms": evals,
        "llm_terms": llm,
        "off_domain_only": off_domain_only,
        "n_core": len(core),
    }
    return score, evidence


# --- skills (trust-weighted) ----------------------------------------------
_RELEVANT_SKILL_TERMS = (
    jd.DOMAIN_CORE_TERMS + jd.VECTOR_DB_TERMS + jd.LLM_TERMS + jd.GENERAL_ML_TERMS
)


def skill_fit(candidate):
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {}) or {}
    trusted = 0.0
    relevant_names = []

    for skill in candidate.get("skills", []):
        name = (skill.get("name") or "")
        low = name.lower()
        if not any(term in low for term in _RELEVANT_SKILL_TERMS):
            continue
        prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency"), 0.5)
        endorsements = skill.get("endorsements") or 0
        duration = skill.get("duration_months") or 0
        end_factor = 0.5 + 0.5 * min(endorsements, 20) / 20
        dur_factor = 0.5 + 0.5 * min(duration, 36) / 36
        trust = prof * (0.4 + 0.3 * end_factor + 0.3 * dur_factor)

        # Validate claimed skill against the platform assessment when present:
        # an "expert" tag with a 30/100 assessment is exactly the kind of
        # keyword-stuffing the JD warns about.
        if name in assessments:
            trust *= 0.4 + 0.6 * (assessments[name] / 100.0)
        trusted += trust
        relevant_names.append(name)

    score = min(trusted / 5.0, 1.0)
    return score, {"relevant_skills": relevant_names, "n_relevant_skills": len(relevant_names)}


# --- experience ------------------------------------------------------------
def experience_fit(candidate):
    yoe = candidate.get("profile", {}).get("years_of_experience") or 0
    if jd.YOE_IDEAL_MIN <= yoe <= jd.YOE_IDEAL_MAX:
        score = 1.0
    elif jd.YOE_ACCEPT_MIN <= yoe <= jd.YOE_ACCEPT_MAX:
        score = 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 10:
        score = 0.65
    elif 3 <= yoe < 4 or 10 < yoe <= 12:
        score = 0.45
    else:
        score = 0.25

    if "junior" in candidate.get("profile", {}).get("current_title", "").lower():
        score *= 0.7
    return score, {"yoe": yoe}


# --- company / career ------------------------------------------------------
def _is_services(company, industry):
    company = (company or "").lower()
    industry = (industry or "").lower()
    if any(firm in company for firm in jd.CONSULTING_FIRMS):
        return True
    return any(ind in industry for ind in jd.SERVICES_INDUSTRIES)


def company_career_fit(candidate):
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    evidence = {"consulting_only": False, "job_hopper": False}

    if not career:
        return 0.4, evidence

    services_flags = [_is_services(j.get("company"), j.get("industry")) for j in career]
    product_frac = 1.0 - (sum(services_flags) / len(services_flags))
    consulting_only = all(services_flags)
    evidence["consulting_only"] = consulting_only

    # Title-chaser / job-hopper check: many short stints.
    completed = [j for j in career if (j.get("duration_months") or 0) > 0 and not j.get("is_current")]
    if len(career) >= 3 and completed:
        avg_tenure = sum((j.get("duration_months") or 0) for j in completed) / len(completed)
        if avg_tenure < 18:
            stability = 0.0
            evidence["job_hopper"] = True
        else:
            stability = min((avg_tenure - 18) / 24, 1.0)
    else:
        stability = 0.7

    score = 0.35 + 0.4 * product_frac + 0.2 * stability

    # External validation (open-source/github) is a JD plus.
    github = signals.get("github_activity_score", -1)
    if github is not None and github > 30:
        score += 0.05

    if consulting_only:
        score *= 0.35  # "only consulting firms in their entire career" => not a fit
    return min(score, 1.0), evidence


# --- location --------------------------------------------------------------
def location_fit(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate"))

    if country == "india":
        if any(c in location for c in jd.PREFERRED_LOCATIONS):
            score = 1.0
        elif any(c in location for c in jd.TIER1_INDIA_LOCATIONS):
            score = 0.85
        else:
            score = 0.7 if relocate else 0.55
    else:
        score = 0.45 if relocate else 0.2
    return score, {"location": profile.get("location"), "country": profile.get("country"), "relocate": relocate}


# --- education -------------------------------------------------------------
_TIER_WEIGHT = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45, "unknown": 0.5}


def education_fit(candidate):
    education = candidate.get("education", [])
    if not education:
        return 0.5, {}
    best = max(_TIER_WEIGHT.get(e.get("tier", "unknown"), 0.5) for e in education)
    return best, {}


# --- behavioral availability ----------------------------------------------
def _days_since(date_str):
    if not date_str:
        return 9999
    try:
        y, m, d = (int(x) for x in date_str.split("-")[:3])
        return (REFERENCE_DATE - date(y, m, d)).days
    except (ValueError, TypeError):
        return 9999


def _recency_score(days):
    if days < 30:
        return 1.0
    if days < 60:
        return 0.92
    if days < 90:
        return 0.8
    if days < 180:
        return 0.55
    return 0.3


def behavioral(candidate):
    s = candidate.get("redrob_signals", {})
    days = _days_since(s.get("last_active_date"))
    recency = _recency_score(days)
    response = max(0.0, min(1.0, s.get("recruiter_response_rate") or 0.0))
    open_score = 1.0 if s.get("open_to_work_flag") else 0.45
    interview = 0.5 + 0.5 * max(0.0, min(1.0, s.get("interview_completion_rate") or 0.0))

    notice = s.get("notice_period_days")
    if notice is None:
        notice_score = 0.7
    elif notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.82
    elif notice <= 90:
        notice_score = 0.62
    else:
        notice_score = 0.42

    saved = min((s.get("saved_by_recruiters_30d") or 0) / 10.0, 1.0)
    appearances = min((s.get("search_appearance_30d") or 0) / 200.0, 1.0)
    demand = 0.5 * saved + 0.5 * appearances
    completeness = (s.get("profile_completeness_score") or 0) / 100.0

    avail = (
        0.30 * recency
        + 0.25 * response
        + 0.12 * open_score
        + 0.10 * interview
        + 0.10 * notice_score
        + 0.08 * demand
        + 0.05 * completeness
    )
    multiplier = 0.5 + 0.5 * avail  # [0.5, 1.0]: halves the score of the unreachable
    evidence = {
        "days_inactive": days,
        "response_rate": response,
        "open_to_work": bool(s.get("open_to_work_flag")),
        "notice_period_days": notice,
    }
    return multiplier, evidence
