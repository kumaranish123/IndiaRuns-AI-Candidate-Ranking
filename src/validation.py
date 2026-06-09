"""Profile-consistency checks that catch honeypots and impossible profiles.

The spec seeds ~80 honeypots with subtly impossible profiles ("8 years at a
company founded 3 years ago", "expert in 10 skills with 0 years used"). Ranking
any of these in the top 10 is a tell that the system embeds keywords instead of
reading profiles, and >10% honeypots in the top 100 is an outright DQ.

We don't special-case the honeypots; we just verify internal arithmetic. A
profile that contradicts itself is heavily down-weighted, which naturally keeps
the seeded impossibilities out of the ranking.
"""

from datetime import date

CURRENT_YEAR = 2026


def _year(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


def consistency_flags(candidate):
    """Return a list of human-readable red flags for an internally inconsistent profile."""
    flags = []
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience") or 0
    yoe_months = yoe * 12.0

    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # NOTE: we deliberately do NOT flag "a skill's duration exceeds total
    # experience". In this dataset ~9% of perfectly ordinary candidates have a
    # skill whose months-used slightly exceeds their tenure, so it is noise, not
    # a honeypot tell. We only flag hard, rare impossibilities below.

    # 1. "Expert" in several skills with zero months of actual use — the
    #    canonical honeypot ("expert in 10 skills with 0 years used").
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if expert_zero >= 3:
        flags.append(f"{expert_zero} 'expert' skills with 0 months of use")

    # 2. One single role lasting longer than the entire stated career
    #    (e.g. "8 years at a company that has existed for 3").
    for job in career:
        if (job.get("duration_months") or 0) > yoe_months + 18:
            flags.append("a single role is longer than total experience")
            break

    # 3. Career durations summing to far more than the career could hold.
    total_career_months = sum((j.get("duration_months") or 0) for j in career)
    if total_career_months > yoe_months + 60 and yoe > 0:
        flags.append("career durations sum to far more than total experience")

    # 4. Education timeline that runs backwards or into the future.
    for edu in candidate.get("education", []):
        start, end = edu.get("start_year"), edu.get("end_year")
        if start and end and end < start:
            flags.append("education ends before it starts")
            break
        if end and end > CURRENT_YEAR + 1:
            flags.append("education ends in the future")
            break

    # 5. is_current flag disagreeing with the presence of an end date.
    for job in career:
        if job.get("is_current") and job.get("end_date"):
            flags.append("role marked current but has an end date")
            break

    return flags


def validity_multiplier(candidate):
    """Map red flags to a score multiplier in [0, 1]."""
    n = len(consistency_flags(candidate))
    if n == 0:
        return 1.0
    if n == 1:
        # The remaining flags are rare, hard impossibilities; one is already a
        # strong honeypot signal.
        return 0.12
    return 0.02  # multiple contradictions => almost certainly a honeypot
