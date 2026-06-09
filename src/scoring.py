"""Combine per-component features into a single, explainable fit score.

    final = content_fit  x  profile_validity  x  behavioral_availability

`content_fit` is a weighted blend of how well the candidate matches what the JD
*means* (role, domain, skills, experience, company type, location, education).
The two multipliers express the two hard lessons from the JD and spec: an
internally impossible profile is worthless (honeypot), and a perfect-on-paper
candidate who is unreachable is not actually hireable.
"""

from . import features

WEIGHTS = {
    "role": 0.28,
    "domain": 0.24,
    "skill": 0.14,
    "experience": 0.14,
    "company": 0.11,
    "location": 0.05,
    "education": 0.04,
}


def score_candidate(candidate, tfidf_cosine, validity_mult, text_lower):
    role_s, role_e = features.role_fit(candidate)
    domain_s, domain_e = features.domain_fit(candidate, tfidf_cosine, text_lower)
    skill_s, skill_e = features.skill_fit(candidate)
    exp_s, exp_e = features.experience_fit(candidate)
    company_s, company_e = features.company_career_fit(candidate)
    loc_s, loc_e = features.location_fit(candidate)
    edu_s, _ = features.education_fit(candidate)
    behav_mult, behav_e = features.behavioral(candidate)

    content = (
        WEIGHTS["role"] * role_s
        + WEIGHTS["domain"] * domain_s
        + WEIGHTS["skill"] * skill_s
        + WEIGHTS["experience"] * exp_s
        + WEIGHTS["company"] * company_s
        + WEIGHTS["location"] * loc_s
        + WEIGHTS["education"] * edu_s
    )

    final = content * validity_mult * behav_mult

    detail = {
        "final": final,
        "content": content,
        "validity_mult": validity_mult,
        "behavioral_mult": behav_mult,
        "components": {
            "role": role_s,
            "domain": domain_s,
            "skill": skill_s,
            "experience": exp_s,
            "company": company_s,
            "location": loc_s,
            "education": edu_s,
        },
        "evidence": {
            "role": role_e,
            "domain": domain_e,
            "skill": skill_e,
            "experience": exp_e,
            "company": company_e,
            "location": loc_e,
            "behavioral": behav_e,
        },
    }
    return final, detail
