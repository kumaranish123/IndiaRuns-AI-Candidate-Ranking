"""Unit tests for the ranking engine.

These run with the standard library only and use small inline fixtures, so they
work without the 487 MB candidate pool. Run with:  python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank import rank
from src import features, scoring
from src.validation import consistency_flags, validity_multiplier


def _signals(**over):
    base = {
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.8,
        "interview_completion_rate": 0.8,
        "notice_period_days": 30,
        "saved_by_recruiters_30d": 5,
        "search_appearance_30d": 120,
        "profile_completeness_score": 90,
        "skill_assessment_scores": {},
        "willing_to_relocate": True,
        "github_activity_score": 40,
    }
    base.update(over)
    return base


STRONG = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "current_title": "Machine Learning Engineer",
        "years_of_experience": 7.0,
        "location": "Pune, Maharashtra",
        "country": "India",
        "headline": "ML Engineer | retrieval, ranking, embeddings",
        "summary": "Built recommendation and search ranking systems with embeddings and vector search at a product company.",
    },
    "career_history": [
        {"company": "Swiggy", "title": "Machine Learning Engineer", "industry": "Food Delivery",
         "duration_months": 40, "is_current": True, "end_date": None,
         "description": "Owned the candidate ranking and retrieval pipeline using embeddings, FAISS and learning to rank; evaluated with NDCG and MRR."},
        {"company": "Flipkart", "title": "Data Scientist", "industry": "E-commerce",
         "duration_months": 44, "is_current": False, "end_date": "2022-01-01",
         "description": "Recommendation systems and semantic search relevance with Elasticsearch."},
    ],
    "education": [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
                   "start_year": 2013, "end_year": 2017, "tier": "tier_1"}],
    "skills": [
        {"name": "Retrieval", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
        {"name": "NLP", "proficiency": "advanced", "endorsements": 25, "duration_months": 40},
    ],
    "redrob_signals": _signals(),
}

KEYWORD_STUFFER = {
    "candidate_id": "CAND_0000002",
    "profile": {
        "current_title": "Marketing Manager",
        "years_of_experience": 7.0,
        "location": "Pune, Maharashtra",
        "country": "India",
        "headline": "Marketing Manager | RAG, LLM, embeddings, vector search",
        "summary": "Marketing leader. Skills include retrieval, ranking, embeddings, NLP, LLM, FAISS.",
    },
    "career_history": [
        {"company": "BrandCo", "title": "Marketing Manager", "industry": "Advertising",
         "duration_months": 84, "is_current": True, "end_date": None,
         "description": "Led marketing campaigns and brand strategy."},
    ],
    "education": [],
    "skills": [
        {"name": "Retrieval", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ],
    "redrob_signals": _signals(),
}

HONEYPOT = {
    "candidate_id": "CAND_0000003",
    "profile": {
        "current_title": "ML Engineer",
        "years_of_experience": 3.0,
        "location": "Pune, Maharashtra",
        "country": "India",
        "headline": "ML Engineer | retrieval ranking embeddings",
        "summary": "Retrieval ranking embeddings vector search recommendation NLP.",
    },
    "career_history": [
        {"company": "NewCo", "title": "ML Engineer", "industry": "Software",
         "duration_months": 96, "is_current": True, "end_date": None,
         "description": "Retrieval and ranking."},  # 96 months but only 3 yrs experience
    ],
    "education": [],
    "skills": [
        {"name": "Retrieval", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Ranking", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ],
    "redrob_signals": _signals(),
}


class TestValidation(unittest.TestCase):
    def test_strong_profile_is_consistent(self):
        self.assertEqual(consistency_flags(STRONG), [])
        self.assertEqual(validity_multiplier(STRONG), 1.0)

    def test_honeypot_is_flagged(self):
        self.assertGreaterEqual(len(consistency_flags(HONEYPOT)), 1)
        self.assertLess(validity_multiplier(HONEYPOT), 0.3)


class TestFeatures(unittest.TestCase):
    def test_keyword_stuffer_role_is_low(self):
        strong_role, _ = features.role_fit(STRONG)
        stuffer_role, _ = features.role_fit(KEYWORD_STUFFER)
        self.assertGreater(strong_role, 0.9)
        self.assertLess(stuffer_role, 0.2)

    def test_assessment_validation_reduces_unbacked_skills(self):
        s, ev = features.skill_fit(STRONG)
        self.assertGreater(s, 0.3)
        self.assertGreater(ev["n_relevant_skills"], 0)


class TestEndToEnd(unittest.TestCase):
    def _write(self, candidates):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")
        return path

    def test_ranking_order_and_format(self):
        path = self._write([KEYWORD_STUFFER, HONEYPOT, STRONG])
        try:
            rows = rank(path, top=3, verbose=False)
        finally:
            os.remove(path)

        self.assertEqual(len(rows), 3)
        # Strong genuine candidate must outrank the stuffer and the honeypot.
        self.assertEqual(rows[0][0], "CAND_0000001")
        # Ranks are 1..3 and scores are non-increasing.
        self.assertEqual([r[1] for r in rows], [1, 2, 3])
        scores = [r[2] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Reasoning is non-empty and names the real title.
        self.assertIn("Machine Learning Engineer", rows[0][3])


if __name__ == "__main__":
    unittest.main()
