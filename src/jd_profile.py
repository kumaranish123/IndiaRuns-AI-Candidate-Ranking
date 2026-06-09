"""Structured interpretation of the released job description.

The JD is deliberately written in prose and tells you, in plain language, what
the role *means* versus what a naive keyword reader would extract. This module
turns that prose into the structured target the scorer ranks against:

  - the role this person actually does (an ML/AI/SWE practitioner),
  - the domain that matters (retrieval / ranking / search / recsys / NLP),
  - the experience band (5-9 yrs, 6-8 ideal, applied at product companies),
  - the things the JD explicitly does NOT want.

Everything here is sourced from job_description.docx; nothing is invented.
"""

# --- Experience band -------------------------------------------------------
YOE_IDEAL_MIN = 6.0
YOE_IDEAL_MAX = 8.0
YOE_ACCEPT_MIN = 5.0
YOE_ACCEPT_MAX = 9.0

# --- Role classification ---------------------------------------------------
# Titles that ARE the job (an engineer/scientist who builds ML systems).
CORE_ROLE_TERMS = (
    "ml engineer", "machine learning engineer", "applied ml", "ai engineer",
    "ai research engineer", "ai specialist", "research engineer", "applied scientist",
    "data scientist", "nlp engineer", "search engineer", "recommendation systems engineer",
    "recommendation engineer", "relevance engineer", "ranking engineer",
    "research scientist",
)

# Adjacent engineering titles: real software/data builders who can be a strong
# fit if their work shows ML/IR exposure (the "plain-language Tier 5" path).
ADJACENT_ROLE_TERMS = (
    "software engineer", "backend engineer", "data engineer", "analytics engineer",
    "full stack", "platform engineer", "staff engineer", "principal engineer",
)

# Titles that are NOT the job, no matter how many AI keywords sit in the skills
# list. This is the keyword-stuffer trap the JD warns about explicitly.
NON_TECHNICAL_ROLE_TERMS = (
    "marketing manager", "hr manager", "human resources", "content writer",
    "graphic designer", "accountant", "sales executive", "sales manager",
    "customer support", "operations manager", "project manager", "business analyst",
    "civil engineer", "mechanical engineer", "electrical engineer", "recruiter",
    "office manager", "product manager",
)

# --- Domain signal ---------------------------------------------------------
# What "AI experience" must mean for this role: retrieval / ranking / search /
# recommendation / NLP / IR — not just "uses an LLM API".
DOMAIN_CORE_TERMS = (
    "retrieval", "ranking", "rank", "search", "recommendation", "recommender",
    "recsys", "information retrieval", "relevance", "embedding", "embeddings",
    "semantic search", "vector search", "nlp", "natural language",
    "learning to rank", "bm25", "hybrid search",
)

VECTOR_DB_TERMS = (
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector db",
)

EVAL_TERMS = ("ndcg", "mrr", "map", "recall@", "precision@", "a/b test", "ab test", "offline eval")

LLM_TERMS = (
    "llm", "large language model", "transformer", "fine-tun", "lora", "qlora",
    "peft", "rag", "sentence-transformer", "bge", "e5",
)

GENERAL_ML_TERMS = (
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit",
    "xgboost", "feature engineering", "model training", "inference", "mlops",
)

# Domains the JD says are NOT the focus when present *without* NLP/IR.
OFF_DOMAIN_TERMS = (
    "computer vision", "image classification", "object detection", "segmentation",
    "speech recognition", "robotics", "gans", "image generation", "ocr", "tts",
)

# --- Company type ----------------------------------------------------------
# "People who have only worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "lti", "larsen", "mphasis",
    "dxc", "deloitte", "pwc", "kpmg", "ernst", "ibm global",
)

SERVICES_INDUSTRIES = ("it services", "consulting", "staffing", "bpo")

# --- Location --------------------------------------------------------------
# Pune/Noida preferred; Tier-1 Indian cities welcome; relocation considered.
PREFERRED_LOCATIONS = ("pune", "noida")
TIER1_INDIA_LOCATIONS = (
    "bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "ncr", "chennai", "kolkata",
)

# --- TF-IDF query ----------------------------------------------------------
# A compact bag of the concepts that define a strong match, used as the query
# vector for lexical similarity against each candidate's free text.
JD_QUERY_TEXT = (
    "senior ai engineer machine learning embeddings retrieval ranking search "
    "recommendation systems information retrieval relevance vector database "
    "pinecone weaviate qdrant milvus faiss elasticsearch opensearch hybrid search "
    "bm25 semantic search nlp natural language processing transformer llm "
    "fine tuning learning to rank ndcg mrr map evaluation offline online ab testing "
    "production deployment product company recommendation engine python pytorch "
    "feature engineering model serving inference scale recruiter candidate matching"
)
