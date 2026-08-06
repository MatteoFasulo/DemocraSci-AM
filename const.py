# EMBEDDINGS
N_TOP_BILLS = 50
EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
#RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
RERANK_THRESHOLD = 0.40

# STANCE DETECTION
# STANCE_DETECTION_MODEL = "MatteoFasulo/mdeberta-v3-xstance"
# STANCE_DETECTION_MODEL = "deepseek-v4-flash"
STANCE_DETECTION_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
STANCE_LLM_LABELS = {"A": "IN FAVOR", "B": "AGAINST", "C": "NEUTRAL"}
STANCE_LLM_SYSTEM_PROMPT = """You are an annotator for political science.
Decide whether the passage is in favor, against, or neutral toward the claim.
The label must be one of "A", "B", or "C".

A means IN FAVOR
B means AGAINST
C means NEUTRAL
"""

# MISC
PREDEFINED_21_TOPICS = ['Civil Rights', 'Technology, Communication', 'International Affairs', 'Government Operations', 'Economy', 'Education, Research', 'Culture, Arts, Religion, Sport, Entertainment', 'Public Finances, Taxation', 'Agriculture', 'Social Welfare', 'Labor', 'Health', 'National Security, Armed Forces, Defense', 'Environment', 'Energy', '(Public) Transportation', 'Finances, Financial Sector', 'Spatial Planing, Civil Engineering', 'Migration', 'Parliamentary Procedure', 'Criminal Law']