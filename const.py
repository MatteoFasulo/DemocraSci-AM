# EMBEDDINGS
N_TOP_BILLS = 50
EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
#RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
RERANK_THRESHOLD = 0.40

# STANCE DETECTION
STANCE_DETECTION_MODEL = "MatteoFasulo/mdeberta-v3-xstance"
# STANCE_DETECTION_MODEL = "deepseek-v4-flash"
# STANCE_DETECTION_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
STANCE_LLM_LABELS = {"A": "IN FAVOR", "B": "AGAINST", "C": "NEUTRAL"}
STANCE_LLM_SYSTEM_PROMPT = """You are an annotator for political science.

Classify the passage's stance toward the given claim.

Labels:

* A: The passage is IN FAVOR of the claim.
* B: The passage is AGAINST the claim.
* C: The passage is NEUTRAL toward the claim, is unrelated, or does not provide enough information to determine a stance.

Return exactly one valid JSON object in this format:
{"answer":"A"}

The value of "answer" must be exactly one of: "A", "B", or "C".

Do not include explanations, reasoning, markdown, or any other text.
"""

# Few-shot demonstrations shown to the stance-classification LLM as prior
# user/assistant turns, loaded from this file and keyed by exact stance-claim
# text. See stance_demonstrations.json for the format.
STANCE_LLM_DEMONSTRATIONS_PATH = "stance_demonstrations.json"

# MISC
PREDEFINED_21_TOPICS = ['Civil Rights', 'Technology, Communication', 'International Affairs', 'Government Operations', 'Economy', 'Education, Research', 'Culture, Arts, Religion, Sport, Entertainment', 'Public Finances, Taxation', 'Agriculture', 'Social Welfare', 'Labor', 'Health', 'National Security, Armed Forces, Defense', 'Environment', 'Energy', '(Public) Transportation', 'Finances, Financial Sector', 'Spatial Planing, Civil Engineering', 'Migration', 'Parliamentary Procedure', 'Criminal Law']