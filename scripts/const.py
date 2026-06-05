# DB / Neo4J
DB_FILE = "/home/matteo/Downloads/democrasci/democrasci_rdb.db"
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "neo4j")
NEO4J_DB = "neo4j"

# vLLM Prompts
# From https://github.com/MaartenGr/BERTopic/blob/master/bertopic/representation/_openai.py
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that extracts high-level topics from texts."
)

DEFAULT_CHAT_PROMPT = """You will extract a short topic label from given documents and keywords.

Your task:
1. Provide a concise and descriptive label that represents the cluster.

Here an example of topic you created before:

# Example 1
Sample texts from this topic:
- Les primes d'assurance maladie augmentent chaque année...
- Wir müssen die Gesundheitskosten dämpfen...
Keywords: primes, Krankenkasse, santé, Gesundheitskosten, insurance
topic: Healthcare costs and insurance premiums

# Your current task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic_label>
"""
