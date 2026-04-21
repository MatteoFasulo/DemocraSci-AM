DB_FILE = "/home/matteo/Downloads/democrasci/democrasci_rdb.db"
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "neo4j")
NEO4J_DB = "neo4j"

# From https://github.com/MaartenGr/BERTopic/blob/master/bertopic/representation/_openai.py
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that extracts high-level topics from texts."
)

DEFAULT_CHAT_PROMPT = """You will extract a short topic label from given documents and keywords.

You are provided with a non-exhaustive list of 21 predefined political topics which might be relevant but do not cover all possible topics. The list is as follows:
[BILLS_TOPICS]

Your task:
1. Check if the cluster described by the keywords and documents fits exactly into ONE of the 21 predefined topics listed above.
2. If it does fit, return exactly that topic name otherswise provide a new, concise, and descriptive label (max 5 words) that represents the cluster.

Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Die Schweiz muss ihre Neutralität in diesen Konflikten wahren.
- La neutralité armée est le pilier de notre politique étrangère.
Keywords: Neutralität, neutralité, armes, Waffen, export, foreign policy
topic: National Security, Armed Forces, Defense

# Example 2
Sample texts from this topic:
- Les primes d'assurance maladie augmentent chaque année...
- Wir müssen die Gesundheitskosten dämpfen...
Keywords: primes, Krankenkasse, santé, Gesundheitskosten, insurance
topic: Healthcare costs and insurance premiums

# Your current task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
"""
