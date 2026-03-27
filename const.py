DB_FILE = "/home/matteo/Downloads/democrasci/democrasci_rdb.db"
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "neo4j")
NEO4J_DB = "neo4j"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in Swiss political science and multilingual text analysis. "
    "Your task is to extract high-level, cohesive political topic labels from parliamentary speeches."
)

DEFAULT_CHAT_PROMPT = """You will extract a short, highly descriptive political topic label from given documents and keywords.
The input texts and keywords may be in German, French, or Italian.
**You must always output the final topic label in English.**

Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Die Schweiz muss ihre Neutralität in diesen Konflikten wahren. (Switzerland must maintain its neutrality in these conflicts.)
- La neutralité armée est le pilier de notre politique étrangère. (Armed neutrality is the pillar of our foreign policy.)
- Wir dürfen keine Waffen an kriegsführende Parteien exportieren. (We must not export weapons to warring parties.)

Keywords: Neutralität neutralité armes Waffen export foreign policy sécurité
topic: Swiss armed neutrality and exports

# Example 2
Sample texts from this topic:
- Les primes d'assurance maladie augmentent chaque année, c'est inacceptable pour les familles. (Health insurance premiums increase every year...)
- Wir müssen die Gesundheitskosten dämpfen, sonst können sich die Bürger die Krankenkasse nicht mehr leisten. (We must curb healthcare costs...)
- I costi della salute sono troppo alti nel nostro cantone. (Health costs are too high in our canton.)

Keywords: primes Krankenkasse santé Gesundheitskosten insurance malattia costi
topic: Healthcare costs and insurance premiums

# Your task
Sample texts from this topic:[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (between 2 and 5 words) in the following format:
topic: <topic_label>
"""
