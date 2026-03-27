import os
import sqlite3
from neo4j import GraphDatabase
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from transformers import pipeline
import argparse
import llm
from bertopic.representation._base import BaseRepresentation

DB_FILE = "/home/matteo/Downloads/democrasci/democrasci_rdb.db"
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "neo4j")
NEO4J_DB = "neo4j"

# PROMPTS
DEFAULT_CHAT_PROMPT = """You will extract a short topic label from given documents and keywords.
Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in political science that extracts high-level topics from texts."
)

tqdm.pandas()

def get_sqlite_connection(db_file):
    return sqlite3.connect(db_file)

def get_neo4j_driver(uri, auth):
    return GraphDatabase.driver(uri, auth=auth)

def get_transcripts_from_sqlite(conn, table_name="transcript"):
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, conn)

def fetch_speech_metadata_bulk(df, batch_size, fields=["role", "language_of_text", "name"]):
    ids = df["ID"].astype(str).dropna().unique().tolist()
    rows = []

    query = """
    UNWIND $uids AS uid
    MATCH (n:Speech {uid: uid})
    RETURN n.uid AS ID, n.role AS role, n.language_of_text AS language_of_text, n.name AS name
    """

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=NEO4J_DB) as session:
            for i in tqdm(range(0, len(ids), batch_size), desc="Fetching speech metadata"):
                chunk = ids[i : i + batch_size]
                rows.extend(session.run(query, uids=chunk).data())

    meta_df = pd.DataFrame(rows).drop_duplicates(subset=["ID"])
    return meta_df

def merge_metadata(transcript_df, meta_df):
    merged_df = transcript_df.assign(ID=transcript_df["ID"].astype(str)).merge(meta_df, on="ID", how="left")
    return merged_df

def extract_and_clean_paragraphs(row):
    """Parses the XML/HTML, extracts paragraphs, and filters out garbage."""
    raw_text = str(row["TEXT"])

    soup = BeautifulSoup(raw_text, "html.parser")
    paragraphs = soup.find_all("p")

    cleaned_paragraphs = []

    for i, p in enumerate(paragraphs):
        text = p.get_text(strip=True)

        word_count = len(text.split())

        is_garbage = bool(re.match(r"^(asdf)+$", text.lower()))

        if word_count >= 5 and not is_garbage:
            cleaned_paragraphs.append(
                {
                    "Speech_ID": row["ID"],
                    "Paragraph_ID": f"{row['ID']}_p{i}",
                    "TEXT_SPAN": text,
                    "role": row["role"],
                    "language_of_text": row["language_of_text"],
                    "name": row["name"],
                }
            )

    return cleaned_paragraphs

def get_clean_paragraphs(df):
    df.dropna(inplace=True)
    all_paragraphs = df.progress_apply(extract_and_clean_paragraphs, axis=1).tolist()
    flat_paragraphs = [item for sublist in all_paragraphs for item in sublist]
    return pd.DataFrame(flat_paragraphs)


def load_valid_topics(valid_topics_file):
    if not os.path.exists(valid_topics_file):
        raise FileNotFoundError(f"Valid topics file not found: {valid_topics_file}")
    with open(valid_topics_file, "r", encoding="utf-8") as file:
        topics = [line.strip() for line in file.readlines() if line.strip()]
    if not topics:
        raise ValueError("Valid topics file is empty.")
    return topics


def normalize_topic_choice(raw_output, valid_topics):
    if not raw_output:
        return None

    cleaned = raw_output.strip().splitlines()[0].strip()
    cleaned = cleaned.replace('"', "").replace("'", "").strip()

    if cleaned in valid_topics:
        return cleaned

    lower_map = {topic.lower(): topic for topic in valid_topics}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]

    for topic in valid_topics:
        if topic.lower() in cleaned.lower():
            return topic
    return None


def clean_topic_label(label):
    cleaned = str(label).strip()
    cleaned = re.sub(r"^-?\d+_", "", cleaned)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned


def load_refined_topics(topic_refinement_output, refined_topic_column="Name"):
    if not os.path.exists(topic_refinement_output):
        raise FileNotFoundError(f"Topic refinement output not found: {topic_refinement_output}")

    refined_df = pd.read_csv(topic_refinement_output)
    if refined_topic_column not in refined_df.columns:
        available = ", ".join(refined_df.columns)
        raise ValueError(
            f"Column '{refined_topic_column}' not found in refinement file. Available columns: {available}"
        )

    if "Topic" in refined_df.columns:
        refined_df = refined_df[refined_df["Topic"] != -1]

    hypotheses = []
    for value in refined_df[refined_topic_column].dropna().tolist():
        candidate = clean_topic_label(value)
        if candidate:
            hypotheses.append(candidate)

    deduplicated = list(dict.fromkeys(hypotheses))
    if not deduplicated:
        raise ValueError("No usable hypotheses found in refined topic output.")
    return deduplicated


def build_nli_hypotheses(
    nli_hypotheses_source,
    valid_topics_file,
    topic_refinement_output,
    refined_topic_column,
    restrict_refined_to_valid,
):
    valid_topics = load_valid_topics(valid_topics_file)

    if nli_hypotheses_source == "valid_topics":
        return valid_topics, "valid_topics"

    if nli_hypotheses_source == "refined_topics":
        refined_topics = load_refined_topics(topic_refinement_output, refined_topic_column)
    else:
        refined_topics = (
            load_refined_topics(topic_refinement_output, refined_topic_column)
            if os.path.exists(topic_refinement_output)
            else []
        )

    if nli_hypotheses_source == "auto" and not refined_topics:
        return valid_topics, "valid_topics (auto fallback)"

    if restrict_refined_to_valid:
        normalized = [normalize_topic_choice(topic, valid_topics) for topic in refined_topics]
        filtered = [topic for topic in normalized if topic is not None]
        filtered = list(dict.fromkeys(filtered))
        if filtered:
            return filtered, "refined_topics (restricted to valid_topics)"
        if nli_hypotheses_source == "auto":
            return valid_topics, "valid_topics (auto fallback: no refined topics matched)"
        raise ValueError("No refined topics matched the valid topics list after restriction.")

    return refined_topics, "refined_topics"


class QwenTopic(BaseRepresentation):
    def __init__(
        self,
        model: str = "sdsc-qwen-32b",
        prompt=DEFAULT_CHAT_PROMPT,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        nr_docs: int = 4,
        diversity: float | None = None,
        **kwargs,
    ):
        """
        Use Qwen to extract topic labels from a set of documents and keywords.

        Arguments:
            model: The Qwen model to use (e.g., "sdsc-qwen-32b")
            prompt: The prompt template to use for Qwen. It should contain [KEYWORDS] and/or [DOCUMENTS] as placeholders.
            system_prompt: The system prompt to set the behavior of the Qwen model.
            nr_docs: The number of documents to pass if a prompt
                 with the `["DOCUMENTS"]` tag is used.
            diversity: Accepts values between 0 and 1. A higher
                   values results in passing more diverse documents
                   whereas lower values passes more similar documents.
        """
        self.model = model
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.nr_docs = nr_docs
        self.diversity = diversity

        self.prompts_ = []

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """Extract topics

        Arguments:
            topic_model: The BERTopic model
            documents: A dataframe of documents with their related topics
            c_tf_idf: The c-TF-IDF matrix
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        updated_topics = {}
        for topic, docs in tqdm(
            repr_docs_mappings.items(), disable=not topic_model.verbose
        ):
            prompt = self._create_prompt(docs, topic, topics)
            self.prompts_.append(prompt)

            qwen_model = llm.get_model(self.model)
            response = qwen_model.prompt(prompt=prompt, system=self.system_prompt)
            answer = response.text()
            after_thinking = answer.split("</think>")[-1]
            label = after_thinking.replace("topic:", "").strip()
            updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = next(zip(*topics[topic]))

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt

    def get_prompts(self):
        """Returns the prompts used for topic extraction."""
        return self.prompts_

def run_nli_stance(
    text_df,
    hypotheses,
    nli_model_name,
    entailment_threshold,
    batch_size,
    max_texts,
):
    if max_texts is not None:
        text_df = text_df.head(max_texts).copy()

    texts = text_df["TEXT_SPAN"].tolist()
    paragraph_ids = text_df["Paragraph_ID"].tolist()

    classifier = pipeline(
        "zero-shot-classification",
        model=nli_model_name,
        device_map="auto",
    )

    results = []
    for hypothesis in tqdm(hypotheses, desc="NLI outer loop over hypotheses"):
        predictions = classifier(
            texts,
            candidate_labels=[hypothesis],
            hypothesis_template="The author of this document supports {}.",
            multi_label=True,
            batch_size=batch_size,
        )

        if isinstance(predictions, dict):
            predictions = [predictions]

        for i, pred in enumerate(predictions):
            score = float(pred["scores"][0])
            results.append(
                {
                    "Paragraph_ID": paragraph_ids[i],
                    "TEXT_SPAN": texts[i],
                    "hypothesis_topic": hypothesis,
                    "entailment_score": score,
                    "is_positive": score >= entailment_threshold,
                }
            )

    result_df = pd.DataFrame(results)
    top_df = (
        result_df.sort_values(["Paragraph_ID", "entailment_score"], ascending=[True, False])
        .groupby("Paragraph_ID", as_index=False)
        .first()
        .rename(
            columns={
                "hypothesis_topic": "top_topic",
                "entailment_score": "top_entailment_score",
                "is_positive": "top_is_positive",
            }
        )
    )

    return result_df, top_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic extraction and stance detection pipeline")
    parser.add_argument("--task", type=str, required=True, choices=["paragraph_extraction", "clustering", "refine_topics", "stance_nli"], help="Task to perform")
    parser.add_argument("--n_samples", type=int, default=10_000, help="Number of paragraphs to sample for clustering")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", choices=["all-MiniLM-L6-v2", "BAAI/bge-m3"], help="Sentence embedding model to use")
    parser.add_argument("--alignment_with_predefined_topics", type=float, default=0.65, help="Minimum similarity for zero-shot topic assignment")
    parser.add_argument("--valid_topics_file", type=str, default="valid_topics.txt", help="Path to file containing valid topics for zero-shot assignment")
    parser.add_argument("--clean_paragraphs_path", type=str, default="clean_paragraphs.parquet", help="Path to clean paragraphs parquet")
    parser.add_argument("--clustered_paragraphs_path", type=str, default="clustered_paragraphs.parquet", help="Path to clustered paragraphs parquet")
    parser.add_argument("--topic_model_path", type=str, default="bertopic_model", help="Path to save/load BERTopic model")
    parser.add_argument("--topic_info_path", type=str, default="topic_info.csv", help="Path to save topic info from BERTopic")
    parser.add_argument("--topic_refinement_output", type=str, default="topic_refinements.csv", help="Output CSV for topic refinement")
    parser.add_argument("--refined_topic_column", type=str, default="Name", help="Column in topic_refinement_output used as NLI hypotheses")
    parser.add_argument("--top_n_docs", type=int, default=5, help="Top N docs per topic used in refinement prompt")
    parser.add_argument("--min_topic_size", type=int, default=15, help="Minimum topic size to keep during refinement")
    parser.add_argument("--nli_model_name", type=str, default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", help="NLI model for stance detection")
    parser.add_argument("--nli_hypotheses_source", type=str, default="auto", choices=["auto", "valid_topics", "refined_topics"], help="Source for NLI hypotheses")
    parser.add_argument("--restrict_refined_to_valid_topics", action="store_true", help="Map and restrict refined hypotheses to valid topics list")
    parser.add_argument("--entailment_threshold", type=float, default=0.8, help="Threshold for positive entailment")
    parser.add_argument("--nli_batch_size", type=int, default=16, help="Batch size for NLI inference")
    parser.add_argument("--max_texts_for_nli", type=int, default=None, help="Optional cap on number of texts for NLI")
    parser.add_argument("--stance_pairs_output", type=str, default="stance_entailment_pairs.parquet", help="Output parquet for all (hypothesis, text) scores")
    parser.add_argument("--stance_top_output", type=str, default="stance_top_predictions.parquet", help="Output parquet for top topic per text")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    assert args.task in ["paragraph_extraction", "clustering", "refine_topics", "stance_nli"], "Invalid task. Choose one of: paragraph_extraction, clustering, refine_topics, stance_nli."

    if args.task == "paragraph_extraction":
        print("Connecting to SQLite and fetching transcripts...")
        conn = get_sqlite_connection(DB_FILE)
        transcript_df = get_transcripts_from_sqlite(conn)

        print("Fetching metadata from Neo4j...")
        meta_df = fetch_speech_metadata_bulk(transcript_df, batch_size=10_000)

        print("Merging metadata with transcripts...")
        merged_df = merge_metadata(transcript_df, meta_df)
        merged_df.to_parquet("speeches_with_text.parquet", index=False)

        print("Extracting and cleaning paragraphs...")
        df_paragraphs = get_clean_paragraphs(merged_df)
        df_paragraphs.to_parquet("clean_paragraphs.parquet", index=False)

        print(f"Extracted {len(df_paragraphs)} clean paragraphs.")

    elif args.task == "clustering":
        print("Loading clean paragraphs...")
        df_paragraphs = pd.read_parquet(args.clean_paragraphs_path)

        print(f"Sampling {args.n_samples} paragraphs for clustering...")
        df_paragraphs_sample = df_paragraphs.sample(args.n_samples, random_state=42).reset_index(drop=True)

        print(f"Using embedding model: {args.embedding_model}")
        embedder = SentenceTransformer(args.embedding_model)

        print("Encoding paragraphs...")
        texts = df_paragraphs_sample["TEXT_SPAN"].tolist()
        if os.path.exists(f"paragraph_embeddings_{args.embedding_model.replace('/', '_')}.npy"):
            print("Loading precomputed embeddings...")
            embeddings = np.load(f"paragraph_embeddings_{args.embedding_model.replace('/', '_')}.npy")
        else:
            embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
            np.save(f"paragraph_embeddings_{args.embedding_model.replace('/', '_')}.npy", embeddings)

        print("Clustering with HDBSCAN...")
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean")

        print("Fitting BERTopic...")
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        representation_model = MaximalMarginalRelevance(diversity=0.2)

        valid_topics = load_valid_topics(args.valid_topics_file)
        print(f"Valid topics for zero-shot assignment: {valid_topics}")
        topic_model = BERTopic(
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            hdbscan_model=hdbscan_model,
            zeroshot_topic_list=valid_topics,
            zeroshot_min_similarity=args.alignment_with_predefined_topics,
            verbose=args.verbose,
        )
        topics, _ = topic_model.fit_transform(texts, embeddings)
        topic_info = topic_model.get_topic_info()
        print(topic_info.head(50))

        df_paragraphs_sample = df_paragraphs_sample.copy()
        df_paragraphs_sample["Topic"] = topics
        df_paragraphs_sample.to_parquet(args.clustered_paragraphs_path, index=False)
        topic_info.to_csv(args.topic_info_path, index=False)
        topic_model.save(args.topic_model_path, serialization="safetensors")

        print(f"Saved clustered paragraphs to: {args.clustered_paragraphs_path}")
        print(f"Saved topic info to: {args.topic_info_path}")
        print(f"Saved BERTopic model to: {args.topic_model_path}")

    elif args.task == "refine_topics":
        print("Loading BERTopic model and clustered paragraphs...")
        topic_model = BERTopic.load(args.topic_model_path, embedding_model=SentenceTransformer(args.embedding_model))
        clustered_df = pd.read_parquet(args.clustered_paragraphs_path)

        print("Refining topic labels")
        qwen = QwenTopic(model="sdsc-qwen-32b", nr_docs=args.top_n_docs)
        topic_model.update_topics(
            docs=clustered_df["TEXT_SPAN"].tolist(),
            topics=clustered_df["Topic"].tolist(),
            ctfidf_model=topic_model.c_tf_idf_,
            representation_model=qwen,
        )
        updated_info = topic_model.get_topic_info()
        updated_info.to_csv(args.topic_refinement_output, index=False)
        print(f"Saved refined topic info to: {args.topic_refinement_output}")
        print(updated_info.head(50))

        topic_model.save(args.topic_model_path, serialization="safetensors")
        print(f"Updated BERTopic model saved to: {args.topic_model_path}")

    elif args.task == "stance_nli":
        print("Loading clustered paragraphs and hypotheses...")
        clustered_df = pd.read_parquet(args.clustered_paragraphs_path)
        hypotheses, source = build_nli_hypotheses(
            nli_hypotheses_source=args.nli_hypotheses_source,
            valid_topics_file=args.valid_topics_file,
            topic_refinement_output=args.topic_refinement_output,
            refined_topic_column=args.refined_topic_column,
            restrict_refined_to_valid=args.restrict_refined_to_valid_topics,
        )
        print(f"Using {len(hypotheses)} hypotheses from: {source}")

        print(f"Running NLI with model: {args.nli_model_name}")
        pair_df, top_df = run_nli_stance(
            text_df=clustered_df,
            hypotheses=hypotheses,
            nli_model_name=args.nli_model_name,
            entailment_threshold=args.entailment_threshold,
            batch_size=args.nli_batch_size,
            max_texts=args.max_texts_for_nli,
        )

        pair_df.to_parquet(args.stance_pairs_output, index=False)
        top_df.to_parquet(args.stance_top_output, index=False)

        print(f"Saved all entailment pairs to: {args.stance_pairs_output}")
        print(f"Saved top predictions to: {args.stance_top_output}")
        print(top_df.head(30))
