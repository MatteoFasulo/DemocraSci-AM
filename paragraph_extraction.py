import sqlite3
import pandas as pd
from neo4j import GraphDatabase
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

from const import DB_FILE, URI, AUTH, NEO4J_DB

def get_sqlite_connection(db_file):
    return sqlite3.connect(db_file)

def get_neo4j_driver(uri, auth):
    return GraphDatabase.driver(uri, auth=auth)


def get_transcripts_from_sqlite(conn, table_name="transcript"):
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, conn)

def fetch_speech_metadata_bulk(
    df, batch_size, fields=["role", "language_of_text", "name"]
):
    ids = df["ID"].astype(str).dropna().unique().tolist()
    rows = []

    query = """
    UNWIND $uids AS uid
    MATCH (n:Speech {uid: uid})
    RETURN n.uid AS ID, n.role AS role, n.language_of_text AS language_of_text, n.name AS name
    """

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=NEO4J_DB) as session:
            for i in tqdm(
                range(0, len(ids), batch_size), desc="Fetching speech metadata"
            ):
                chunk = ids[i : i + batch_size]
                rows.extend(session.run(query, uids=chunk).data())

    meta_df = pd.DataFrame(rows).drop_duplicates(subset=["ID"])
    return meta_df

def merge_metadata(transcript_df, meta_df):
    merged_df = transcript_df.assign(ID=transcript_df["ID"].astype(str)).merge(
        meta_df, on="ID", how="left"
    )
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


if __name__ == "__main__":
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