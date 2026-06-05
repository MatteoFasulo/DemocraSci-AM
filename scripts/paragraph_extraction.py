import sqlite3
import re
from html import unescape
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from const import DB_FILE, URI, AUTH, NEO4J_DB

tqdm.pandas()

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

def get_sqlite_connection(db_file):
    return sqlite3.connect(db_file)


def get_neo4j_driver(uri, auth):
    return GraphDatabase.driver(uri, auth=auth)


def get_transcripts_from_sqlite(conn, table_name="transcript"):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    if "TEXT" in df.columns and "TEXT_SPAN" not in df.columns:
        df = df.rename(columns={"TEXT": "TEXT_SPAN"})
    if "ID" in df.columns and "Speech_ID" not in df.columns:
        df = df.rename(columns={"ID": "Speech_ID"})
    return df


def strip_html(text):
    if pd.isna(text):
        return text
    text = str(text)
    if "<" not in text:
        return unescape(text).strip()

    stripped = _TAG_RE.sub(" ", text)
    stripped = unescape(stripped)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def clean_transcript_rows(df):
    cleaned = df.copy()
    text = cleaned["TEXT_SPAN"].astype("string")
    html_mask = text.str.contains("<", na=False)
    cleaned.loc[~html_mask, "TEXT_SPAN"] = text.loc[~html_mask].str.strip().map(unescape)
    cleaned.loc[html_mask, "TEXT_SPAN"] = (
        text.loc[html_mask]
        .str.replace(_TAG_RE, " ", regex=True)
        .map(unescape)
        .str.replace(_WHITESPACE_RE, " ", regex=True)
        .str.strip()
    )
    cleaned = cleaned[cleaned["TEXT_SPAN"].notna() & (cleaned["TEXT_SPAN"] != "")]
    cleaned = cleaned[cleaned["name"].notna() & (cleaned["name"].astype(str).str.strip() != "")]
    return cleaned


def collapse_paragraphs_by_speech(df):
    sort_cols = [col for col in ["Speech_ID", "Paragraph_ID"] if col in df.columns]
    ordered = df.sort_values(sort_cols, kind="stable") if sort_cols else df

    agg_map = {"TEXT_SPAN": lambda s: "\n\n".join(s.astype(str))}
    for col in [
        "name",
        "role",
        "language_of_text",
        "Partition_ID",
        "Discussion_ID",
    ]:
        if col in ordered.columns:
            agg_map[col] = "first"

    for col in [
        "Paragraph_ID",
        "order_stamp",
        "time_start",
        "time_end",
        "length",
    ]:
        if col in ordered.columns:
            agg_map[col] = "first"

    group_cols = ["Speech_ID"]
    aggregated = ordered.groupby(group_cols, as_index=False, dropna=False).agg(agg_map)

    preferred_order = [
        "Speech_ID",
        "Debate_ID",
        "Debate_Text",
        "Discussion_ID",
        "Partition_ID",
        "Paragraph_ID",
        "name",
        "role",
        "language_of_text",
        "order_stamp",
        "time_start",
        "time_end",
        "length",
        "TEXT_SPAN",
    ]
    existing = [col for col in preferred_order if col in aggregated.columns]
    remaining = [col for col in aggregated.columns if col not in existing]
    return aggregated[existing + remaining]


def fetch_speech_metadata_bulk(
    df, batch_size, label_query=":Speech", fields=None
):
    if fields is None:
        fields = [
            "role",
            "language_of_text",
            "name",
            "Partition_ID",
            "Discussion_ID",
            "Debate_ID",
            "Debate_Text",
        ]

    ids = (
        df["Speech_ID"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    rows = []

    return_clause = ", ".join(
        [
            f"n.{field} AS {field}"
            for field in fields
            if field
            not in {"Partition_ID", "Discussion_ID", "Debate_ID", "Debate_Text"}
        ]
    )
    query = """
    UNWIND $uids AS uid
    MATCH (n{label_query} {{uid: uid}})
    OPTIONAL MATCH (n)-[:CONTAINS]-(partition:Partition)
    OPTIONAL MATCH (partition)-[:DIVIDED_IN]-(discussion:Discussion)
    OPTIONAL MATCH (discussion)-[:DISCUSSED_IN]-(debate:Debate)
    RETURN n.uid AS ID,
           {return_clause},
           partition.uid AS Partition_ID,
           discussion.uid AS Discussion_ID,
           debate.uid AS Debate_ID,
           coalesce(debate.text, debate.name, debate.title) AS Debate_Text
    """.format(label_query=label_query, return_clause=return_clause)

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=NEO4J_DB) as session:
            for i in tqdm(
                range(0, len(ids), batch_size), desc="Fetching speech metadata"
            ):
                chunk = ids[i : i + batch_size]
                rows.extend(session.run(query, uids=chunk).data())

    meta_df = pd.DataFrame(rows).drop_duplicates(subset=["ID"])
    matched_ids = set(meta_df["ID"].astype(str)) if not meta_df.empty else set()
    missing_ids = sorted(set(ids) - matched_ids, key=int)

    print(f"Matched {len(matched_ids):,} / {len(ids):,} transcript IDs in Neo4j.")
    if missing_ids:
        print(f"Missing {len(missing_ids):,} IDs. Sample: {missing_ids[:20]}")
        pd.Series(missing_ids, name="missing_id").to_csv(
            "missing_speech_ids.csv", index=False
        )
        print("Saved missing IDs to missing_speech_ids.csv")

    return meta_df


def merge_metadata(transcript_df, meta_df):
    merged_df = transcript_df.assign(Speech_ID=transcript_df["Speech_ID"].astype(str)).merge(
        meta_df.rename(columns={"ID": "Speech_ID"}), on="Speech_ID", how="left"
    )
    return merged_df


if __name__ == "__main__":
    print("Connecting to SQLite and fetching transcripts...")
    conn = get_sqlite_connection(DB_FILE)
    transcript_df = get_transcripts_from_sqlite(conn)

    print("Fetching metadata from Neo4j...")
    meta_df = fetch_speech_metadata_bulk(transcript_df, batch_size=10_000)

    print("Merging metadata with transcripts...")
    merged_df = merge_metadata(transcript_df, meta_df)
    print("Dropping unknown speakers and stripping HTML...")
    merged_df = clean_transcript_rows(merged_df)
    print("Collapsing paragraphs into one row per speech and speaker...")
    merged_df = collapse_paragraphs_by_speech(merged_df)
    print("Saving merged data to CSV...")
    merged_df.to_csv("data/speeches_with_text.csv", index=False)
