import sqlite3
import re
import unicodedata
from html import unescape
import pandas as pd
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from tqdm import tqdm

from const import DB_FILE, URI, AUTH, NEO4J_DB

tqdm.pandas()

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

SPEAKER_NAME_ALIASES = {
    "rickli natalie simone": "rickli natalie",
    "roth bernasconi maria": "bernasconi maria",
    "pieren nadja": "umbricht pieren nadja",
    "bruderer pascale": "bruderer wyss pascale",
    "markwalder bar christa": "markwalder christa",
    "diener verena": "diener lenz verena",
    "rutz gregor a": "rutz gregor",
    "humbel naf ruth": "humbel ruth",
    "weichelt picard manuela": "weichelt manuela",
    "haller ursula": "haller vannini ursula",
    "walder nicolas michel": "walder nicolas",
    "ruegger hurschler monika": "ruegger monika",
}


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


def normalize_name(value):
    if pd.isna(value):
        return pd.NA
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold()
    normalized = re.sub(r"[^\w\s]+", " ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized if normalized else pd.NA


def extract_paragraphs_from_text(text):
    if pd.isna(text):
        return []
    soup = BeautifulSoup(str(text), "html.parser")
    p_tags = soup.find_all("p")
    if p_tags:
        paragraphs = [strip_html(tag.get_text(" ", strip=True)) for tag in p_tags]
    else:
        paragraphs = [strip_html(soup.get_text(" ", strip=True))]
    return [paragraph for paragraph in paragraphs if pd.notna(paragraph) and str(paragraph).strip()]


def explode_paragraphs_by_speech(df):
    exploded = df.copy()
    exploded["paragraph_texts"] = exploded["TEXT_SPAN"].map(extract_paragraphs_from_text)
    exploded = exploded[exploded["paragraph_texts"].map(bool)].copy()
    exploded = exploded.explode("paragraph_texts", ignore_index=True)
    exploded["Paragraph_ID"] = exploded.groupby("Speech_ID", sort=False).cumcount() + 1
    exploded["TEXT_SPAN"] = exploded["paragraph_texts"]
    exploded = exploded.drop(columns=["paragraph_texts"])

    preferred_order = [
        "Speech_ID",
        "Paragraph_ID",
        "speaker_uid",
        "speaker_first_name",
        "speaker_last_name",
        "speaker_fullname",
        "party",
        "party_name",
        "gender",
        "source_name",
        "source_abbrev",
        "meeting_uid",
        "meeting_number",
        "meeting_datetime",
        "meeting_date",
        "session_name",
        "year",
        "legislative_period_uid",
        "Debate_ID",
        "Bill_Number",
        "Bill_Title_DE",
        "Bill_Title_FR",
        "Bill_Labels",
        "Discussion_ID",
        "Partition_ID",
        "name",
        "role",
        "language_of_text",
        "TEXT_SPAN",
    ]
    existing = [col for col in preferred_order if col in exploded.columns]
    remaining = [col for col in exploded.columns if col not in existing]
    return exploded[existing + remaining]


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
            "Bill_Number",
            "Bill_Title_DE",
            "Bill_Title_FR",
            "Bill_Labels",
            "source_name",
            "source_abbrev",
            "meeting_uid",
            "meeting_number",
            "meeting_datetime",
            "session_name",
            "year",
            "legislative_period_uid",
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
            not in {
                "Partition_ID",
                "Discussion_ID",
                "Debate_ID",
                "Bill_Number",
                "Bill_Title_DE",
                "Bill_Title_FR",
                "Bill_Labels",
                "source_name",
                "source_abbrev",
                "meeting_uid",
                "meeting_number",
                "meeting_datetime",
                "session_name",
                "year",
                "legislative_period_uid",
            }
        ]
    )
    query = """
    UNWIND $uids AS uid
    MATCH (n{label_query} {{uid: uid}})
    OPTIONAL MATCH (n)-[:CONTAINS]-(partition:Partition)
    OPTIONAL MATCH (partition)-[:DIVIDED_IN]-(discussion:Discussion)
    OPTIONAL MATCH (discussion)-[:DISCUSSED_IN]-(debate:Debate)
    OPTIONAL MATCH (discussion)<-[:CONTAINS]-(meeting:Meeting)
    OPTIONAL MATCH (meeting)-[:DURING]->(session:Session)
    OPTIONAL MATCH (meeting)-[:DURING]->(year:Year)
    OPTIONAL MATCH (meeting)-[:DURING]->(lr:`Legislative Period`)
    OPTIONAL MATCH (discussion)-[:PUBLISHED_IN]->(source:Source)
    OPTIONAL MATCH (bill:Bill)-[:CONTAINS]->(debate)
    RETURN n.uid AS ID,
           {return_clause},
           partition.uid AS Partition_ID,
           discussion.uid AS Discussion_ID,
           debate.uid AS Debate_ID,
           head(collect(DISTINCT bill.bill_number)) AS Bill_Number,
           head(collect(DISTINCT bill.title_de)) AS Bill_Title_DE,
           head(collect(DISTINCT bill.title_fr)) AS Bill_Title_FR,
           head(collect(DISTINCT CASE
               WHEN bill IS NULL THEN NULL
               ELSE reduce(
                   label_str = "",
                   label IN labels(bill) |
                   label_str + CASE WHEN label_str = "" THEN "" ELSE "|" END + label
               )
           END)) AS Bill_Labels,
           head(collect(DISTINCT source.name)) AS source_name,
           head(collect(DISTINCT source.abbrev)) AS source_abbrev,
           head(collect(DISTINCT meeting.uid)) AS meeting_uid,
           head(collect(DISTINCT meeting.meeting_number)) AS meeting_number,
           head(collect(DISTINCT meeting.datetime)) AS meeting_datetime,
           head(collect(DISTINCT session.name)) AS session_name,
           head(collect(DISTINCT year.year)) AS year,
           head(collect(DISTINCT lr.uid)) AS legislative_period_uid
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


def add_party_metadata(df, mp_metadata_path="data/mp_metadata.csv"):
    mp_df = pd.read_csv(mp_metadata_path)
    if "fullname" not in mp_df.columns:
        raise ValueError("mp metadata is missing fullname column")

    mp_df = mp_df.copy()
    mp_df["speaker_name_key"] = mp_df["fullname"].map(normalize_name)
    mp_df["match_priority"] = 0
    mp_reverse = mp_df.copy()
    mp_reverse["speaker_name_key"] = mp_reverse["fullname"].map(
        lambda value: normalize_name(" ".join(reversed(str(value).split())))
        if pd.notna(value)
        else pd.NA
    )
    mp_reverse["match_priority"] = 1
    mp_lookup = (
        pd.concat([mp_df, mp_reverse], ignore_index=True)
        .dropna(subset=["speaker_name_key"])
        .sort_values(by=["speaker_name_key", "match_priority"])
        .drop_duplicates(subset=["speaker_name_key"], keep="first")
    )

    alias_rows = []
    for alias_key, canonical_key in SPEAKER_NAME_ALIASES.items():
        match = mp_lookup.loc[mp_lookup["speaker_name_key"] == canonical_key]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["speaker_name_key"] = alias_key
        row["match_priority"] = 2
        alias_rows.append(row)
    if alias_rows:
        alias_df = pd.DataFrame(alias_rows)
        mp_lookup = pd.concat([mp_lookup, alias_df], ignore_index=True)

    enriched = df.copy()
    enriched["speaker_name_key"] = enriched["name"].map(normalize_name)

    party_cols = [
        col
        for col in ["uid", "first_name", "last_name", "fullname", "gender", "party", "party_name"]
        if col in mp_lookup.columns
    ]
    mp_lookup = mp_lookup[["speaker_name_key"] + party_cols]
    enriched = enriched.merge(
        mp_lookup,
        on="speaker_name_key",
        how="left",
    )
    enriched = enriched.rename(
        columns={
            "uid": "speaker_uid",
            "first_name": "speaker_first_name",
            "last_name": "speaker_last_name",
            "fullname": "speaker_fullname",
            "gender": "gender",
            "party": "party",
            "party_name": "party_name",
        }
    )
    if "speaker_fullname" in enriched.columns:
        enriched["speaker_fullname"] = enriched["speaker_fullname"].combine_first(enriched["name"])

    # Retain a small set of blank speaker fields only if the lookup truly fails.
    drop_cols = ["speaker_name_key"]
    return enriched.drop(columns=drop_cols, errors="ignore")


def derive_meeting_date(df):
    enriched = df.copy()
    if "meeting_datetime" in enriched.columns:
        meeting_date = (
            enriched["meeting_datetime"]
            .astype("string")
            .str.slice(0, 10)
            .where(enriched["meeting_datetime"].notna())
        )
        if "meeting_date" in enriched.columns:
            existing = pd.to_datetime(enriched["meeting_date"], errors="coerce").dt.date.astype("string")
            enriched["meeting_date"] = existing.combine_first(meeting_date)
        else:
            enriched["meeting_date"] = meeting_date
    return enriched


def finalize_output_columns(df):
    preferred_order = [
        "Speech_ID",
        "speaker_uid",
        "speaker_first_name",
        "speaker_last_name",
        "speaker_fullname",
        "party",
        "party_name",
        "gender",
        "source_name",
        "source_abbrev",
        "meeting_uid",
        "meeting_number",
        "meeting_datetime",
        "meeting_date",
        "session_name",
        "year",
        "legislative_period_uid",
        "Debate_ID",
        "Bill_Number",
        "Bill_Title_DE",
        "Bill_Title_FR",
        "Bill_Labels",
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
    existing = [col for col in preferred_order if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    return df[existing + remaining]


if __name__ == "__main__":
    print("Connecting to SQLite and fetching transcripts...")
    conn = get_sqlite_connection(DB_FILE)
    transcript_df = get_transcripts_from_sqlite(conn)

    print("Fetching metadata from Neo4j...")
    meta_df = fetch_speech_metadata_bulk(transcript_df, batch_size=10_000)

    print("Merging metadata with transcripts...")
    merged_df = merge_metadata(transcript_df, meta_df)
    print("Adding party metadata from MP mapping...")
    merged_df = add_party_metadata(merged_df)
    print("Exploding speeches into paragraphs and stripping HTML...")
    merged_df = explode_paragraphs_by_speech(merged_df)
    print("Dropping unknown speakers and stripping HTML...")
    merged_df = clean_transcript_rows(merged_df)
    print("Deriving meeting dates and final column order...")
    merged_df = derive_meeting_date(merged_df)
    merged_df = finalize_output_columns(merged_df)
    print("Saving merged data to CSV...")
    merged_df.to_csv("data/speeches_with_text.csv", index=False)
