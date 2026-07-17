import sqlite3
import re
import unicodedata
from html import unescape
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from tqdm import tqdm

try:
    from const import DB_FILE, URI, AUTH, NEO4J_DB
except ImportError:  # Supports importing this file as scripts.paragraph_extraction.
    from .const import DB_FILE, URI, AUTH, NEO4J_DB

tqdm.pandas()

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# These values are identifiers, not measurements.  Keeping them textual is
# important because values such as 06.0780 are meaningful exactly as written.
IDENTIFIER_COLUMNS = {
    "Speech_ID",
    "Partition_ID",
    "Discussion_ID",
    "Debate_ID",
    "Bill_Number",
    "Affair_ID",
    "meeting_uid",
    "legislative_period_uid",
}

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

def get_affair_from_bill_number(bill_number: str, timeout=20) -> str:
    """Return the OpenParldata affair ID for an exact bill number match."""
    bill_number = str(bill_number).strip()
    if not bill_number:
        return pd.NA
    url = f"https://api.openparldata.ch/v1/affairs/?search={bill_number}"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    for affair in payload.get("data", []):
        if str(affair.get("number", "")).strip() == bill_number and len(str(affair.get("external_id", ""))) == 8:
            external_id = affair.get("external_id")
            return str(external_id) if external_id is not None else pd.NA
    return pd.NA


def enrich_bill_numbers(df, max_workers=16):
    """Resolve bill numbers to OpenParldata affair IDs."""
    if "Bill_Number" not in df.columns:
        return df
    enriched = df.copy()
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")
    bill_numbers = {
        str(value).strip()
        for value in enriched["Bill_Number"].dropna()
        if str(value).strip()
    }
    cache = {}

    def lookup(short_number):
        try:
            return short_number, get_affair_from_bill_number(short_number)
        except Exception as exc:
            print(f"Warning: could not resolve bill number {short_number!r}: {exc}")
            return short_number, pd.NA

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(lookup, number) for number in bill_numbers]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resolving affair IDs"):
            number, affair_id = future.result()
            cache[number] = affair_id

    resolved = enriched["Bill_Number"].map(
        lambda value: cache.get(str(value).strip(), pd.NA) if pd.notna(value) else pd.NA
    )
    enriched["Bill_Number"] = enriched["Bill_Number"].astype("string")
    enriched["Affair_ID"] = resolved.astype("string")
    return enriched

def get_sqlite_connection(db_file):
    return sqlite3.connect(str(db_file))


def get_neo4j_driver(uri, auth):
    return GraphDatabase.driver(uri, auth=auth)


def get_transcripts_from_sqlite(conn, table_name="transcript"):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"Invalid SQLite table name: {table_name!r}")
    query = f'SELECT * FROM "{table_name}"'
    df = pd.read_sql_query(query, conn)
    if "TEXT" in df.columns and "TEXT_SPAN" not in df.columns:
        df = df.rename(columns={"TEXT": "TEXT_SPAN"})
    if "ID" in df.columns and "Speech_ID" not in df.columns:
        df = df.rename(columns={"ID": "Speech_ID"})
    # Speaker names are supplied by Neo4j metadata, not by the SQLite
    # transcript table in all database versions.
    required = {"Speech_ID", "TEXT_SPAN"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Transcript table is missing required columns: {sorted(missing)}")
    df = preserve_identifier_columns(df)
    return df


def preserve_identifier_columns(df):
    """Convert identifier columns to nullable strings without changing values."""
    result = df.copy()
    for column in IDENTIFIER_COLUMNS.intersection(result.columns):
        result[column] = result[column].astype("string")
    return result


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
    required = {"TEXT_SPAN", "name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cannot clean transcripts; missing columns: {sorted(missing)}")
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
    required = {"Speech_ID", "TEXT_SPAN"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cannot extract paragraphs; missing columns: {sorted(missing)}")
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
    df, batch_size, label_query=":Speech", fields=None, uri=URI, auth=AUTH,
    database=NEO4J_DB, missing_ids_path="missing_speech_ids.csv",
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if "Speech_ID" not in df.columns:
        raise ValueError("Input data is missing Speech_ID")
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

    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field) for field in fields):
        raise ValueError("Metadata field names must be simple Neo4j property names")
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
           // Bill numbers are identifiers; never let the driver/pandas infer
           // them as numbers (for example, 06.4050 must remain text).
           toString(head(collect(DISTINCT bill.bill_number))) AS Bill_Number,
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

    if not ids:
        return pd.DataFrame(columns=["ID", *fields])

    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=database) as session:
            for i in tqdm(
                range(0, len(ids), batch_size), desc="Fetching speech metadata"
            ):
                chunk = ids[i : i + batch_size]
                rows.extend(session.run(query, uids=chunk).data())

    meta_df = pd.DataFrame(rows)
    if meta_df.empty:
        meta_df = pd.DataFrame(columns=["ID", *fields])
    meta_df = meta_df.drop_duplicates(subset=["ID"])
    meta_df = preserve_identifier_columns(meta_df)
    matched_ids = set(meta_df["ID"].astype(str)) if not meta_df.empty else set()
    missing_ids = sorted(set(ids) - matched_ids)

    print(f"Matched {len(matched_ids):,} / {len(ids):,} transcript IDs in Neo4j.")
    if missing_ids:
        print(f"Missing {len(missing_ids):,} IDs")
        pd.Series(missing_ids, name="missing_id").to_csv(missing_ids_path, index=False)
        print(f"Saved missing IDs to {missing_ids_path}")

    return meta_df


def merge_metadata(transcript_df, meta_df):
    if "Speech_ID" not in transcript_df.columns:
        raise ValueError("Transcript data is missing Speech_ID")
    if "ID" not in meta_df.columns:
        raise ValueError("Metadata data is missing ID")
    merged_df = transcript_df.assign(Speech_ID=transcript_df["Speech_ID"].astype(str)).merge(
        meta_df.rename(columns={"ID": "Speech_ID"}), on="Speech_ID", how="left"
    )
    return preserve_identifier_columns(merged_df)


def fetch_mp_metadata(uri=URI, auth=AUTH, database=NEO4J_DB):
    """Fetch the MP lookup directly from Neo4j (the old CSV is obsolete)."""
    query = """
    MATCH (p:`Unique Person`)-[:MEMBER_OF]->(party:Party)
    RETURN DISTINCT p.uid AS uid, p.first_name AS first_name,
           p.last_name AS last_name,
           p.first_name + ' ' + p.last_name AS fullname,
           p.gender AS gender, party.abbrev AS party, party.name AS party_name
    ORDER BY last_name ASC
    """
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=database) as session:
            return pd.DataFrame(session.run(query).data())


def add_party_metadata(df, mp_metadata_path=None, uri=URI, auth=AUTH, database=NEO4J_DB):
    # Keep the optional path for backwards compatibility, but use Neo4j by
    # default so the pipeline does not depend on a generated local artifact.
    mp_df = pd.read_csv(mp_metadata_path) if mp_metadata_path else fetch_mp_metadata(uri, auth, database)
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
    df = preserve_identifier_columns(df)
    if "Bill_Labels" in df.columns:
        df["Bill_Labels"] = (
            df["Bill_Labels"].astype("string")
            .str.replace("Bill|", "", regex=False)
            .str.strip("|")
        )

    removed_columns = {
        "source_name", "source_abbrev", "meeting_uid", "meeting_number",
        "meeting_datetime", "legislative_period_uid",
    }
    df = df.drop(columns=removed_columns, errors="ignore")
    df = df.rename(columns={
        "Speech_ID": "speech_id",
        "Paragraph_ID": "paragraph_id",
        "speaker_uid": "speaker_id",
        "speaker_first_name": "speaker_first_name",
        "speaker_last_name": "speaker_last_name",
        "speaker_fullname": "speaker_full_name",
        "Bill_Number": "bill_number",
        "Affair_ID": "Affair_ID",
        "Bill_Title_DE": "bill_title_de",
        "Bill_Title_FR": "bill_title_fr",
        "Bill_Labels": "bill_labels",
        "Debate_ID": "debate_id",
        "Discussion_ID": "discussion_id",
        "Partition_ID": "partition_id",
        "name": "speaker_name",
        "role": "speaker_role",
        "language_of_text": "language",
        "TEXT_SPAN": "text",
    })
    preferred_order = [
        "speech_id",
        "speaker_id",
        "speaker_first_name",
        "speaker_last_name",
        "speaker_full_name",
        "party",
        "party_name",
        "gender",
        "meeting_date",
        "session_name",
        "year",
        "debate_id",
        "bill_number",
        "Affair_ID",
        "bill_title_de",
        "bill_title_fr",
        "bill_labels",
        "discussion_id",
        "partition_id",
        "paragraph_id",
        "speaker_name",
        "speaker_role",
        "language",
        "order_stamp",
        "time_start",
        "time_end",
        "length",
        "text",
    ]
    existing = [col for col in preferred_order if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    return df[existing + remaining]


if __name__ == "__main__":
    print("Connecting to SQLite and fetching transcripts...")
    with get_sqlite_connection(DB_FILE) as conn:
        transcript_df = get_transcripts_from_sqlite(conn)

    print("Fetching metadata from Neo4j...")
    meta_df = fetch_speech_metadata_bulk(transcript_df, batch_size=10_000)

    print("Merging metadata with transcripts...")
    merged_df = merge_metadata(transcript_df, meta_df)
    print("Resolving canonical bill numbers...")
    merged_df = enrich_bill_numbers(merged_df)
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
