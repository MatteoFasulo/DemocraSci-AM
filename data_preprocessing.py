"""Prepare parliamentary speech data and enrich bills with OpenParldata metadata.

The public pipeline is ``raw speech dataframe -> df_preprocessing ->
df_bill_metadata``. Metadata is cached locally so normal pipeline runs do not
repeatedly call the OpenParldata API.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
import re
import requests
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

METADATA_PATH = Path("data/bill_metadata_cache.json")
DEFAULT_BILL_TYPES = ("Motion", "Parliamentary Initiative")

def fetch_bill_metadata(affair_id: str) -> dict[str, str] | None:
    """Fetch German bill texts and source URLs for one OpenParldata affair ID."""
    URL = f"https://api.openparldata.ch/v1/affairs/?search={affair_id}"
    response = requests.get(URL, headers={"Accept": "application/json"}, timeout=20)
    if not response.ok:
        return None

    res = response.json()
    data = res.get("data", [])
    links = data[0].get("links", {})
    texts = links.get("texts")

    if texts is None:
        return None

    # We found the external texts, now we need to find the ideological summary and statutory text
    response = requests.get(texts, headers={"Accept": "application/json"}, timeout=20)
    if not response.ok:
        return None

    res = response.json()
    new_data = res.get("data", [])
    if not new_data:
        return None

    # We have data
    ideological_summary = ""
    statutory_text = ""

    for entry in new_data:
        entry_type = entry.get("type") or {}
        type_de = entry_type.get("de", "")
        type_fr = entry_type.get("fr", "")

        text_dict = entry.get("text") or {}
        text_de = text_dict.get("de", "")

        if type_de == "Begründung" or type_fr == "Développement":
            ideological_summary = text_de
        elif type_de == "Eingereichter Text" or type_fr == "Texte déposé":
            statutory_text = text_de

    return {
        "ideological_summary": ideological_summary,
        "statutory_text": statutory_text,
        "url_api": data[0].get("url_api"),
        "url_external": data[0].get("url_external")["de"],
    }


def build_bill_metadata_cache(df: pd.DataFrame, path: Path = METADATA_PATH) -> None:
    """Fetch and persist metadata for every bill when the cache is absent.

    Fetches are concurrent because this is an I/O-bound operation. Existing
    cache files are treated as authoritative and are not refreshed implicitly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    cache = {}
    unique_ids = df["Affair_ID"].dropna().astype(str).drop_duplicates().tolist()
    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() // 2)) as executor:
        future_to_id = {
            executor.submit(fetch_bill_metadata, affair_id): affair_id
            for affair_id in unique_ids
        }

        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Fetching bill metadata"):
            affair_id = future_to_id[future]

            try:
                metadata = future.result()
                if metadata is not None:
                    cache[affair_id] = {
                        "ideological_summary": clean_text(metadata.get("ideological_summary", "") or ""),
                        "statutory_text": clean_text(metadata.get("statutory_text", "") or ""),
                        "url_api": metadata.get("url_api"),
                        "url_external": metadata.get("url_external"),
                    }
            except Exception as e:
                print(f"Error fetching ID {affair_id}: {e}")

    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bill_metadata(path: Path = METADATA_PATH) -> dict[str, dict[str, Any]]:
    """
    Load bill metadata from a JSON file.
    Returns a dictionary mapping Affair_ID to metadata dictionaries.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: Any) -> str:
    """
    Clean and normalize text for processing.
    - Converts to string, unescapes HTML entities, and removes HTML tags.
    - Collapses whitespace and trims leading/trailing spaces.
    - Returns an empty string for None or NaN values.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = " ".join(str(item) for item in value.values())
    elif pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_paragraphs(
    df: pd.DataFrame,
    patterns: Iterable[str] = (
        r"\[VS\]",  # voting session
        r"\[PAGE \d+\]",  # page number
        r"\[GZ\]", # GZ reference
    ),
) -> pd.DataFrame:
    """
    A series of patterns to remove from paragraphs

    - [VS] -> voting session
    - [PAGE 1240] -> page number
    - [GZ] -> GZ reference
    """
    # Copy the dataframe to avoid modifying the original
    df = df.copy()

    # Compile the regex patterns
    compiled_patterns = [re.compile(pattern) for pattern in patterns]

    # Function to remove patterns from a single text
    def remove_patterns(text: str) -> str:
        for pattern in compiled_patterns:
            text = pattern.sub("", text)
        return text.strip()

    # Apply the function to the 'text' column
    df["text"] = df["text"].apply(remove_patterns)

    # Remove rows where the 'text' column is empty after cleaning (i.e., only contained the patterns)
    df = df[df["text"] != ""]

    return df

def df_preprocessing(
    df: pd.DataFrame,
    bill_types: Iterable[str] = DEFAULT_BILL_TYPES,
    metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Filter speech rows, clean text, and attach cached bill metadata.

    Rows are retained when their ``bill_labels`` match ``bill_types`` and both
    ``bill_number`` and ``text`` are present. Metadata is joined by ``Affair_ID``;
    missing API metadata is represented by empty fields rather than discarded.
    """
    filtered = df[df["bill_labels"].isin(tuple(bill_types))]
    filtered = filtered.dropna(subset=["bill_number", "text"]).copy().reset_index(drop=True)
    filtered["text"] = filtered["text"].map(clean_text)
    filtered = remove_paragraphs(filtered) # New: Remove unwanted patterns from the 'text' column
    filtered = filtered[filtered["text"].ne("")].reset_index(drop=True)

    build_bill_metadata_cache(filtered, metadata_path)
    metadata = load_bill_metadata(metadata_path)
    records = filtered["Affair_ID"].map(
        lambda affair_id: metadata.get(str(affair_id), {}) if pd.notna(affair_id) else {}
    )
    normalized = pd.json_normalize(records).add_suffix("_metadata")
    normalized.index = filtered.index
    filtered["metadata_available"] = records.map(bool)
    result = pd.concat([filtered, normalized], axis=1)

    # Preserve the corpus-derived identifier; API metadata is auxiliary and may be absent.
    for field in ("ideological_summary", "statutory_text", "source_url"):
        metadata_field = f"{field}_metadata"
        if metadata_field in result:
            result[field] = result[metadata_field].map(clean_text)
        elif field not in result:
            result[field] = ""
    return result


def df_bill_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one searchable record per bill.

    ``bill_search_text`` intentionally combines only the German bill title and
    German statutory text. This is the text embedded for bill-level retrieval;
    speech paragraphs are retrieved in the next stage.
    """
    bill_meta = (
        df.dropna(subset=["Affair_ID"])
        .groupby("Affair_ID", as_index=False)
        .agg(
            bill_number=("bill_number", "first"),
            bill_title_de=("bill_title_de", "first"),
            bill_title_fr=("bill_title_fr", "first"),
            ideological_summary=("ideological_summary", "first"),
            statutory_text=("statutory_text", "first"),
            metadata_available=("metadata_available", "max"),
            n_paragraphs=("text", "size"),
            n_speeches=("speech_id", "nunique"),
        )
    )
    search_fields = ("bill_title_de", "statutory_text")
    bill_meta["bill_search_text"] = bill_meta.apply(
        lambda row: clean_text(" ".join(clean_text(row.get(field, "")) for field in search_fields)),
        axis=1,
    )
    return bill_meta
