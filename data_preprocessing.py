from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


METADATA_PATH = Path("data/bill_metadata_cache.json")
DEFAULT_BILL_TYPES = ("Bill|Motion", "Bill|Parliamentary Initiative")


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


def normalize_swiss_bill(value: Any) -> str | float:
    """
    Normalize Swiss bill identifiers to a standard format.
    - Pads the year part to 2 digits and the number part to 3 digits.
    - Returns NaN for None or NaN values.
    """
    if pd.isna(value) or str(value).strip().lower() == "nan":
        return np.nan
    value_str = str(value).strip()
    if "." not in value_str:
        return value_str
    left, right = value_str.split(".", 1)
    return f"{left.zfill(2)}.{right.ljust(3, '0')}"


def business_number_to_affair_id(normalized_str: Any, year_pivot: int = 50) -> str | float:
    """
    Convert a normalized Swiss bill identifier to an Affair_ID.
    - Uses a pivot year to determine the full year (e.g., 50 -> 1950, 49 -> 2049).
    - Returns NaN for None or NaN values.
    """
    if pd.isna(normalized_str) or str(normalized_str).strip().lower() == "nan":
        return np.nan
    left, right = str(normalized_str).split(".", 1)
    year_short = int(left)
    year_full = 2000 + year_short if year_short <= year_pivot else 1900 + year_short
    return f"{year_full}{str(int(right)).zfill(4)}"


def df_preprocessing(
    df: pd.DataFrame,
    bill_types: Iterable[str] = DEFAULT_BILL_TYPES,
    metadata_path: Path = METADATA_PATH,
    year_pivot: int = 50,
) -> pd.DataFrame:
    """
    Preprocess the bill DataFrame by filtering, cleaning, and enriching with metadata.
    - Filters rows based on specified bill types.
    - Cleans text fields and normalizes bill identifiers.
    - Maps Affair_IDs to metadata from a JSON file.
    """
    filtered = df[df["Bill_Labels"].isin(tuple(bill_types))]
    filtered = filtered.dropna(subset=["Bill_Number", "TEXT_SPAN"]).copy().reset_index(drop=True)
    filtered["TEXT_SPAN"] = filtered["TEXT_SPAN"].map(clean_text)
    filtered = filtered[filtered["TEXT_SPAN"].ne("")].reset_index(drop=True)
    filtered["Bill_Number"] = filtered["Bill_Number"].map(normalize_swiss_bill)
    filtered["Affair_ID"] = filtered["Bill_Number"].map(
        lambda value: business_number_to_affair_id(value, year_pivot=year_pivot)
    )

    metadata = load_bill_metadata(metadata_path)
    records = filtered["Affair_ID"].map(
        lambda affair_id: metadata.get(str(affair_id), {}) if pd.notna(affair_id) else {}
    )
    normalized = pd.json_normalize(records).add_suffix("_metadata")
    normalized.index = filtered.index
    filtered["metadata_available"] = records.map(bool)
    result = pd.concat([filtered, normalized], axis=1)

    # Preserve the corpus-derived identifier; API metadata is auxiliary and may be absent.
    for field in ("title", "keywords", "ideological_summary", "statutory_text", "source_url"):
        metadata_field = f"{field}_metadata"
        if metadata_field in result:
            result[field] = result[metadata_field].map(clean_text)
        elif field not in result:
            result[field] = ""
    return result


def df_bill_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bill metadata from the preprocessed DataFrame.
    - Groups by Affair_ID and collects relevant fields.
    - Cleans and combines text fields for search purposes.
    """
    bill_meta = (
        df.dropna(subset=["Affair_ID"])
        .groupby("Affair_ID", as_index=False)
        .agg(
            Bill_Number=("Bill_Number", "first"),
            Bill_Title_DE=("Bill_Title_DE", "first"),
            Bill_Title_FR=("Bill_Title_FR", "first"),
            title=("title", "first"),
            keywords=("keywords", "first"),
            ideological_summary=("ideological_summary", "first"),
            statutory_text=("statutory_text", "first"),
            metadata_available=("metadata_available", "max"),
            n_paragraphs=("TEXT_SPAN", "size"),
            n_speeches=("Speech_ID", "nunique"),
        )
    )
    search_fields = ("Bill_Title_DE", "Bill_Title_FR", "title", "keywords", "ideological_summary")
    bill_meta["bill_search_text"] = bill_meta.apply(
        lambda row: clean_text(" ".join(clean_text(row.get(field, "")) for field in search_fields)),
        axis=1,
    )
    return bill_meta
