from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import json

BILLS_METADATA = json.load(open(Path("data/bill_metadata_cache.json"), "r", encoding="utf-8"))

def normalize_swiss_bill(value: Any) -> str | float:
    if pd.isna(value) or str(value).strip().lower() == "nan":
        return np.nan

    value_str = str(value).strip()

    if "." not in value_str:
        return value_str

    left, right = value_str.split(".", 1)
    return f"{left.zfill(2)}.{right.ljust(3, '0')}"

def business_number_to_affair_id(normalized_str: Any) -> str | float:
    if pd.isna(normalized_str) or str(normalized_str).strip().lower() == "nan":
        return np.nan

    left, right = str(normalized_str).split(".", 1)
    year_short = int(left)
    sequence = int(right)

    year_full = 2000 + year_short if year_short <= 26 else 1900 + year_short
    return f"{year_full}{str(sequence).zfill(4)}"

def get_bill_metadata(affair_id: str):
    return BILLS_METADATA.get(affair_id, {})

def df_preprocessing(df: pd.DataFrame,bill_types: list[str] = ["Bill|Motion", "Bill|Parliamentary Initiative"]) -> pd.DataFrame:
    df_filtered = df[df["Bill_Labels"].isin(bill_types)]
    df_filtered = df_filtered.dropna(subset=["Bill_Number", "TEXT_SPAN"]).copy()
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["Bill_Number"] = df_filtered["Bill_Number"].map(normalize_swiss_bill)
    df_filtered["Affair_ID"] = df_filtered["Bill_Number"].map(business_number_to_affair_id)
    df_filtered["Affair_ID"] = df_filtered["Affair_ID"].astype(str)
    df_filtered["bill_metadata"] = df_filtered["Affair_ID"].map(get_bill_metadata, na_action="ignore")
    df_augmented = df_filtered.reset_index(drop=True)
    df_augmented = pd.concat([
        df_augmented.drop(columns=["bill_metadata", "Affair_ID"]),
        pd.json_normalize(df_augmented["bill_metadata"])
    ], axis=1)
    return df_augmented

def df_bill_metadata(df: pd.DataFrame) -> pd.DataFrame:
    bill_meta = (
        df
        .groupby("Affair_ID", as_index=False)
        .agg(
            Bill_Number=("Bill_Number", "first"),
            Bill_Title_DE=("Bill_Title_DE", "first"),
            Bill_Title_FR=("Bill_Title_FR", "first"),
            title=("title", "first"),
            keywords=("keywords", "first"),
            ideological_summary=("ideological_summary", "first"),
            statutory_text=("statutory_text", "first"),
            n_paragraphs=("TEXT_SPAN", "size"),
            n_speeches=("Speech_ID", "nunique"),
        )
    )

    bill_meta["bill_search_text"] = bill_meta.apply(
        lambda row: " ".join([
            #str(row.get("Bill_Number", "")),
            str(row.get("title", "")),
            str(row.get("keywords", "")),
            str(row.get("ideological_summary", "")),
            #str(row.get("statutory_text", "")),
        ]).strip(),
        axis=1
    )

    return bill_meta