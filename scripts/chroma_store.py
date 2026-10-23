"""Shared Chroma storage and corpus-preparation helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import torch
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from const import EMBEDDING_MODEL
from data_preprocessing import clean_text, remove_paragraphs

DEFAULT_CHROMA_PATH = Path(
    os.environ.get("CHROMA_PATH", "/mydata/politperformance/shared/matteof/chromadb")
)
BILLS_COLLECTION = "DemocraSci_bills"
PARAGRAPHS_COLLECTION = "DemocraSci_paragraphs"
DEFAULT_HNSW_EF_SEARCH = 100
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_BATCH_SIZE = 4096
METADATA_PATH = Path("data/bill_metadata_cache.json")
ELIGIBLE_BILL_TYPES = ("Motion", "Parliamentary Initiative")

logger = logging.getLogger("democrasci.chroma")


def prepare_paragraphs(raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw parquet rows without fetching external metadata."""
    required = {"bill_labels", "bill_number", "Affair_ID", "text", "speech_id", "paragraph_id"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing paragraph columns: {', '.join(sorted(missing))}")
    speeches = raw[raw["bill_labels"].isin(ELIGIBLE_BILL_TYPES)].copy()
    speeches = speeches.dropna(subset=["bill_number", "Affair_ID", "text"])
    speeches["text"] = speeches["text"].map(clean_text)
    speeches = remove_paragraphs(speeches)
    return speeches[speeches["text"].ne("")].reset_index(drop=True)


def load_statutory_text(path: Path = METADATA_PATH) -> dict[str, str]:
    """Load only statutory text from the existing bill metadata cache."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        cache = json.load(handle)
    return {str(k): clean_text((v or {}).get("statutory_text", "")) for k, v in cache.items()}


def build_bill_records(speeches: pd.DataFrame, statutory_text: dict[str, str] | None = None) -> pd.DataFrame:
    """Build one bill document and metadata record per ``Affair_ID``."""
    statutory_text = statutory_text or {}
    grouped = (
        speeches.assign(Affair_ID=speeches["Affair_ID"].astype(str))
        .groupby("Affair_ID", as_index=False)
        .agg(
            bill_number=("bill_number", "first"),
            bill_title_de=("bill_title_de", "first"),
            bill_title_fr=("bill_title_fr", "first"),
            bill_labels=("bill_labels", "first"),
            n_paragraphs=("text", "size"),
            n_speeches=("speech_id", "nunique"),
        )
    )
    for column in ("bill_title_de", "bill_title_fr"):
        grouped[column] = grouped[column].map(clean_text)
    grouped["statutory_text"] = grouped["Affair_ID"].map(statutory_text).fillna("")
    grouped["bill_search_text"] = grouped.apply(
        lambda row: "\n\n".join(
            part for part in (
                f"Title (DE): {row['bill_title_de']}" if row["bill_title_de"] else "",
                f"Title (FR): {row['bill_title_fr']}" if row["bill_title_fr"] else "",
                f"Statutory text:\n{row['statutory_text']}" if row["statutory_text"] else "",
            ) if part
        ),
        axis=1,
    )
    return grouped


def _metadata_value(value: Any) -> str | int | float | bool:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def paragraph_records(speeches: pd.DataFrame) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return stable IDs, documents, and Chroma-compatible metadata."""
    columns = [c for c in (
        "Affair_ID", "speech_id", "paragraph_id", "bill_number", "bill_title_de",
        "bill_title_fr", "bill_labels", "speaker_id", "speaker_full_name", "party",
        "party_name", "meeting_date", "language",
    ) if c in speeches.columns]
    ids, documents, metadatas = [], [], []
    for row in speeches.itertuples(index=False):
        values = row._asdict()
        ids.append(f"{values['speech_id']}:{values['paragraph_id']}")
        documents.append(str(values["text"]))
        metadatas.append({column: _metadata_value(values[column]) for column in columns})
    return ids, documents, metadatas


def bill_records_for_chroma(bills: pd.DataFrame) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return stable IDs, documents, and Chroma-compatible bill metadata."""
    columns = ["Affair_ID", "bill_number", "bill_title_de", "bill_title_fr", "bill_labels", "n_paragraphs", "n_speeches"]
    ids = bills["Affair_ID"].astype(str).tolist()
    documents = bills["bill_search_text"].tolist()
    metadatas = [{column: _metadata_value(row[column]) for column in columns} for _, row in bills.iterrows()]
    return ids, documents, metadatas


def embedding_function(device: torch.device | str | None = None):
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        normalize_embeddings=True,
    )


def open_chroma_collections(
    db_path: Path = DEFAULT_CHROMA_PATH,
    bills_name: str = BILLS_COLLECTION,
    paragraphs_name: str = PARAGRAPHS_COLLECTION,
    device: torch.device | str | None = None,
) -> tuple[Collection, Collection]:
    """Open pre-built bill and paragraph collections for querying."""
    client = chromadb.PersistentClient(path=str(db_path))
    ef = embedding_function(device)
    try:
        return (
            client.get_collection(name=bills_name, embedding_function=ef),
            client.get_collection(name=paragraphs_name, embedding_function=ef),
        )
    except Exception as exc:
        raise RuntimeError(f"Could not open Chroma collections at {db_path}; run scripts/chroma.py first") from exc


def create_collection(client: chromadb.Client, name: str, ef: Any, ef_search: int, ef_construction: int, recreate: bool) -> Collection:
    """Create/recreate a collection with explicit HNSW settings."""
    if ef_search < 1 or ef_construction < 1:
        raise ValueError("HNSW ef_search and ef_construction must be positive")
    if recreate:
        try:
            client.delete_collection(name=name)
        except Exception as exc:
            message = str(exc).lower()
            if "not found" not in message and "does not exist" not in message:
                raise
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "hnsw_ef_search": ef_search,
        "hnsw_ef_construction": ef_construction,
    }
    # https://docs.trychroma.com/docs/collections/configure#hnsw-index-configuration
    configuration = {
        "hnsw": {
            "space": "cosine",
            "ef_search": ef_search,
            "ef_construction": ef_construction,
            "max_neighbors": 16,
            "batch_size": 100,
        }
    }
    kwargs = {"name": name, "embedding_function": ef, "metadata": metadata, "configuration": configuration}
    if recreate:
        return client.create_collection(**kwargs)
    try:
        collection = client.get_collection(name=name, embedding_function=ef)
    except Exception as exc:
        message = str(exc).lower()
        if "not found" not in message and "does not exist" not in message:
            raise
        return client.create_collection(**kwargs)
    existing_construction = (collection.metadata or {}).get("hnsw_ef_construction")
    if existing_construction is not None and int(existing_construction) != ef_construction:
        raise ValueError(f"Collection {name!r} was built with ef_construction={existing_construction}; use --recreate to change it")
    collection.modify(configuration={"hnsw": {"ef_search": ef_search}}, metadata=metadata)
    return collection
