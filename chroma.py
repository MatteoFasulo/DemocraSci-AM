"""Build the persistent Chroma bill and paragraph collections."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import chromadb
import pandas as pd
import torch
from tqdm import tqdm

from scripts.chroma_store import (
    BILLS_COLLECTION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHROMA_PATH,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    METADATA_PATH,
    PARAGRAPHS_COLLECTION,
    bill_records_for_chroma,
    build_bill_records,
    create_collection,
    embedding_function,
    load_statutory_text,
    paragraph_records,
    prepare_paragraphs,
)

logger = logging.getLogger("democrasci.chroma")


def upsert_records(
    collection: chromadb.Collection,
    ids: list,
    documents: list,
    metadatas: list,
    batch_size: int,
    description: str,
) -> None:
    for start in tqdm(range(0, len(ids), batch_size), desc=description):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


@click.command(context_settings={"show_default": True})
@click.option(
    "--input-data",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=Path("data/speeches_with_text.parquet"),
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_CHROMA_PATH,
)
@click.option("--bill-collection", default=BILLS_COLLECTION)
@click.option("--paragraph-collection", default=PARAGRAPHS_COLLECTION)
@click.option(
    "--metadata-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=METADATA_PATH,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=DEFAULT_BATCH_SIZE)
@click.option("--bills-ef-search", type=click.IntRange(min=1), default=DEFAULT_HNSW_EF_SEARCH)
@click.option(
    "--bills-ef-construction",
    type=click.IntRange(min=1),
    default=DEFAULT_HNSW_EF_CONSTRUCTION,
)
@click.option("--paragraphs-ef-search", type=click.IntRange(min=1), default=DEFAULT_HNSW_EF_SEARCH)
@click.option(
    "--paragraphs-ef-construction",
    type=click.IntRange(min=1),
    default=DEFAULT_HNSW_EF_CONSTRUCTION,
)
@click.option(
    "--recreate/--no-recreate",
    default=False,
    help="Delete and rebuild both collections first.",
)
def cli(
    input_data: Path,
    db_path: Path,
    bill_collection: str,
    paragraph_collection: str,
    metadata_path: Path,
    batch_size: int,
    bills_ef_search: int,
    bills_ef_construction: int,
    paragraphs_ef_search: int,
    paragraphs_ef_construction: int,
    recreate: bool,
):
    """Ingest the canonical paragraph corpus into two Chroma collections."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw = pd.read_parquet(input_data)
    speeches = prepare_paragraphs(raw)
    statutory_text = load_statutory_text(metadata_path)
    bills = build_bill_records(speeches, statutory_text)
    logger.info("Prepared %d paragraphs and %d bills", len(speeches), len(bills))
    logger.info("Bills without statutory text: %d", int(bills["statutory_text"].eq("").sum()))

    client = chromadb.PersistentClient(path=str(db_path))
    ef = embedding_function("cuda" if torch.cuda.is_available() else "cpu")
    bills_store = create_collection(client, bill_collection, ef, bills_ef_search, bills_ef_construction, recreate)
    paragraphs_store = create_collection(
        client,
        paragraph_collection,
        ef,
        paragraphs_ef_search,
        paragraphs_ef_construction,
        recreate,
    )

    bill_ids, bill_documents, bill_metadatas = bill_records_for_chroma(bills)
    paragraph_ids, paragraph_documents, paragraph_metadatas = paragraph_records(speeches)
    upsert_records(
        bills_store,
        bill_ids,
        bill_documents,
        bill_metadatas,
        batch_size,
        "Ingesting bills",
    )
    upsert_records(
        paragraphs_store,
        paragraph_ids,
        paragraph_documents,
        paragraph_metadatas,
        batch_size,
        "Ingesting paragraphs",
    )
    logger.info(
        "Collections contain %d bills and %d paragraphs",
        bills_store.count(),
        paragraphs_store.count(),
    )


if __name__ == "__main__":
    cli()
