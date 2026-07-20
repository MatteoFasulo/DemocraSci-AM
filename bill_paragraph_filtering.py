"""Retrieve bills and rank evidence passages within selected bills.

Bill retrieval uses a multilingual E5 bi-encoder for high recall. Paragraphs
are reduced to an embedding dense pool and reranked with a BGE cross-encoder
for higher precision.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer, CrossEncoder

from const import RERANKER_MODEL, EMBEDDING_MODEL, RERANK_THRESHOLD

DEFAULT_BILL_EMBEDDINGS_CACHE = Path("data/bill_embeddings.safetensors")


def load_embedding_model(device: torch.device | str | None = None) -> SentenceTransformer:
    """Load the configured multilingual E5 embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def load_reranker_model(device: torch.device | str | None = None) -> CrossEncoder:
    """Load the configured BGE cross-encoder reranker."""
    return CrossEncoder(RERANKER_MODEL, device=device, max_length=512)

def encode_queries(
    model: SentenceTransformer,
    texts: str | Iterable[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode retrieval queries with the E5 ``query:`` prefix."""
    if isinstance(texts, str):
        texts = [texts]

    return model.encode(
        [f"query: {str(text).strip()}" for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def encode_passages(
    model: SentenceTransformer,
    texts: str | Iterable[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode searchable passages with the E5 ``passage:`` prefix."""
    if isinstance(texts, str):
        texts = [texts]

    return model.encode(
        [f"passage: {str(text).strip()}" for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

def load_or_build_bill_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    cache_path: Path = DEFAULT_BILL_EMBEDDINGS_CACHE,
) -> np.ndarray:
    """Load bill embeddings from the cache and encode changed bills.

    Cache entries are keyed by ``Affair_ID`` and invalidated when the source
    ``bill_search_text`` hash changes.
    """
    ids = df["Affair_ID"].astype(str).to_numpy()
    texts = df["bill_search_text"].fillna("").astype(str).to_numpy()
    text_hashes = np.array([sha256(text.encode()).hexdigest() for text in texts])
    cached_embeddings: dict[str, np.ndarray] = {}
    cached_hashes: dict[str, str] = {}
    if cache_path.exists():
        try:
            with safe_open(str(cache_path), framework="numpy") as cached:
                matrix = cached.get_tensor("bill_embeddings")
                cached_ids = cached.get_tensor("affair_ids").astype(str)
                cached_hashes = json.loads(cached.metadata().get("text_hashes", "{}"))
            cached_embeddings = dict(zip(cached_ids, matrix, strict=False))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    missing = [i for i, affair_id in enumerate(ids) if cached_hashes.get(affair_id) != text_hashes[i]]
    if missing:
        new_embeddings = encode_queries(model, texts[missing]).astype(np.float32)
        for row, embedding in zip(missing, new_embeddings, strict=False):
            cached_embeddings[ids[row]] = embedding

    embeddings = np.vstack([cached_embeddings[affair_id] for affair_id in ids]).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"bill_embeddings": embeddings, "affair_ids": np.asarray([int(affair_id) for affair_id in ids], dtype=np.int64)},
        str(cache_path),
        metadata={"text_hashes": json.dumps(dict(zip(ids, text_hashes, strict=False)))},
    )
    return embeddings

def score_bills(
    retrieval_query: str,
    bill_metadata_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    top_k_bills: int,
) -> pd.DataFrame:
    """Return the top bills ranked by cosine similarity to the query."""
    if bill_metadata_df.empty:
        return bill_metadata_df.copy()

    result = bill_metadata_df.copy().reset_index(drop=True)
    query_embedding = encode_queries(embedding_model, retrieval_query)
    bill_embeddings = load_or_build_bill_embeddings(result, embedding_model)
    similarities = query_embedding @ bill_embeddings.T
    result["bill_embedding_score"] = similarities.max(axis=0)
    result["bill_embedding_rank"] = result["bill_embedding_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    result = result.sort_values("bill_embedding_score", ascending=False).head(top_k_bills).copy()
    result["bill_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def select_target_speeches(df: pd.DataFrame, candidate_bills: pd.DataFrame) -> pd.DataFrame:
    """Keep speech paragraphs whose ``Affair_ID`` belongs to a candidate bill."""
    if df.empty or candidate_bills.empty:
        return df.iloc[0:0].copy()
    affair_ids = set(candidate_bills["Affair_ID"].astype(str))
    return df[df["Affair_ID"].astype(str).isin(affair_ids)].copy().reset_index(drop=True)


def score_paragraphs(
    topic: str,
    retrieval_query: str,
    speeches_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    dense_pool_size: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank candidate paragraphs and apply ``RERANK_THRESHOLD``.

    The embedding score selects the dense pool. The cross-encoder assigns
    ``rerank_score`` and determines the final filtered set. The returned tuple
    is ``(pool_before_threshold, filtered_passages)``.
    """
    if speeches_df.empty:
        empty = speeches_df.copy()
        return empty, empty

    passages = speeches_df.copy().reset_index(drop=True)
    query_embeddings = encode_queries(embedding_model, retrieval_query, batch_size=batch_size)
    passage_embeddings = encode_passages(embedding_model, passages["text"], batch_size=batch_size)
    passages["embedding_score"] = (
        query_embeddings @ passage_embeddings.T
    ).ravel()
    dense_pool = (
        passages
        .sort_values("embedding_score", ascending=False)
        .head(min(dense_pool_size, len(passages)))
        .copy()
        .reset_index(drop=True)
    )
    ranked = reranker.rank(
        retrieval_query,
        dense_pool["text"].tolist(),
        top_k=len(dense_pool),
        return_documents=False,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    ranking = pd.DataFrame(ranked)

    pool = ranking.merge(
        dense_pool.reset_index().rename(columns={"index": "corpus_id"}),
        on="corpus_id",
        how="left",
    )
    if "score" in pool.columns:
        pool = pool.rename(columns={"score": "rerank_score"})
    elif "rerank_score" not in pool.columns:
        raise ValueError("Reranker output does not expose a score field")

    pool = pool.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    pool["evidence_rank"] = np.arange(1, len(pool) + 1)
    pool["retrieval_query"] = retrieval_query
    pool["topic"] = topic

    filtered = pool[pool["rerank_score"] >= RERANK_THRESHOLD].copy().reset_index(drop=True)
    filtered["evidence_rank"] = np.arange(1, len(filtered) + 1)
    return pool, filtered
