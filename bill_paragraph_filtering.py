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
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def load_reranker_model(device: torch.device | str | None = None) -> CrossEncoder:
    # map the scores into 0-1 -> will apply sigmoid function to the score
    return CrossEncoder(RERANKER_MODEL, device=device, max_length=512)


def encode_texts(
    model: SentenceTransformer,
    texts: str | Iterable[str],
    batch_size: int = 256,
) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(
        [f"{str(text).strip()}" for text in texts],
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
        new_embeddings = encode_texts(model, texts[missing]).astype(np.float32)
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
    if bill_metadata_df.empty:
        return bill_metadata_df.copy()

    result = bill_metadata_df.copy().reset_index(drop=True)
    query_embeddings = encode_texts(embedding_model, retrieval_query)
    bill_embeddings = load_or_build_bill_embeddings(result, embedding_model)
    similarities = query_embeddings @ bill_embeddings.T
    result["bill_embedding_score"] = similarities.max(axis=0)
    result["bill_embedding_rank"] = result["bill_embedding_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    result = result.sort_values("bill_embedding_score", ascending=False).head(top_k_bills).copy()
    result["bill_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def select_target_speeches(df: pd.DataFrame, candidate_bills: pd.DataFrame) -> pd.DataFrame:
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
    if speeches_df.empty:
        empty = speeches_df.copy()
        return empty, empty

    passages = speeches_df.copy().reset_index(drop=True)
    query_embeddings = encode_texts(embedding_model, retrieval_query)
    passage_embeddings = encode_texts(embedding_model, passages["text"])
    passages["embedding_score"] = (query_embeddings @ passage_embeddings.T).max(axis=0)
    ranked = reranker.rank(
        retrieval_query,
        passages["text"].tolist(),
        top_k=min(dense_pool_size, len(passages)),
        return_documents=True,
        batch_size=batch_size,
        show_progress_bar=False,
        device=reranker.device,
    )
    pool = pd.DataFrame(ranked)
    if "corpus_id" in pool.columns:
        pool = pool.merge(
            passages.reset_index().rename(columns={"index": "corpus_id"}),
            on="corpus_id",
            how="left",
            suffixes=("", "_orig"),
        )
    elif "text" in pool.columns:
        pool = pool.merge(
            passages,
            left_on="text",
            right_on="text",
            how="left",
            suffixes=("", "_orig"),
        )
    else:
        raise ValueError("Reranker output does not expose corpus_id or text for reconstruction")

    # Keep one canonical passage column for downstream stance detection.
    if "text_orig" in pool.columns:
        pool["text"] = pool["text_orig"]
        pool = pool.drop(columns=["text_orig"])
    elif "text" not in pool.columns:
        raise ValueError("Could not reconstruct the passage text from reranker output")

    if "score" in pool.columns:
        pool["rerank_score"] = pool["score"]
    elif "rerank_score" not in pool.columns:
        raise ValueError("Reranker output does not expose a score field")

    if "score" in pool.columns:
        pool = pool.drop(columns=["score"])

    pool = pool.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    pool["evidence_rank"] = np.arange(1, len(pool) + 1)
    pool["retrieval_query"] = retrieval_query
    pool["topic"] = topic
    pool = pool.rename(columns={"meeting_date": "date"})

    filtered = pool[pool["rerank_score"] >= RERANK_THRESHOLD].copy().reset_index(drop=True)
    filtered["evidence_rank"] = np.arange(1, len(filtered) + 1)
    return pool, filtered
