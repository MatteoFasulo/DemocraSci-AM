from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file, save_file
from sentence_transformers import CrossEncoder, SentenceTransformer

from const import CROSS_ENCODER_MODEL, EMBEDDING_MODEL, N_TOP_BILLS, RERANK_THRESHOLD

DEFAULT_BILL_EMBEDDINGS_CACHE = Path("data/bill_embeddings.safetensors")
DEFAULT_BILL_EMBEDDINGS_META = Path("data/bill_embeddings.meta.json")


@dataclass(frozen=True)
class BillSelectionConfig:
    top_k_bills: int = N_TOP_BILLS
    rerank_threshold: float = RERANK_THRESHOLD
    batch_size: int = 256


def load_embedding_model(device: torch.device | str | None = None) -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def load_cross_encoder_model(device: torch.device | str | None = None) -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL, device=device)


def encode_texts(
    model: SentenceTransformer,
    texts: Iterable[str],
    prefix: str,
    batch_size: int = 256,
) -> np.ndarray:
    prefixed = [f"{prefix} {text}".strip() for text in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def _bill_text_fingerprint(bill_metadata_df: pd.DataFrame) -> str:
    payload = bill_metadata_df.loc[:, [c for c in ["Affair_ID", "bill_search_text"] if c in bill_metadata_df.columns]].copy()
    if "Affair_ID" not in payload.columns:
        payload["Affair_ID"] = bill_metadata_df.index.astype(str)
    if "bill_search_text" not in payload.columns:
        payload["bill_search_text"] = ""

    payload["Affair_ID"] = payload["Affair_ID"].astype(str)
    payload["bill_search_text"] = payload["bill_search_text"].fillna("").astype(str)
    records = payload.sort_values("Affair_ID").to_dict(orient="records")
    normalized = json.dumps(records, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()


def _read_cache_meta(path: Path) -> dict[str, str | int] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_cache_meta(path: Path, meta: dict[str, str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def load_or_build_bill_embeddings(
    bill_metadata_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    cache_path: Path = DEFAULT_BILL_EMBEDDINGS_CACHE,
    meta_path: Path = DEFAULT_BILL_EMBEDDINGS_META,
) -> np.ndarray:
    bill_metadata_df = bill_metadata_df.copy().reset_index(drop=True)
    current_meta = {
        "model_name": EMBEDDING_MODEL,
        "prefix": "passage:",
        "fingerprint": _bill_text_fingerprint(bill_metadata_df),
        "n_rows": int(len(bill_metadata_df)),
        "embedding_dim": int(embedding_model.get_embedding_dimension()),
    }

    cached_meta = _read_cache_meta(meta_path)
    if cache_path.exists() and cached_meta is not None and all(cached_meta.get(k) == v for k, v in current_meta.items()):
        cached = load_file(str(cache_path))
        embeddings = cached.get("bill_embeddings")
        if embeddings is not None and embeddings.shape[0] == len(bill_metadata_df):
            return embeddings

    embeddings = encode_texts(
        embedding_model,
        bill_metadata_df["bill_search_text"].fillna("").tolist(),
        prefix="passage:",
    ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"bill_embeddings": embeddings}, str(cache_path))
    _write_cache_meta(meta_path, current_meta)
    return embeddings


def score_bills(
    question: str,
    bill_metadata_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder | None = None,
    top_k_bills: int = N_TOP_BILLS,
    cache_path: Path = DEFAULT_BILL_EMBEDDINGS_CACHE,
    meta_path: Path = DEFAULT_BILL_EMBEDDINGS_META,
) -> pd.DataFrame:
    if bill_metadata_df.empty:
        return bill_metadata_df.copy()

    df = bill_metadata_df.copy().reset_index(drop=True)
    query_embedding = encode_texts(embedding_model, [question], prefix="query:")[0]
    bill_embeddings = load_or_build_bill_embeddings(
        bill_metadata_df=df,
        embedding_model=embedding_model,
        cache_path=cache_path,
        meta_path=meta_path,
    )

    df["bill_relevance_score"] = (query_embedding @ bill_embeddings.T).reshape(-1)
    df = df.sort_values("bill_relevance_score", ascending=False).reset_index(drop=True)

    if reranker is not None and not df.empty:
        rerank_pairs = [[question, text] for text in df["bill_search_text"].fillna("")]
        df["rerank_score"] = reranker.predict(
            rerank_pairs,
            batch_size=256,
            show_progress_bar=True,
        )
        df = df.sort_values(["rerank_score", "bill_relevance_score"], ascending=False).reset_index(drop=True)
    else:
        df["rerank_score"] = np.nan

    df["bill_rank"] = np.arange(1, len(df) + 1)
    return df.head(top_k_bills).reset_index(drop=True)


def select_target_speeches(df: pd.DataFrame, candidate_bills: pd.DataFrame) -> pd.DataFrame:
    if df.empty or candidate_bills.empty:
        return df.iloc[0:0].copy()

    affair_ids = set(candidate_bills["Affair_ID"].astype(str))
    target = df[df["Affair_ID"].astype(str).isin(affair_ids)].copy()
    return target.reset_index(drop=True)


def score_paragraphs(
    topic: str,
    question: str,
    speeches_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    threshold: float = RERANK_THRESHOLD,
) -> pd.DataFrame:
    if speeches_df.empty:
        return speeches_df.copy()

    passages = speeches_df.copy().reset_index(drop=True)
    passage_embeddings = encode_texts(
        embedding_model,
        passages["TEXT_SPAN"].fillna("").tolist(),
        prefix="passage:",
    )
    question_embedding = encode_texts(embedding_model, [question], prefix="query:")[0]
    passages["embedding_score"] = (question_embedding @ passage_embeddings.T).reshape(-1)

    candidate_passages = passages.sort_values("embedding_score", ascending=False).reset_index(drop=True)
    rerank_pairs = [[question, text] for text in candidate_passages["TEXT_SPAN"].fillna("")]
    candidate_passages["rerank_score"] = reranker.predict(
        rerank_pairs,
        batch_size=256,
        show_progress_bar=True,
    )
    candidate_passages = candidate_passages.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    candidate_passages["is_relevant"] = candidate_passages["rerank_score"] > threshold

    # Keep format consistent
    candidate_passages['question'] = question
    candidate_passages['topic'] = topic
    candidate_passages.rename(columns={
        "language_of_text": "language",
        "meeting_date": "date",
    }, inplace=True)
    return candidate_passages
