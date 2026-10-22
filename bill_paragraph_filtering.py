"""Retrieve bills and evidence paragraphs from Chroma."""

from __future__ import annotations

import pandas as pd
import torch
from chromadb.api.models.Collection import Collection
from sentence_transformers import CrossEncoder

from const import RERANKER_MODEL, RERANK_THRESHOLD


def load_reranker_model(
    device: torch.device | str | None = None,
    act_fn: str = "sigmoid",
    instruction: str = "Classify whether the document matches the query policy claim",
) -> CrossEncoder:
    """Load the Qwen cross-encoder reranker."""
    return CrossEncoder(
        RERANKER_MODEL,
        device=device,
        activation_fn=torch.nn.Sigmoid() if act_fn == "sigmoid" else torch.nn.Identity(),
        prompts={"classification": instruction},
        default_prompt_name="classification",
    )


def score_bills(
    retrieval_query: str,
    bill_collection: Collection,
    top_k_bills: int,
    affair_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Return bills ranked by Chroma cosine similarity."""
    if affair_ids is not None and not affair_ids:
        return pd.DataFrame()
    query_args = {
        "query_texts": [retrieval_query],
        "n_results": top_k_bills,
        "include": ["documents", "metadatas", "distances"],
    }
    if affair_ids is not None:
        query_args["where"] = {"Affair_ID": {"$in": affair_ids}}
    result = bill_collection.query(
        **query_args,
    )
    ids = result.get("ids", [[]])[0]
    if not ids:
        return pd.DataFrame()
    rows = []
    for rank, (bill_id, document, metadata, distance) in enumerate(
        zip(
            ids,
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ),
        start=1,
    ):
        row = dict(metadata or {})
        row["Affair_ID"] = str(row.get("Affair_ID", bill_id))
        row["bill_search_text"] = document
        row["bill_embedding_score"] = (1.0 - float(distance)) * 100
        row["bill_embedding_rank"] = rank
        row["bill_rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows)


def select_target_speeches(df: pd.DataFrame, candidate_bills: pd.DataFrame) -> pd.DataFrame:
    """Retained for callers that need to inspect the local corpus."""
    if df.empty or candidate_bills.empty:
        return df.iloc[0:0].copy()
    affair_ids = set(candidate_bills["Affair_ID"].astype(str))
    return df[df["Affair_ID"].astype(str).isin(affair_ids)].copy().reset_index(drop=True)


def build_rerank_documents(passages: pd.DataFrame, contextual_reranking: bool) -> list[str]:
    """Build the documents sent to Qwen while preserving original output text."""
    if not contextual_reranking:
        return passages["text"].astype(str).tolist()

    documents = []
    for row in passages.itertuples(index=False):
        values = row._asdict()
        language = str(values.get("language", "")).upper()
        title_column = "bill_title_de" if language == "DE" else "bill_title_fr"
        title = str(values.get(title_column, "") or "").strip()
        paragraph = str(values.get("text", ""))
        documents.append(f"Bill title: {title}\n\nParagraph: {paragraph}" if title else paragraph)
    return documents


def score_paragraphs(
    topic: str,
    retrieval_query: str,
    stance_claim: str,
    paragraph_collection: Collection,
    candidate_bills: pd.DataFrame,
    reranker: CrossEncoder,
    dense_pool_size: int,
    batch_size: int,
    contextual_reranking: bool = False,
    speaker_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve a Chroma paragraph pool and rerank it with Qwen."""
    if candidate_bills.empty:
        empty = pd.DataFrame()
        return empty, empty
    affair_ids = candidate_bills["Affair_ID"].astype(str).tolist()
    where: dict = {"Affair_ID": {"$in": affair_ids}}
    selected_speakers = [name.strip() for name in (speaker_names or []) if name.strip()]
    if selected_speakers:
        where = {
            "$and": [
                {"Affair_ID": {"$in": affair_ids}},
                {"speaker_full_name": {"$in": selected_speakers}},
            ]
        }
    result = paragraph_collection.query(
        query_texts=[retrieval_query],
        n_results=dense_pool_size,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0]
    if not ids:
        empty = pd.DataFrame()
        return empty, empty
    passages = []
    for index, (paragraph_id, document, metadata, distance) in enumerate(
        zip(
            ids,
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
    ):
        row = dict(metadata or {})
        row.update(
            {
                "chroma_id": paragraph_id,
                "text": document,
                "corpus_id": index,
                "dense_score": (1.0 - float(distance)) * 100,
            }
        )
        passages.append(row)
    passage_df = pd.DataFrame(passages)
    rerank_documents = build_rerank_documents(passage_df, contextual_reranking)
    ranked = reranker.rank(
        query=stance_claim,
        documents=rerank_documents,
        top_k=len(passage_df),
        return_documents=False,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    pool = pd.DataFrame(ranked).merge(passage_df, on="corpus_id", how="left")
    if "score" in pool.columns:
        pool = pool.rename(columns={"score": "rerank_score"})
    elif "rerank_score" not in pool.columns:
        raise ValueError("Reranker output does not expose a score field")
    pool = pool.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    pool["retrieval_query"] = retrieval_query
    pool["stance_claim"] = stance_claim
    pool["topic"] = topic
    filtered = pool[pool["rerank_score"] >= RERANK_THRESHOLD].copy().reset_index(drop=True)
    return pool, filtered
