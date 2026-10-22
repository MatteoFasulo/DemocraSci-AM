"""Local synchronous service for the persistent retrieval models."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from bill_paragraph_filtering import load_reranker_model, score_bills, score_paragraphs
from const import N_TOP_BILLS
from scripts.chroma_store import (
    BILLS_COLLECTION,
    DEFAULT_CHROMA_PATH,
    PARAGRAPHS_COLLECTION,
    open_chroma_collections,
)


class SearchRequest(BaseModel):
    topic: str
    retrieval_query: str
    stance_claim: str
    paragraph_dense_pool: int = Field(default=100, ge=1)
    batch_size: int = Field(default=128, ge=1)
    contextual_reranking: bool = False
    speaker_names: list[str] = Field(default_factory=list)


class BillSearchRequest(BaseModel):
    retrieval_query: str
    top_k_bills: int = Field(default=N_TOP_BILLS, ge=1)
    speaker_name: str | None = None
    min_speaker_paragraphs: int = Field(default=1, ge=1)


class ParagraphSearchRequest(BaseModel):
    topic: str = ""
    retrieval_query: str
    stance_claim: str
    affair_ids: list[str]
    paragraph_dense_pool: int = Field(default=100, ge=1)
    batch_size: int = Field(default=128, ge=1)
    contextual_reranking: bool = False
    speaker_name: str | None = None


class ServiceState:
    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bill_store, self.paragraph_store = open_chroma_collections(
            db_path=Path(DEFAULT_CHROMA_PATH),
            bills_name=BILLS_COLLECTION,
            paragraphs_name=PARAGRAPHS_COLLECTION,
            device=device,
        )
        self.reranker = load_reranker_model(device, act_fn="sigmoid")
        self.lock = threading.Lock()
        self.facets_lock = threading.Lock()
        self._speaker_names: list[str] | None = None
        self._speaker_affair_counts: dict[str, dict[str, int]] | None = None
        self._speaker_ids: dict[str, str] | None = None

    def _load_speaker_facets(self) -> None:
        if self._speaker_affair_counts is not None:
            return
        counts: dict[str, dict[str, int]] = {}
        speaker_ids: dict[str, str] = {}
        page_size = 10_000
        offset = 0
        while True:
            page = self.paragraph_store.get(
                include=["metadatas"], limit=page_size, offset=offset
            )
            metadatas = page.get("metadatas", []) or []
            for metadata in metadatas:
                values = metadata or {}
                name = str(values.get("speaker_full_name", "")).strip()
                affair_id = str(values.get("Affair_ID", "")).strip()
                raw_speaker_id = values.get("speaker_id")
                if name and raw_speaker_id not in (None, "") and name not in speaker_ids:
                    if isinstance(raw_speaker_id, float) and raw_speaker_id.is_integer():
                        speaker_ids[name] = str(int(raw_speaker_id))
                    else:
                        speaker_ids[name] = str(raw_speaker_id).strip()
                if name and affair_id:
                    speaker_counts = counts.setdefault(name, {})
                    speaker_counts[affair_id] = speaker_counts.get(affair_id, 0) + 1
            offset += len(metadatas)
            if len(metadatas) < page_size:
                break
        self._speaker_affair_counts = counts
        self._speaker_names = sorted(counts, key=str.casefold)
        self._speaker_ids = speaker_ids

    def speaker_names(self) -> list[str]:
        """Return cached distinct speaker names from paragraph metadata."""
        with self.facets_lock:
            self._load_speaker_facets()
            return self._speaker_names

    def speaker_ids(self) -> dict[str, str]:
        """Return the parliamentary member ID associated with each speaker."""
        with self.facets_lock:
            self._load_speaker_facets()
            return self._speaker_ids

    def speaker_bill_counts(self, speaker_names: list[str]) -> dict[str, int]:
        """Return each bill's maximum paragraph count among selected speakers."""
        with self.facets_lock:
            self._load_speaker_facets()
            affair_counts: dict[str, int] = {}
            for speaker_name in speaker_names:
                for affair_id, count in self._speaker_affair_counts.get(speaker_name, {}).items():
                    affair_counts[affair_id] = max(affair_counts.get(affair_id, 0), count)
            return affair_counts


app = FastAPI(title="DemocraSci model service")


@app.on_event("startup")
def load_models() -> None:
    app.state.models = ServiceState()


def get_service_state() -> ServiceState:
    """Return the startup-initialized service state for request handlers."""
    state = getattr(app.state, "models", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Models are not initialized")
    return state


@app.get("/health")
def health(state: ServiceState = Depends(get_service_state)) -> dict[str, Any]:
    return {
        "status": "ok",
        "bills": state.bill_store.count(),
        "paragraphs": state.paragraph_store.count(),
    }


@app.get("/facets")
def facets(state: ServiceState = Depends(get_service_state)) -> dict[str, Any]:
    """Return selectable search facets derived from the indexed corpus."""
    return {"speakers": state.speaker_names(), "speaker_ids": state.speaker_ids()}


@app.post("/bills")
def retrieve_bills(
    request: BillSearchRequest,
    state: ServiceState = Depends(get_service_state),
) -> dict[str, Any]:
    """Retrieve relevant bills, optionally requiring selected-speaker participation."""
    speaker_names = [request.speaker_name] if request.speaker_name else []
    speaker_bill_counts = state.speaker_bill_counts(speaker_names) if speaker_names else None
    affair_ids = None
    if speaker_bill_counts is not None:
        affair_ids = [
            affair_id
            for affair_id, count in speaker_bill_counts.items()
            if count >= request.min_speaker_paragraphs
        ]
    with state.lock:
        candidate_bills = score_bills(
            request.retrieval_query,
            state.bill_store,
            request.top_k_bills,
            affair_ids=affair_ids,
        )
    if speaker_bill_counts is not None and not candidate_bills.empty:
        candidate_bills["speaker_paragraph_count"] = candidate_bills["Affair_ID"].map(
            speaker_bill_counts
        )
    return {
        "candidate_bills": candidate_bills.to_dict(orient="records"),
        "available_bills": state.bill_store.count(),
        "speaker_filtered": bool(speaker_names),
    }


@app.post("/paragraphs")
def retrieve_paragraphs(
    request: ParagraphSearchRequest,
    state: ServiceState = Depends(get_service_state),
) -> dict[str, Any]:
    """Retrieve and rerank paragraphs from previously selected candidate bills."""
    candidate_bills = pd.DataFrame({"Affair_ID": request.affair_ids})
    with state.lock:
        before_threshold, passages = score_paragraphs(
            request.topic,
            request.retrieval_query,
            request.stance_claim,
            state.paragraph_store,
            candidate_bills,
            state.reranker,
            request.paragraph_dense_pool,
            request.batch_size,
            request.contextual_reranking,
            [request.speaker_name] if request.speaker_name else [],
        )
    return {
        "candidate_passages_before_threshold": before_threshold.to_dict(orient="records"),
        "candidate_passages": passages.to_dict(orient="records"),
    }


@app.post("/search")
def search(
    request: SearchRequest,
    state: ServiceState = Depends(get_service_state),
) -> dict[str, Any]:
    with state.lock:
        candidate_bills = score_bills(request.retrieval_query, state.bill_store, N_TOP_BILLS)
        before_threshold, passages = score_paragraphs(
            request.topic,
            request.retrieval_query,
            request.stance_claim,
            state.paragraph_store,
            candidate_bills,
            state.reranker,
            request.paragraph_dense_pool,
            request.batch_size,
            request.contextual_reranking,
            request.speaker_names,
        )
    return {
        "candidate_bills": candidate_bills.to_dict(orient="records"),
        "candidate_passages_before_threshold": before_threshold.to_dict(orient="records"),
        "candidate_passages": passages.to_dict(orient="records"),
        "available_bills": state.bill_store.count(),
        "eligible_paragraphs": state.paragraph_store.count(),
    }
