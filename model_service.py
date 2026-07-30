"""Local synchronous service for the persistent retrieval models."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
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


app = FastAPI(title="DemocraSci model service")


@app.on_event("startup")
def load_models() -> None:
    app.state.models = ServiceState()


@app.get("/health")
def health() -> dict[str, Any]:
    state = getattr(app.state, "models", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Models are not initialized")
    return {
        "status": "ok",
        "bills": state.bill_store.count(),
        "paragraphs": state.paragraph_store.count(),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    state: ServiceState = app.state.models
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
        )
    return {
        "candidate_bills": candidate_bills.to_dict(orient="records"),
        "candidate_passages_before_threshold": before_threshold.to_dict(orient="records"),
        "candidate_passages": passages.to_dict(orient="records"),
        "available_bills": state.bill_store.count(),
        "eligible_paragraphs": state.paragraph_store.count(),
    }
