from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
from huggingface_hub.utils import disable_progress_bars
from transformers.utils import logging as transformers_logging

from bill_paragraph_filtering import (
    load_embedding_model as load_embedding_model_uncached,
    load_reranker_model as load_reranker_model_uncached,
    score_bills,
    score_paragraphs,
    select_target_speeches,
)
from const import (
    EMBEDDING_MODEL,
    N_TOP_BILLS,
    PREDEFINED_21_TOPICS,
    RERANK_THRESHOLD,
    STANCE_DETECTION_MODEL,
)
from data_preprocessing import df_bill_metadata, df_preprocessing
from main import make_output_dir
from stance_detection import (
    load_stance_classifier as load_stance_classifier_uncached,
    save_stance_results,
    score_stance,
)

logger = logging.getLogger("democrasci")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="DemocraSci", page_icon=":bar_chart:", layout="wide")


def configure_logging() -> None:
    disable_progress_bars()
    transformers_logging.set_verbosity_warning()
    for logger_name in ("huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", force=True)


@st.cache_data
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    speeches = df_preprocessing(pd.read_parquet(input_path))
    return speeches, df_bill_metadata(speeches)


@st.cache_resource
def load_embedding_model(device: str):
    return load_embedding_model_uncached(torch.device(device))


@st.cache_resource
def load_reranker_model(device: str):
    return load_reranker_model_uncached(torch.device(device))


@st.cache_resource
def load_stance_classifier():
    return load_stance_classifier_uncached()


def save_records(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", force_ascii=False, indent=2)


def save_manifest(path: Path, config: dict[str, Any], counts: dict[str, int]) -> None:
    path.write_text(
        json.dumps(
            {"created_at": datetime.now(timezone.utc).isoformat(), "device": str(DEVICE), "config": config, "counts": counts},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_dashboard_pipeline(
    *, speeches: pd.DataFrame, bills: pd.DataFrame, run_dir: Path, topic: str,
    retrieval_query: str, stance_claim: str, paragraph_dense_pool: int, batch_size: int,
) -> dict[str, int]:
    embedding_model = load_embedding_model(str(DEVICE))
    reranker = load_reranker_model(str(DEVICE))
    candidate_bills = score_bills(retrieval_query, bills, embedding_model, N_TOP_BILLS)
    save_records(candidate_bills, run_dir / "candidate_bills.json")
    target_speeches = select_target_speeches(speeches, candidate_bills)
    before_threshold, passages = score_paragraphs(
        topic, retrieval_query, target_speeches, embedding_model, reranker, paragraph_dense_pool, batch_size
    )
    save_records(before_threshold, run_dir / "candidate_passages_before_threshold.json")
    save_records(passages, run_dir / "candidate_passages.json")
    tokenizer, model, stance_device = load_stance_classifier()
    stance_results = score_stance(passages, stance_claim, tokenizer, model, stance_device, batch_size)
    save_stance_results(stance_results, run_dir / "stance_predictions.json")
    counts = {
        "eligible_paragraphs": len(speeches), "available_bills": len(bills),
        "candidate_bills": len(candidate_bills),
        "candidate_passages_before_threshold": len(before_threshold),
        "candidate_passages": len(passages), "stance_predictions": len(stance_results),
    }
    save_manifest(
        run_dir / "manifest.json",
        {"topic": topic, "retrieval_query": retrieval_query, "stance_claim": stance_claim,
         "top_bills": N_TOP_BILLS, "paragraph_dense_pool": paragraph_dense_pool,
         "batch_size": batch_size, "rerank_threshold": RERANK_THRESHOLD,
         "embedding_model": EMBEDDING_MODEL, "stance_model": STANCE_DETECTION_MODEL},
        counts,
    )
    return counts


def main() -> None:
    configure_logging()
    st.title("DemocraSci stance retrieval")
    st.caption(f"Running on {DEVICE}")
    with st.sidebar:
        input_data = st.text_input("Input parquet", "data/speeches_with_text.parquet")
        output_dir = st.text_input("Output directory", "outputs")
        run_name = st.text_input("Run name", "democrasci-run")
        topic = st.selectbox("Topic", PREDEFINED_21_TOPICS)
        paragraph_dense_pool = st.number_input("Paragraph dense pool", min_value=1, value=100)
        batch_size = st.number_input("Batch size", min_value=1, value=64)
    retrieval_query = st.text_area("Retrieval query", placeholder="Should smoking in public area be prohibited?")
    stance_claim = st.text_area("Stance claim (optional; defaults to retrieval query)")
    if not st.button("Run analysis", type="primary"):
        return
    if not retrieval_query.strip():
        st.error("Enter a retrieval query before running the analysis.")
        return
    if not run_name.strip():
        st.error("Enter a run name before running the analysis.")
        return
    try:
        with st.spinner("Loading and preprocessing data…"):
            speeches, bills = load_data(input_data)
        run_dir = make_output_dir(Path(output_dir), run_name)
        with st.spinner("Retrieving passages and scoring stances…"):
            counts = run_dashboard_pipeline(
                speeches=speeches, bills=bills, run_dir=run_dir, topic=topic,
                retrieval_query=retrieval_query, stance_claim=stance_claim.strip() or retrieval_query,
                paragraph_dense_pool=int(paragraph_dense_pool), batch_size=int(batch_size),
            )
    except Exception as exc:
        logger.exception("Dashboard pipeline failed")
        st.error(f"Analysis failed: {exc}")
        return
    st.success(f"Analysis complete. Results saved to {run_dir}")
    st.json(counts)


if __name__ == "__main__":
    main()
