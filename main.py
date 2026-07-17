from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd
import torch
from huggingface_hub.utils import disable_progress_bars
from transformers.utils import logging as transformers_logging

from bill_paragraph_filtering import (
    load_reranker_model,
    load_embedding_model,
    score_bills,
    score_paragraphs,
    select_target_speeches,
)
from const import EMBEDDING_MODEL, N_TOP_BILLS, PREDEFINED_21_TOPICS, RERANK_THRESHOLD, STANCE_DETECTION_MODEL
from data_preprocessing import df_bill_metadata, df_preprocessing
from stance_detection import (
    load_stance_classifier,
    save_stance_results,
    score_stance,
)


logger = logging.getLogger("democrasci")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logging() -> None:
    # Keep pipeline milestones visible without flooding the terminal with
    # Hugging Face download and inference progress output.
    disable_progress_bars()
    transformers_logging.set_verbosity_warning()
    for logger_name in ("huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "run"


def make_output_dir(base_dir: Path, run_name: str) -> Path:
    output_dir = base_dir / slugify(run_name)
    suffix = 2
    while output_dir.exists():
        output_dir = base_dir / f"{slugify(run_name)}-{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def save_records(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", force_ascii=False, indent=2)


def save_manifest(path: Path, config: dict[str, Any], counts: dict[str, int]) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "config": config,
        "counts": counts,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")




@click.command(context_settings={"show_default": True})
@click.option(
    "--input-data",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=Path("data/speeches_with_text.parquet"),
)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), default=Path("outputs"))
@click.option("--run-name", required=True, help="Stable label for this retrieval and annotation run.")
@click.option("--topic", type=click.Choice(PREDEFINED_21_TOPICS), required=True, help="Broad policy target topic used for grouping runs.")
@click.option("--retrieval-query", required=True, help="Broad query used to retrieve bills and evidence.")
@click.option("--stance-claim", default=None, help="Policy claim used for stance detection.")
@click.option("--paragraph-dense-pool", type=click.IntRange(min=1), default=100, help="Passages sent to the cross-encoder.")
@click.option("--batch-size", type=click.IntRange(min=1), default=64, help="Batch size for embedding and cross-encoder scoring.")
def cli(
    input_data: Path,
    output_dir: Path,
    run_name: str,
    topic: str,
    retrieval_query: str,
    stance_claim: str,
    paragraph_dense_pool: int,
    batch_size: int,
) -> None:
    configure_logging()
    stance_claim = stance_claim or retrieval_query
    run_dir = make_output_dir(output_dir, run_name)
    logger.info("Writing run artifacts to %s", run_dir)

    raw = pd.read_parquet(input_data)
    logger.info("Loaded %d rows from %s", len(raw), input_data)

    speeches = df_preprocessing(raw)
    logger.info("Preprocessed %d rows into %d paragraphs", len(raw), len(speeches))

    bills = df_bill_metadata(speeches)
    logger.info("Prepared %d paragraphs and %d bills", len(speeches), len(bills))

    embedding_model = load_embedding_model(DEVICE)
    reranker = load_reranker_model(DEVICE)
    candidate_bills = score_bills(
        retrieval_query=retrieval_query,
        bill_metadata_df=bills,
        embedding_model=embedding_model,
        top_k_bills=N_TOP_BILLS,
    )
    save_records(candidate_bills, run_dir / "candidate_bills.json")

    target_speeches = select_target_speeches(speeches, candidate_bills)
    passages_before_threshold, passages = score_paragraphs(
        topic=topic,
        retrieval_query=retrieval_query,
        speeches_df=target_speeches,
        embedding_model=embedding_model,
        reranker=reranker,
        dense_pool_size=paragraph_dense_pool,
        batch_size=batch_size,
    )
    save_records(passages_before_threshold, run_dir / "candidate_passages_before_threshold.json")
    save_records(passages, run_dir / "candidate_passages.json")

    tokenizer, stance_model_object, stance_device = load_stance_classifier()
    stance_results = score_stance(
        passages,
        stance_claim=stance_claim,
        tokenizer=tokenizer,
        model=stance_model_object,
        device=stance_device,
        batch_size=batch_size,
    )
    save_stance_results(stance_results, run_dir / "stance_predictions.json")

    config = {
        "input_data": str(input_data),
        "run_name": run_name,
        "topic": topic,
        "retrieval_query": retrieval_query,
        "stance_claim": stance_claim,
        "top_bills": N_TOP_BILLS,
        "paragraph_dense_pool": paragraph_dense_pool,
        "rerank_threshold": RERANK_THRESHOLD,
        "embedding_model": EMBEDDING_MODEL,
        "stance_model": STANCE_DETECTION_MODEL,
    }
    save_manifest(
        run_dir / "manifest.json",
        config,
        {
            "raw_rows": len(raw),
            "eligible_paragraphs": len(speeches),
            "available_bills": len(bills),
            "candidate_bills": len(candidate_bills),
            "candidate_passages_before_threshold": len(passages_before_threshold),
            "candidate_passages": len(passages),
            "stance_predictions": len(stance_results),
        },
    )
    logger.info("Saved JSON records and manifest")


if __name__ == "__main__":
    cli()
