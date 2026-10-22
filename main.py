from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from openai import AsyncOpenAI
import pandas as pd
import requests
import torch
from huggingface_hub.utils import disable_progress_bars
from transformers.utils import logging as transformers_logging

from scripts.chroma_store import DEFAULT_CHROMA_PATH
from const import (
    EMBEDDING_MODEL,
    N_TOP_BILLS,
    PREDEFINED_21_TOPICS,
    RERANK_THRESHOLD,
    STANCE_DETECTION_MODEL,
)
from stance_detection import (
    load_stance_classifier,
    save_stance_results,
    score_stance,
    score_stance_custom,
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
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("outputs"),
)
@click.option(
    "--run-name",
    required=True,
    help="Stable label for this retrieval and annotation run.",
)
@click.option(
    "--topic",
    type=click.Choice(PREDEFINED_21_TOPICS),
    required=True,
    help="Broad policy target topic used for grouping runs.",
)
@click.option(
    "--retrieval-query",
    required=True,
    help="Broad query used to retrieve bills and evidence.",
)
@click.option("--stance-claim", default=None, help="Policy claim used for stance detection.")
@click.option(
    "--paragraph-dense-pool",
    type=click.IntRange(min=1),
    default=100,
    help="Passages sent to the cross-encoder.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=128,
    help="Batch size for Qwen reranking.",
)
@click.option(
    "--chroma-db-path",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_CHROMA_PATH,
    help="Persistent Chroma database directory.",
)
@click.option("--bill-collection", default="DemocraSci_bills", help="Chroma bill collection name.")
@click.option(
    "--paragraph-collection",
    default="DemocraSci_paragraphs",
    help="Chroma paragraph collection name.",
)
@click.option(
    "--service-url",
    default="http://127.0.0.1:8000",
    show_default=True,
    help="Local model service URL.",
)
@click.option(
    "--contextual-reranking/--no-contextual-reranking",
    default=False,
    help="Include the language-matched bill title in Qwen reranking documents.",
)
@click.option("--stance-api-base-url", default=None,
              help="Base URL for the custom stance detection API. "
                   "If provided, a custom stance model will be used instead of the default model.")
@click.option("--stance-api-key", default=None,
              help="API key for the custom stance detection API. "
                   "If provided, a custom stance model will be used instead of the default model.")
def cli(
    output_dir: Path,
    run_name: str,
    topic: str,
    retrieval_query: str,
    stance_claim: str,
    paragraph_dense_pool: int,
    batch_size: int,
    stance_api_base_url: str | None,
    stance_api_key: str | None,
    chroma_db_path: Path,
    bill_collection: str,
    paragraph_collection: str,
    service_url: str,
    contextual_reranking: bool,
) -> None:
    configure_logging()
    stance_claim = stance_claim or retrieval_query
    run_dir = make_output_dir(output_dir, run_name)
    logger.info("Writing run artifacts to %s", run_dir)

    start_time = time.time()
    payload = {
        "topic": topic,
        "retrieval_query": retrieval_query,
        "stance_claim": stance_claim,
        "paragraph_dense_pool": paragraph_dense_pool,
        "batch_size": batch_size,
        "contextual_reranking": contextual_reranking,
    }
    try:
        response = requests.post(
            f"{service_url.rstrip('/')}/search",
            json=payload,
            timeout=3600,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        raise click.ClickException(
            f"Could not contact model service at {service_url}. "
        ) from exc

    candidate_bills = pd.DataFrame(result["candidate_bills"])
    save_records(candidate_bills, run_dir / "candidate_bills.json")

    passages_before_threshold = pd.DataFrame(result["candidate_passages_before_threshold"])
    passages = pd.DataFrame(result["candidate_passages"])
    save_records(passages_before_threshold, run_dir / "candidate_passages_before_threshold.json")
    save_records(passages, run_dir / "candidate_passages.json")

    if stance_api_base_url:
        async def score_with_custom_api() -> pd.DataFrame:
            async with AsyncOpenAI(api_key=stance_api_key, base_url=stance_api_base_url) as stance_client:
                return await score_stance_custom(
                    passages,
                    stance_claim=stance_claim,
                    client=stance_client,
                    model_name=STANCE_DETECTION_MODEL,
                )

        stance_results = asyncio.run(score_with_custom_api())
    else:
        if passages.empty:
           stance_results = passages.copy()
        else:
           tokenizer, stance_model_object, stance_device = load_stance_classifier()
           stance_results = asyncio.run(score_stance(
               passages,
               stance_claim=stance_claim,
               tokenizer=tokenizer,
               model=stance_model_object,
               device=stance_device,
               batch_size=batch_size,
           ))
    save_stance_results(stance_results, run_dir / "stance_predictions.json")

    config = {
        "chroma_db_path": str(chroma_db_path),
        "bill_collection": bill_collection,
        "paragraph_collection": paragraph_collection,
        "contextual_reranking": contextual_reranking,
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
            "eligible_paragraphs": result["eligible_paragraphs"],
            "available_bills": result["available_bills"],
            "candidate_bills": len(candidate_bills),
            "candidate_passages_before_threshold": len(passages_before_threshold),
            "candidate_passages": len(passages),
            "stance_predictions": len(stance_results),
        },
    )
    logger.info(
        "Saved JSON records and manifest. Run completed in %.2f seconds.",
        time.time() - start_time,
    )


if __name__ == "__main__":
    cli()
