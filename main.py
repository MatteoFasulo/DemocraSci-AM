from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from bill_paragraph_filtering import (
    load_cross_encoder_model,
    load_embedding_model,
    score_bills,
    score_paragraphs,
    select_target_speeches,
)
from const import N_TOP_BILLS, RERANK_THRESHOLD, STANCE_DETECTION_MODEL
from data_preprocessing import df_bill_metadata, df_preprocessing
from stance_detection import load_stance_classifier, save_stance_results, score_stance

logger = logging.getLogger("mappingdemocrasci")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root.addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bill retrieval, paragraph filtering, and stance detection.")
    parser.add_argument("--input-data", type=Path, default=Path("data/speeches_with_text.parquet"), help="Input speech-level Parquet file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artifacts.")
    parser.add_argument("--topic", type=str, required=True, help="Policy topic / target query.")
    parser.add_argument("--question", type=str, required=True, help="Natural-language question for retrieval.")
    parser.add_argument("--top-bills", type=int, default=N_TOP_BILLS, help="Number of bills to keep after ranking.")
    parser.add_argument("--rerank-threshold", type=float, default=RERANK_THRESHOLD, help="Cross-encoder threshold for paragraph relevance.")
    parser.add_argument("--stance-model", type=str, default=STANCE_DETECTION_MODEL, help="Sequence-classification stance model.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for stance scoring.")
    return parser


def main(args: argparse.Namespace) -> None:
    configure_logging()

    slugified_topic = args.topic.lower().replace(" ", "_")
    slugified_question = args.question.lower().replace(" ", "_")
    output_dir: Path = args.output_dir / f"{slugified_topic}_{slugified_question}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_data)
    df_augmented = df_preprocessing(df)
    bill_metadata_df = df_bill_metadata(df_augmented)
    logger.info("Loaded %d speeches and %d unique bills.", len(df_augmented), len(bill_metadata_df))

    embedding_model = load_embedding_model(device=DEVICE)
    logger.info("Loaded embedding model '%s' on device %s.", embedding_model.__class__.__name__, DEVICE)
    reranker = load_cross_encoder_model(device=DEVICE)
    logger.info("Loaded cross-encoder model '%s' on device %s.", reranker.__class__.__name__, DEVICE)

    candidate_bills = score_bills(
        question=args.question,
        bill_metadata_df=bill_metadata_df,
        embedding_model=embedding_model,
        reranker=reranker,
        top_k_bills=args.top_bills,
    )
    logger.info("Selected %d candidate bills for topic '%s'.", len(candidate_bills), args.topic)

    target_speeches = select_target_speeches(df_augmented, candidate_bills)
    logger.info("Selected %d target speeches.", len(target_speeches))
    paragraph_scores = score_paragraphs(
        topic=args.topic,
        question=args.question,
        speeches_df=target_speeches,
        embedding_model=embedding_model,
        reranker=reranker,
        threshold=args.rerank_threshold,
    )
    logger.info("Scored %d paragraphs for relevance to topic '%s'.", len(paragraph_scores), args.topic)

    tokenizer, stance_model, stance_device = load_stance_classifier(args.stance_model)
    logger.info("Loaded stance detection model '%s' on device %s.", stance_model.__class__.__name__, stance_device)

    stance_input = paragraph_scores[paragraph_scores["is_relevant"]].copy()
    stance_results = score_stance(
        stance_input,
        hypothesis=args.question,
        tokenizer=tokenizer,
        model=stance_model,
        device=stance_device,
        batch_size=args.batch_size,
    )
    logger.info("Detected stances for %d paragraphs.", len(stance_results))

    selected_bills_path = output_dir / "candidate_bills.json"
    paragraphs_path = output_dir / "candidate_paragraphs.json"
    stance_path = output_dir / f"{slugified_topic}_{slugified_question}.json"
    json_export_args = {"orient": "records", "force_ascii": False, "indent": 2}

    candidate_bills.to_json(selected_bills_path, **json_export_args)
    paragraph_scores.to_json(paragraphs_path, **json_export_args)
    save_stance_results(stance_results, stance_path)
    logger.info("Saved candidate bills to %s", selected_bills_path)
    logger.info("Saved scored paragraphs to %s", paragraphs_path)
    logger.info("Saved stance results to %s", stance_path)

if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
