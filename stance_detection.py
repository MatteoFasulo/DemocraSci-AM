"""Run the fine-tuned XLM-RoBERTa stance classifier on evidence passages.

The model configuration is the source of truth for label IDs. Binary-model
uncertainty is represented by the stored support/oppose margin rather than
silently relabeled as a trained neutral class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from const import STANCE_DETECTION_MODEL

def load_stance_classifier(model_name: str = STANCE_DETECTION_MODEL) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """Load the tokenizer and classifier, selecting CUDA when available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def label_ids(model: AutoModelForSequenceClassification) -> dict[str, int]:
    """Return ``model.config.label2id`` with normalized label names."""
    return {
        str(label).strip().lower(): int(idx)
        for label, idx in model.config.label2id.items()
    }


def score_stance(
    df: pd.DataFrame,
    stance_claim: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Score passages against a stance claim in batches.

    The claim is the first sequence and the passage is the second sequence;
    only the passage is truncated. The result preserves input columns and adds
    probabilities, margin, predicted stance, confidence, and hypothesis.
    """
    if df.empty:
        return df.copy()

    scored = df.copy().reset_index(drop=True)
    # Accept serialized candidate files that call the corpus text "passage".
    text_column = "text" if "text" in scored.columns else "passage" if "passage" in scored.columns else None
    if text_column is None:
        raise ValueError("Stance scoring requires a 'text' or 'passage' column")
    passages = scored[text_column].fillna("").astype(str).tolist()
    if "text" not in scored.columns:
        scored["text"] = passages
    hypotheses = [stance_claim for _ in passages]

    probs = []
    with torch.no_grad():
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            batch_hypotheses = hypotheses[start : start + batch_size]
            encoded = tokenizer(
                batch_hypotheses,
                batch_passages,
                truncation="only_second", # Truncate only the passage (second sequence) to preserve the stance claim (first sequence).
                max_length=512,
                return_tensors="pt",
                padding=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())

    probs_arr = np.vstack(probs)

    configured_labels = label_ids(model)
    support_id = configured_labels.get("favor")
    oppose_id = configured_labels.get("against")
    neutral_id = configured_labels.get("neutral")
    if support_id is None or oppose_id is None:
        raise ValueError(
            "Model config must define 'favor' and 'against' labels; "
            f"found {model.config.label2id}"
        )

    scored["stance_prob_oppose"] = probs_arr[:, oppose_id]
    scored["stance_prob_support"] = probs_arr[:, support_id]
    scored["stance_prob_neutral"] = (
        probs_arr[:, neutral_id] if neutral_id is not None else 0.0
    )
    scored["stance_margin"] = scored["stance_prob_support"] - scored["stance_prob_oppose"]

    predicted = probs_arr.argmax(axis=1)
    stance = []
    confidence = []
    for idx, support, oppose, neutral in zip(
        predicted,
        scored["stance_prob_support"],
        scored["stance_prob_oppose"],
        scored["stance_prob_neutral"],
    ):
        if idx == support_id:
            stance.append("SUPPORT")
            confidence.append(float(support))
        elif idx == oppose_id:
            stance.append("OPPOSE")
            confidence.append(float(oppose))
        else:
            stance.append("NEUTRAL")
            confidence.append(float(neutral))

    scored["stance"] = stance
    scored["confidence"] = confidence
    scored["hypothesis"] = hypotheses
    return scored


def save_stance_results(df: pd.DataFrame, path: str | Path) -> Path:
    """Serialize stance predictions using the stable dataset output schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    # Different input datasets use different speaker-name fields. Coalesce
    # them before renaming so multiple source columns never become duplicate
    # ``MP`` columns (which pandas cannot serialize as records).
    name_columns = [
        column
        for column in ("MP", "name", "speaker_name", "speaker_full_name")
        if column in df.columns
    ]
    if name_columns:
        df["MP"] = df[name_columns].replace("", np.nan).bfill(axis=1).iloc[:, 0]

    df = df.rename(
        columns={
            "text": "passage",
            "stance": "label",
        }
    )

    expected_columns = [
        "speech_id",
        "paragraph_id",
        "Affair_ID",
        "bill_number",
        "date",
        "language",
        "retrieval_query",
        "passage",
        "label",
        "stance_prob_support",
        "stance_prob_neutral",
        "stance_prob_oppose",
        "MP",
        "party",
        "topic",
        "rerank_score",
    ]

    for column in expected_columns:
        if column not in df.columns:
            df[column] = np.nan

    df = df[expected_columns]
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return output_path
