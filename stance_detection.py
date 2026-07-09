from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from const import STANCE_DETECTION_MODEL


def load_stance_classifier(model_name: str = STANCE_DETECTION_MODEL) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _label_map(model: AutoModelForSequenceClassification) -> dict[int, str]:
    labels = {int(idx): str(label).strip().lower() for idx, label in model.config.id2label.items()}
    if all(label.startswith("label_") for label in labels.values()):
        if len(labels) == 3:
            return {0: "contradiction", 1: "neutral", 2: "entailment"}
        if len(labels) == 2:
            return {0: "against", 1: "favor"}
    return labels


def _pick_label(label_map: dict[int, str], candidates: tuple[str, ...]) -> int | None:
    for idx, label in label_map.items():
        if label in candidates:
            return idx
    return None


def score_stance(
    df: pd.DataFrame,
    hypothesis: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    scored = df.copy().reset_index(drop=True)
    passages = scored["TEXT_SPAN"].fillna("").astype(str).tolist()
    hypotheses = [hypothesis for _ in passages]

    probs = []
    with torch.no_grad():
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            batch_hypotheses = hypotheses[start : start + batch_size]
            encoded = tokenizer(
                batch_hypotheses,
                batch_passages,
                truncation="only_second",
                max_length=512,
                return_tensors="pt",
                padding=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())

    probs_arr = np.vstack(probs)
    labels = _label_map(model)
    support_id = _pick_label(labels, ("entailment", "favor", "favour", "support", "supports"))
    oppose_id = _pick_label(labels, ("contradiction", "against", "oppose", "opposes"))
    neutral_id = _pick_label(labels, ("neutral",))

    if support_id is None or oppose_id is None:
        raise ValueError(f"Could not map stance labels from model config: {labels}")

    scored["stance_prob_support"] = probs_arr[:, support_id]
    scored["stance_prob_oppose"] = probs_arr[:, oppose_id]
    scored["stance_prob_neutral"] = probs_arr[:, neutral_id] if neutral_id is not None else 0.0
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
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    return output_path
