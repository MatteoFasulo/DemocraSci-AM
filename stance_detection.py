"""Run stance classification on evidence passages.

Two interchangeable backends are available:

- ``load_stance_classifier`` / ``score_stance``: the fine-tuned XLM-RoBERTa
  classifier. The model configuration is the source of truth for label IDs.
  Binary-model uncertainty is represented by the stored support/oppose margin
  rather than silently relabeled as a trained neutral class.
- ``load_stance_custom_client`` / ``score_stance_custom``: a chat LLM served behind
  an OpenAI-compatible endpoint (e.g. vLLM), constrained to the same
  SUPPORT/OPPOSE/NEUTRAL labels via guided decoding. Use this when no local
  classifier is available for the language/domain.

Both scorers produce the same output schema, so callers can swap one for the
other without touching ``save_stance_results`` or downstream code.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from const import STANCE_DETECTION_MODEL
from llm_labelling import get_openai_client, get_probabilities

_STANCE_LLM_LABELS = ("SUPPORT", "OPPOSE", "NEUTRAL")

_STANCE_LLM_SYSTEM_PROMPT = (
    "You are an annotator for political science. Decide whether the passage "
    "supports, opposes, or is neutral toward the claim. Respond with exactly "
    "one label: SUPPORT, OPPOSE, or NEUTRAL."
)


def load_stance_classifier(model_name: str = STANCE_DETECTION_MODEL) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """Load the tokenizer and classifier, selecting CUDA when available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_stance_custom_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Create an OpenAI-compatible client for a vLLM-served stance LLM.

    Falls back to the ``OPENAI_API_KEY`` / ``OPENAI_API_BASE_URL`` environment
    variables used elsewhere in this project (see ``llm_labelling.py``) when
    ``api_key`` / ``base_url`` aren't passed explicitly.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("OPENAI_API_BASE_URL")
    if not api_key or not base_url:
        raise ValueError(
            "load_stance_custom_client requires an API key and base URL, via "
            "arguments or the OPENAI_API_KEY / OPENAI_API_BASE_URL "
            "environment variables."
        )
    return get_openai_client(api_key, base_url)


def label_ids(model: AutoModelForSequenceClassification) -> dict[str, int]:
    """Return ``model.config.label2id`` with normalized label names."""
    return {
        str(label).strip().lower(): int(idx)
        for label, idx in model.config.label2id.items()
    }


def _prepare_scored_frame(df: pd.DataFrame, stance_claim: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Resolve the passage-text column and stage claim/passage data shared by both scoring backends."""
    scored = df.copy().reset_index(drop=True)
    # Accept serialized candidate files that call the corpus text "passage".
    text_column = "text" if "text" in scored.columns else "passage" if "passage" in scored.columns else None
    if text_column is None:
        raise ValueError("Stance scoring requires a 'text' or 'passage' column")
    passages = scored[text_column].fillna("").astype(str).tolist()
    if "text" not in scored.columns:
        scored["text"] = passages
    hypotheses = [stance_claim for _ in passages]
    return scored, passages, hypotheses


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

    scored, passages, hypotheses = _prepare_scored_frame(df, stance_claim)

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


def _stance_llm_choice(
        client: OpenAI,
        model_name: str,
        stance_claim: str,
        passage: str):
    """Request one stance label, constrained to `_STANCE_LLM_LABELS` via vLLM guided_choice."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _STANCE_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"CLAIM: {stance_claim}\nPASSAGE: {passage}\nLABEL:"},
        ],
        temperature=0,
        max_tokens=4,  # headroom in case a label isn't a single token for this model's tokenizer
        logprobs=True,
        top_logprobs=20,
        extra_body={
            "guided_choice": list(_STANCE_LLM_LABELS),
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0]


def score_stance_custom(
    df: pd.DataFrame,
    stance_claim: str,
    client: OpenAI,
    model_name: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Score passages against a stance claim via a chat LLM endpoint.

    Produces the same output schema as `score_stance`. Requests run
    concurrently since this is network-bound (unlike the classifier's batched
    GPU forward pass); each call is constrained to SUPPORT/OPPOSE/NEUTRAL via
    vLLM guided_choice, and per-label probabilities are recovered from that
    token's logprobs (`get_probabilities`, shared with `llm_labelling.py`),
    then renormalized to sum to 1 so the margin is comparable to the
    classifier's softmax-derived one.
    """
    if df.empty:
        return df.copy()

    scored, passages, hypotheses = _prepare_scored_frame(df, stance_claim)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        choices = list(
            pool.map(
                lambda passage: _stance_llm_choice(client, model_name, stance_claim, passage),
                passages,
            )
        )
    stance = [choice.message.content.strip().upper() for choice in choices]

    raw_probs = [get_probabilities(choice, list(_STANCE_LLM_LABELS)) for choice in choices]
    probs_arr = np.array(
        [[probs.get(label, 0.0) for label in _STANCE_LLM_LABELS] for probs in raw_probs]
    )
    row_sums = probs_arr.sum(axis=1, keepdims=True)
    probs_arr = np.divide(probs_arr, row_sums, out=np.zeros_like(probs_arr), where=row_sums > 0)

    label_index = {label: i for i, label in enumerate(_STANCE_LLM_LABELS)}
    for row, (label, total) in enumerate(zip(stance, row_sums.flatten())):
        if total == 0:
            # guided_choice should always surface the chosen label's logprob; this
            # only triggers if the endpoint omits it from top_logprobs entirely.
            probs_arr[row, label_index[label]] = 1.0

    scored["stance_prob_support"] = probs_arr[:, label_index["SUPPORT"]]
    scored["stance_prob_oppose"] = probs_arr[:, label_index["OPPOSE"]]
    scored["stance_prob_neutral"] = probs_arr[:, label_index["NEUTRAL"]]
    scored["stance_margin"] = scored["stance_prob_support"] - scored["stance_prob_oppose"]
    scored["stance"] = stance
    scored["confidence"] = [probs_arr[row, label_index[label]] for row, label in enumerate(stance)]
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
