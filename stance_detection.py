"""Run stance classification on evidence passages.

Two interchangeable backends are available:

- ``load_stance_classifier`` / ``score_stance``: the fine-tuned XLM-RoBERTa
  classifier. The model configuration is the source of truth for label IDs.
  Binary-model uncertainty is represented by the stored support/oppose margin
  rather than silently relabeled as a trained neutral class.
- ``load_stance_custom_client`` / ``score_stance_custom``: a chat LLM served behind
  an OpenAI-compatible endpoint (e.g. vLLM), requested to emit the labels
  ``A`` (IN FAVOR), ``B`` (AGAINST), or ``C`` (NEUTRAL) via guided decoding.
  If guided decoding is ignored, the scorer uses first-token logprobs for
  those three labels. Use this when no local classifier is available for the
  language/domain.

Both scorers produce the same core outputs (stance, confidence, margin,
and hypothesis), so callers can swap one for the other without changing the
pipeline. Probability column names follow the backend's label vocabulary.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import Literal
from openai import AsyncOpenAI
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from const import STANCE_DETECTION_MODEL, STANCE_LLM_LABELS, STANCE_LLM_SYSTEM_PROMPT

STANCE_CLASSES = tuple(STANCE_LLM_LABELS.values())


class StanceResponse(BaseModel):
    """Structured response returned by the stance-classification LLM."""

    answer: Literal["A", "B", "C"]


def load_stance_classifier(
    model_name: str = STANCE_DETECTION_MODEL,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
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


def _prepare_scored_frame(
    df: pd.DataFrame, stance_claim: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Resolve the passage-text column and stage claim/passage data shared by both scoring backends."""
    scored = df.copy().reset_index(drop=True)
    # Accept serialized candidate files that call the corpus text "passage".
    text_column = (
        "text"
        if "text" in scored.columns
        else "passage" if "passage" in scored.columns else None
    )
    if text_column is None:
        raise ValueError("Stance scoring requires a 'text' or 'passage' column")
    passages = scored[text_column].fillna("").astype(str).tolist()
    if "text" not in scored.columns:
        scored["text"] = passages
    hypotheses = [stance_claim for _ in passages]
    return scored, passages, hypotheses


async def score_stance(
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
                truncation="only_second",  # Truncate only the passage (second sequence) to preserve the stance claim (first sequence).
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
    scored["stance_margin"] = (
        scored["stance_prob_support"] - scored["stance_prob_oppose"]
    )

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


async def _stance_llm_choice(
    client: AsyncOpenAI, model_name: str, stance_claim: str, passage: str
):
    """Request a stance label using the validated JSON response schema."""
    response = await client.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": STANCE_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"CLAIM:\n{stance_claim}\n\nPASSAGE:\n{passage}"},
        ],
        temperature=0.0,
        max_tokens=8,
        logprobs=True,
        top_logprobs=4,
        response_format=StanceResponse if not "deepseek"in model_name.lower() else {'type': 'json_object'}, # DeepSeek API
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
            "thinking": {"type": "disabled"}, # DeepSeek API
        },
    )
    return response.choices[0]


def get_labels_distribution(top_logprobs) -> dict[str, float]:
    """Return probabilities over IN FAVOR, AGAINST, and NEUTRAL.

    The returned probabilities are computed by applying a numerically stable
    softmax to the log probabilities of the allowed output tokens A, B, and C.

    Raises:
        ValueError: If one or more allowed label tokens are absent from the
            returned top-logprob candidates.
    """
    valid_logprobs: dict[str, float] = {}

    for item in top_logprobs:
        # Match only the exact label token; whitespace-prefixed variants are distinct tokens.
        token = item.token

        if token in STANCE_LLM_LABELS:
            valid_logprobs[token] = item.logprob

    missing_tokens = set(STANCE_LLM_LABELS) - set(valid_logprobs)
    if missing_tokens:
        raise ValueError(
            "Missing stance-label logprobs for tokens "
            f"{sorted(missing_tokens)}. Increase top_logprobs or ensure the "
            "endpoint exposes logprobs for every guided-choice label."
        )

    max_logprob = max(valid_logprobs.values())

    exp_logprobs = {
        token: np.exp(logprob - max_logprob)
        for token, logprob in valid_logprobs.items()
    }
    normalizer = sum(exp_logprobs.values())

    return {
        STANCE_LLM_LABELS[token]: exp_logprobs[token] / normalizer
        for token in STANCE_LLM_LABELS
    }


@dataclass(frozen=True)
class StanceDecision:
    """Validated structured LLM response plus optional probability diagnostics."""

    raw_response: str
    emitted_label: str | None
    top_logprobs: list[dict[str, str | float | None]]
    probabilities: dict[str, float]
    reported_label: str
    mismatch: bool
    perplexity: float

    @classmethod
    def from_choice(cls, choice) -> "StanceDecision":
        raw_response = (choice.message.content or "").strip()
        parsed_response = getattr(choice.message, "parsed", None)
        if parsed_response is None:
            parsed_response = StanceResponse.model_validate_json(raw_response)

        emitted_label = STANCE_LLM_LABELS[parsed_response.answer]
        content = choice.logprobs.content if choice.logprobs else []
        answer_token_data = next(
            (token_data for token_data in content if token_data.token == parsed_response.answer),
            None,
        )
        if answer_token_data is None:
            raise ValueError(
                f"Could not find exact emitted stance label token {parsed_response.answer!r} "
                "in the response logprobs."
            )

        label_top_logprobs = answer_token_data.top_logprobs
        if not label_top_logprobs:
            raise ValueError(
                f"Exact emitted stance label token {parsed_response.answer!r} has no top logprobs."
            )

        probabilities = get_labels_distribution(label_top_logprobs)
        probability_label = max(probabilities, key=probabilities.get)
        mismatch = emitted_label != probability_label

        response_logprobs = [
            token.logprob
            for token in content
            if token.logprob is not None
        ]

        return cls(
            raw_response=raw_response,
            emitted_label=emitted_label,
            top_logprobs=[
                {"token": item.token, "logprob": item.logprob}
                for item in (label_top_logprobs or [])
            ],
            probabilities=probabilities,
            reported_label=emitted_label,
            mismatch=mismatch,
            perplexity=(
                float(np.exp(-np.mean(response_logprobs)))
                if response_logprobs
                else float("nan")
            ),
        )



async def score_stance_custom(
    df: pd.DataFrame,
    stance_claim: str,
    client: AsyncOpenAI,
    model_name: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Score passages against a stance claim via a chat LLM endpoint.

    Requests are executed concurrently. The validated ``StanceResponse``
    answer is authoritative. Label probabilities and perplexity are retained
    when the endpoint returns usable logprobs, but are never used to replace
    the structured answer.
    """
    if df.empty:
        return df.copy()

    scored, passages, hypotheses = _prepare_scored_frame(df, stance_claim)

    semaphore = asyncio.Semaphore(max_workers)

    async def score_passage(passage: str):
        async with semaphore:
            return await _stance_llm_choice(client, model_name, stance_claim, passage)

    choices = await asyncio.gather(*(score_passage(passage) for passage in passages))

    decisions = [StanceDecision.from_choice(choice) for choice in choices]
    stance = [decision.reported_label for decision in decisions]
    raw_probs = [decision.probabilities for decision in decisions]
    perplexities = [decision.perplexity for decision in decisions]

    probs_arr = np.asarray(
        [
            [probabilities[label] for label in STANCE_CLASSES]
            for probabilities in raw_probs
        ],
        dtype=np.float64,
    )

    label_index = {label: index for index, label in enumerate(STANCE_CLASSES)}

    scored["stance_prob_in_favor"] = probs_arr[:, label_index["IN FAVOR"]]
    scored["stance_prob_against"] = probs_arr[:, label_index["AGAINST"]]
    scored["stance_prob_neutral"] = probs_arr[:, label_index["NEUTRAL"]]
    scored["stance_margin"] = (
        scored["stance_prob_in_favor"] - scored["stance_prob_against"]
    )
    scored["stance"] = stance
    scored["perplexity"] = perplexities
    scored["confidence"] = [
        probs_arr[row, label_index[label]] for row, label in enumerate(stance)
    ]
    scored["hypothesis"] = hypotheses
    scored["llm_raw_response"] = [decision.raw_response for decision in decisions]
    scored["llm_emitted_label"] = [decision.emitted_label for decision in decisions]
    scored["llm_top_logprobs"] = [decision.top_logprobs for decision in decisions]
    scored["llm_label_probabilities"] = [
        decision.probabilities for decision in decisions
    ]
    scored["llm_label_mismatch"] = [decision.mismatch for decision in decisions]
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
        "llm_raw_response",
        "llm_emitted_label",
        "llm_top_logprobs",
        "llm_label_probabilities",
        "llm_label_mismatch",
        "perplexity",
        "stance_prob_in_favor",
        "stance_prob_neutral",
        "stance_prob_against",
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
