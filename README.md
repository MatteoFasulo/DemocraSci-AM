# MappingDemocraSci

## Workflow

Prerequisite: `data/speeches_with_text.parquet` must already exist. The project does not build it; it consumes the preprocessed speech-level dataset produced upstream.

If the large files have not been downloaded yet, run:
```bash
git lfs install
git lfs pull
```

The current CLI in `main.py` works like this:

1. Load `data/speeches_with_text.parquet`.
2. Run `df_preprocessing()` to keep bill-related speech rows, clean the speech text, and attach bill metadata when available.
3. Aggregate speeches by `Affair_ID` with `df_bill_metadata()` to build the bill table used for retrieval. Each bill's search text combines its German bill title (`bill_title_de`) and statutory text (`statutory_text`).
4. Rank bills with the multilingual E5 embedding model and keep the fixed top 50 bills (by means of cosine similarity) for the next step. This passage needs to have hich recall, meaning that we want to keep as many relevant bills as possible, even if it means including some unrelated ones that are on-topic but potentially unrelated to the exact stance claim. The next step will filter out the unrelated ones.
5. Restrict speeches to those linked to the selected bills.
6. Rank passages with the BGE cross-encoder, save the full dense pool, then filter out passages with `rerank_score < RERANK_THRESHOLD`. This step needs high precision, meaning that we want to keep only the passages that are highly relevant to the stance claim. Given that we have already restricted the candidate passages to those linked to the top 50 bills, we can afford to be more selective here.
7. Run stance scoring with the fine-tuned `MatteoFasulo/xlm-roberta-xstance` model.

>Note: `--topic` is required and must be one of the 21 predefined topics in `const.py`. If `--stance-claim` is omitted, it defaults to `--retrieval-query` meaning that we use the same text for both retrieval and stance scoring. If you want to use a different stance claim, you can provide it with `--stance-claim`.

## Outputs

Each run writes a new directory under `outputs/` containing:

- `candidate_bills.json`: the top 50 bills retrieved for the query.
- `candidate_passages_before_threshold.json`: the full dense pool of passages ranked by the BGE cross-encoder before filtering to inspect the reranker score distribution.
- `candidate_passages.json`: the filtered passages with `rerank_score >= RERANK_THRESHOLD`.
- `stance_predictions.json`: the stance predictions for each passage, including the predicted stance label.
- `manifest.json`: a manifest of the run, including the retrieval query, stance claim, topic, and other parameters.

## Run

```bash
python main.py \
  --run-name free-trade-us \
  --topic "International Affairs" \
  --retrieval-query "Should Switzerland strive for a free-trade agreement with the US?"
```

Optional controls include `--paragraph-dense-pool` and `--batch-size`.

## Dashboard

The same workflow is available through Streamlit:

```bash
streamlit run dashboard.py
```

Enter a retrieval query, select a topic, and optionally provide a separate stance claim. The dashboard writes the same JSON artifacts and manifest as the CLI under the configured output directory.
