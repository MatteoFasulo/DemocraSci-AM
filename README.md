# MappingDemocraSci

MappingDemocraSci is a two-stage retrieval and stance-detection pipeline for Swiss parliamentary speech. It retrieves bills with high recall, narrows their paragraphs with a cross-encoder, and applies a fine-tuned XLM-RoBERTa stance classifier to the retained evidence.

## Pipeline overview

```text
speech parquet -> preprocessing and bill metadata
              -> E5 bill retrieval (top 50)
              -> E5 paragraph dense pool
              -> BGE paragraph reranking
              -> thresholded evidence
              -> XLM-R stance prediction
```

Bill retrieval is the recall stage: bill search text combines the German bill title and statutory text. Paragraph reranking is the precision stage: it selects passages that address the query more closely before stance scoring. The unthresholded pool is always saved for threshold evaluation.

## Setup and data

The project requires Python 3.12 or newer. Install dependencies with:

```bash
pip install -e .
```

The CLI consumes `data/speeches_with_text.parquet`; it does not generate this file. If the split assets are available, reconstruct it with:

```bash
# We first concatenate the 3 parts into a single .gz file.
cat data/speeches_with_text.gz.part_* > data/speeches_with_text.gz

# Then decompress the .gz file into a .parquet file.
gunzip < "data/speeches_with_text.gz" > "data/speeches_with_text.parquet"
```

## Detailed pipeline

1. `df_preprocessing()` filters bill-related rows, cleans text, and attaches cached OpenParldata metadata by `Affair_ID`.
2. `df_bill_metadata()` aggregates one record per bill. `bill_search_text` contains `bill_title_de` plus `statutory_text`.
3. `score_bills()` embeds the query and bill text with `intfloat/multilingual-e5-large`, ranks by cosine similarity, and keeps `N_TOP_BILLS` (50).
4. `select_target_speeches()` restricts the paragraph corpus to the selected bills. E5 similarity forms the configurable dense pool.
5. `score_paragraphs()` reranks that pool with `BAAI/bge-reranker-v2-m3`, saves all scores, and filters on `RERANK_THRESHOLD` (currently `0.4`).
6. `score_stance()` compares each retained passage with the stance claim using `MatteoFasulo/xlm-roberta-xstance`. It outputs support/oppose probabilities and their margin; binary uncertainty is not a trained neutral class.

`--topic` must be one of the 21 topics in `const.py`. If `--stance-claim` is omitted, the retrieval query is used for stance scoring as well.

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

## Human evaluation

TBD via streamlit dashboard.

## Docker

The image for RunAI is available on GitHub Container Registry at `ghcr.io/matteofasulo/democrasci-am:latest`.

To pull the image from GitHub Container Registry, run:

```bash
docker pull ghcr.io/matteofasulo/democrasci-am:latest
```

### Development

To create the image needed for the workflow, run:

```bash
docker build --no-cache --pull -f Dockerfile -t ghcr.io/matteofasulo/democrasci-am:latest .
docker push ghcr.io/matteofasulo/democrasci-am:latest
```