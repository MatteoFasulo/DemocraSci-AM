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
2. Run `df_preprocessing()` to keep bill-related speech rows, normalize `Bill_Number`, derive `Affair_ID`, and attach bill metadata when available.
3. Aggregate speeches by `Affair_ID` with `df_bill_metadata()` to build the bill table used for retrieval.
4. Rank bills with the multilingual E5 embedding model and keep the fixed top 50 bills.
5. Restrict speeches to those linked to the selected bills.
6. Rank passages with the BGE cross-encoder, save the full dense pool, then filter out passages with `rerank_score < RERANK_THRESHOLD`.
7. Run stance scoring with the fine-tuned `MatteoFasulo/xlm-roberta-xstance` model.

`--topic` is required and must be one of the 21 predefined topics in `const.py`.
If `--stance-claim` is omitted, it defaults to `--retrieval-query`.

## Outputs

Each run writes a new directory under `outputs/` containing:

- `candidate_bills.json`
- `candidate_passages_before_threshold.json`
- `candidate_passages.json`
- `stance_predictions.json`
- `manifest.json`

The pre-threshold passage file is the one to inspect if you want to study the reranker score distribution before filtering.

## Run

```bash
python main.py \
  --run-name free-trade-us \
  --topic "International Affairs" \
  --retrieval-query "Should Switzerland strive for a free-trade agreement with the US?"
```

Optional controls include `--input-data`, `--output-dir`, `--stance-claim`, `--paragraph-dense-pool`, and `--batch-size`.
