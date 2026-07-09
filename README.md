# MappingDemocraSci

## Project Description
In modern democracies, the lack of transparent and accessible information about elected representatives leads to voter disillusionment with politics. To address this issue, our project aims to extract and present information about politicians’ stances in a neutral and simplified fashion for public consumption. Politicians often declare interest in specific issues and try to influence the political agenda in their favor. This behavior is known as advocacy, where politicians take positions (or stances) on issues in their political work. By leveraging advances in data science, particularly natural language processing, we analyze legislative bills and parliamentary speeches to map legislative advocacy and detect the political stances of Swiss members of parliament. The project addresses three key questions: how to map legislative advocacy; 2) how to detect political stances accurately; and 3) how to combine insights from advocacy maps to determine stance congruence of MPs.


## Current Workflow
Prerequisite: `data/speeches_with_text.parquet` must already exist. The script does not build it; it consumes the preprocessed speech-level dataset produced upstream.

If the large files have not been downloaded yet, run:
```bash
git lfs install
git lfs pull
```
This will fetch `data/speeches_with_text.parquet`, which is tracked by Git LFS.

The current CLI pipeline in `main.py` is:

1. Load `data/speeches_with_text.parquet`.
2. Run `df_preprocessing()` to keep bill-related speech rows, normalize `Bill_Number`, derive `Affair_ID`, and attach bill metadata when available.
3. Aggregate speeches by `Affair_ID` with `df_bill_metadata()` to build the bill table used for retrieval.
4. Rank bills with `score_bills()` in `bill_paragraph_filtering.py`.
   - bill search text is built from `title`, `keywords`, and `ideological_summary`
   - embeddings are computed with the [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) model
   - bill embeddings are cached in `data/bill_embeddings.safetensors` and reused if the bill table fingerprint and model metadata match
   - the cross-encoder reranker is [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
   - the cross-encoder reranks the bill candidates after the dense pass
5. Restrict the speech dataframe to speeches linked to the selected bills with `select_target_speeches()`.
6. Rank passages with `score_paragraphs()`.
   - paragraphs are embedded with the same E5 model
   - the question embedding is used for dense similarity
   - the cross-encoder reranks paragraph candidates
   - passages above the rerank threshold are marked as relevant
7. Run stance scoring with `score_stance()` in `stance_detection.py`.
   - the current notebook and CLI use `--question` as the hypothesis
   - each passage is paired with that hypothesis and scored with [`MatteoFasulo/xlm-roberta-xstance`](https://huggingface.co/MatteoFasulo/xlm-roberta-xstance)
   - the output contains support/oppose/neutral probabilities, a label, and a confidence score
8. Save outputs to `--output-dir`:
   - `candidate_bills.csv`
   - `candidate_paragraphs.csv`
   - `stance_results.json`
   - optionally `preprocessed_speeches.csv` and `bill_metadata.csv` when `--save-intermediate` is set

Parameter semantics:
- `--topic` is a run label / analysis tag for the policy issue. In the current CLI it is not yet used in retrieval or stance scoring, because the pipeline currently operates on the single `--question` string for both dense retrieval and stance hypothesis construction.
- Planned integration: `--topic` should become the policy filter / retrieval constraint in a later iteration, for example by narrowing the bill candidate pool with topic keywords, topic-specific bill metadata, or a topic-to-query expansion step before dense retrieval. That would let `--topic` drive bill selection while `--question` remains the stance hypothesis.
- `--question` is the actual retrieval target and stance hypothesis. It should be the concrete statement or policy claim you want to test against the passages.
- Planned integration: the retrieval stage and the stance stage currently share the same string for convenience, but these are not conceptually identical tasks. Retrieval benefits from a broader query that improves recall, while stance detection benefits from a precisely framed hypothesis. We should separate them so the retrieval query can be optimized for evidence discovery and the stance hypothesis can be phrased for the classifier.

## Run
```bash
python main.py \
  --input-csv data/speeches_with_text.parquet \
  --output-dir outputs \
  --topic "<topic>" \
  --question "<hypothesis>"
```

