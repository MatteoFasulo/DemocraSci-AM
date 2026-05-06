# MappingDemocraSci

## Project Description
In modern democracies, the lack of transparent and accessible information about elected representatives leads to voter disillusionment with politics. To address this issue, our project aims to extract and present information about politicians’ stances in a neutral and simplified fashion for public consumption. Politicians often declare interest in specific issues and try to influence the political agenda in their favor. This behavior is known as advocacy, where politicians take positions (or stances) on issues in their political work. By leveraging advances in data science, particularly natural language processing, we analyze legislative bills and parliamentary speeches to map legislative advocacy and detect the political stances of Swiss members of parliament. The project addresses three key questions: how to map legislative advocacy; 2) how to detect political stances accurately; and 3) how to combine insights from advocacy maps to determine stance congruence of MPs.

### WP1
**Argument** **detection** and **definition** and **identification** of **policy issues** (WP1):
- We extract **claims** and **premises** from legislative speeches and bills
- identify **advocacy topics** through **embedding-based clustering**
- organize them **hierarchically** to build interpretable advocacy maps.


### WP2
**Stance** **detection** (WP2):
- We apply **multilingual**, **target**-**specific** stance prioritization to create an order between different policy topics discussed by each MPs.
    - This novel approach to capture stances provides a more realistic view of the phenomenon, as politicians are not often frontally again specfic topics, but rather prioritise them.

### WP3
**Advocacy** **analysis** (WP3):
- We combine advocacy and stance data within a knowledge-graph framework to construct indices of MPs’ issue engagement
- analyze stance congruence
- study peer influence in legislative behavior

## Data

- `clean_paragraphs.parquet`: cleaned paragraphs from legislative speeches ready for analysis (generated using `paragraph_extraction.py`). It is derived starting from the last 4 legislative periods (available at https://zenodo.org/records/13920293) combined with the speeches transcripts (available at the same link inside the SQL database)
- `240102_bills_swissdomainbert2_predictions.parquet`: predictions of the SwissDomainBERT2 model for the bills (the 21 predefined bills topics)
- `paragraph_embeddings_Qwen3-Embedding-0.6B.npy`: embeddings of the cleaned paragraphs obtained using the Qwen3-Embedding-0.6B model (available at https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)


## Code

Install the required packages via `pip` or `uv`:

with pip:
```bash
pip install -r pyproject.toml
```

with uv:
```bash
uv sync
```

## SDSC Qwen
Reference: https://towering-canid-e21.notion.site/LLM-Deployment-Usage-229891adfe32804582b9efbc8288ec0e

It requires to have access to the SDSC Qwen. The procedure to obtain access can be found at the reference link above. Once you have access, we will just use regular cURL requests to interact with the endpoint of vLLM.

What is needed is the API Key.

## Notebooks

- `bertopic.ipynb`: notebook for the clustering and topic modelling using [BERTopic](https://maartengr.github.io/BERTopic/index.html) later refined by SDSC Qwen and then used to assign topics to the speeches paragraphs.
- `NLI.ipynb`: notebook for the NLI-based stance detection where we load the output of the clustering phase, split at the sentence level, construct an hypothesis template for each sentence and feed it to the NLI model (mDeBERTaV3 - finetuned on XNLI) to obtain the stances of the MPs on the different topics.

## Python Scripts

- `const.py`: contains the constants used in the project (like some URL for the paragraph extraction and system and user prompts for the LLMs during the topic refinement step).