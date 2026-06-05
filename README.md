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
- `paragraph_embeddings_Qwen3-Embedding-0.6B.npy`: embeddings of the cleaned paragraphs obtained using the Qwen3-Embedding-0.6B model (available at https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## Notion
There is a Notion page associated with this project available here: https://www.notion.so/Stance-Detection-3494ebb13c4b80b0ba17fbe2ded82c21

## Code

Install the required packages via `pip` or `uv`:

with pip:
```bash
pip install -r pyproject.toml
```

with uv:
```bash
uv sync # should work and install all the deps
```

## SDSC Qwen
Reference: https://towering-canid-e21.notion.site/LLM-Deployment-Usage-229891adfe32804582b9efbc8288ec0e

It requires to have access to the SDSC Qwen. The procedure to obtain access can be found at the reference link above. Once you have access, we will just use regular cURL requests to interact with the endpoint of vLLM. What is needed is the API Key.

**Note**: the list of models available in the endpoint might change over time, so it is recommended to check the available models before sending the request. The list of available models can be obtained by sending a GET request to the same endpoint:
```bash
curl -X GET https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1/models \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <YOUR_API_KEY>"
```

After obtaining the list of available models, you can choose the one that best suits your needs and use it in the POST request to generate completions. As of today (May 2026), the following models are available:
- Qwen/Qwen3.6-35B-A3B-FP8
- google/gemma-4-26B-A4B-it

```bash
curl -X POST https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-d '{
  "model": "<MODEL_NAME>",
  "messages": [
    {
      "role": "user",
      "content": "Hi, how are you?"
    }
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 256,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```
here we use the <MODEL_NAME> without thinking mode (note the **chat_template_kwargs** flags). For text generation tasks, it is recommended to check the specific documentation of the model you are using to use the suggested parameters for generation.

### [New] OpenAI Python package
Reference: https://docs.vllm.ai/en/stable/getting_started/quickstart/#openai-chat-completions-api-with-vllm

Alternatively, you can also use the OpenAI Python package to interact with the SDSC Qwen endpoint described above. You can install the package via pip:
```bash
pip install openai
```
Then, you can use the following code to generate completions:
```python
from openai import OpenAI
# Set API key and API base to use vLLM's API server.
openai_api_key = "<YOUR_API_KEY>"
openai_api_base = "https://vllm-gateway-runai-sharedllm-ralf.inference.compute.datascience.ch/v1/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Check which model is available in the vLLM endpoint
models = client.models.list()

chat_response = client.chat.completions.create(
    model="<MODEL_NAME>",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=256,
    presence_penalty=1.5,
    extra_body={
	    "top_k": 20,
	    "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)
print("Chat response:", chat_response)
```

## Notebooks

- `bertopic.ipynb`: notebook for the clustering and topic modelling using [BERTopic](https://maartengr.github.io/BERTopic/index.html) later refined by SDSC Qwen and then used to assign topics to the speeches paragraphs.
- `NLI.ipynb`: notebook for the NLI-based stance detection where we load the output of the clustering phase, construct two hypothesis templates and feed them to the NLI model (mDeBERTaV3 - finetuned on XNLI) to obtain the stances of the MPs on the different topics.

## Python Scripts

- `const.py`: contains the constants used in the project (like some URL for the paragraph extraction and system and user prompts for the LLMs during the topic refinement step).
- `paragraph_extraction.py`: script to extract the paragraphs from the speeches transcripts.
- `MP_metadata.py`: script to extract the metadata of the MPs (like their name, party, etc.) from the Neo4J database.