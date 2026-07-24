import os
import math
import argparse
import json
import openai

SYSTEM_PROMPT = """
You are an annotator for political science.

Decide whether the passage directly addresses the exact policy proposition.

Answer YES when the passage:
- supports, opposes, evaluates, or provides directly applicable information about the proposition;
- addresses the same policy objective, mechanism, or explicit trade-off.

Answer NO when the passage:
- only concerns the same broad topic;
- addresses a different policy instrument, population, institution, or outcome;
- contains no information that helps assess positions on the exact proposition.

Respond with exactly YES or NO.
"""

PROMPT = """
EXAMPLE 1
POLICY: Flatten the progression rates in taxation.
PASSAGE: Herr Nationalrat Hess hat zu Recht gesagt, dass eine Flat Rate Tax Vorteile bei der Paarbesteuerung hätte, weil die steuerliche Ungleichbehandlung zwischen verheirateten und unverheirateten Paaren weitgehend wegfallen würde. Eine Flat Rate Tax würde auch bestehende Anreize für die Steuerplanung beseitigen. Mit dem konstanten Grenzsteuersatz im indirekt progressiven Tarif spielt es in der Regel keine Rolle mehr, ob eine steuerpflichtige Person beispielsweise einen Einkauf in die Pensionskasse in einer einzigen Steuerperiode vornimmt oder auf mehrere Perioden verteilt.
ANSWER: YES

EXAMPLE 2
POLICY: End tax and AHV discrimination against married and registered couples.
PASSAGE: Damit wird eine Ungleichbehandlung in Bezug auf die erleichterte Einbürgerung beseitigt. Für Menschen in einer eingetragenen Partnerschaft ist die heutige Regel diskriminierend.
ANSWER: NO

EXAMPLE 3
POLICY: Give Swiss products priority over cheap imports.
PASSAGE: Diese Vorstösse bezwecken die Öffnung des Schweizer Marktes sowie die potenzielle Verbilligung der Importe.
ANSWER: YES

POLICY: {POLICY}
PASSAGE: {PASSAGE}

ANSWER:
"""

def get_probabilities(choice, allowed_tokens: list = ["YES", "NO"]):
    """
    A function to get the probabilities of the allowed tokens from the choice object returned by OpenAI API.

    It transforms the logprobs of the allowed tokens into probabilities by exponentiating them.
    """
    top_logprobs = choice.logprobs.content[0].top_logprobs
    prob_dict = {}
    for item in top_logprobs:
        token = item.token.strip()
        if token in allowed_tokens:
            prob_dict[token] = math.exp(item.logprob)
    return prob_dict

def get_openai_client(api_key: str, base_url: str):
    """
    A function to create an OpenAI client with the given API key and base URL.
    """
    return openai.OpenAI(api_key=api_key, base_url=base_url)

def get_endpoint_models(client: openai.OpenAI, only_free: bool = False):
    """
    A function to get the list of models available in the OpenAI API.

    It returns a list of model IDs.

    The `only_free` parameter can be set to True to filter the models to only include free models (OpenRouter models).
    """
    models = client.models.list()
    if only_free:
        models = [model for model in models.data if ":free" in model.id]
    return [model.id for model in models.data]

def prompt_llm(client: openai.OpenAI, model: str, system_prompt: str = SYSTEM_PROMPT, user_prompt: str = PROMPT):
    """
    A function to prompt the LLM with the system and user prompts.

    We set the temperature to 0 for deterministic output, and we request logprobs for the allowed tokens YES and NO.
    We remove presence penalty and set max tokens to 1 since we only want a single token output.
    The extra body highly depends on the single model, the one we use here is for vLLM, DeepSeek, and OpenRouter. It disables the thinking feature and enables guided choice for YES and NO.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        temperature=0,
        presence_penalty=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        extra_body={
            "thinking": {"type": "disabled"},
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            "guided_choice": ["YES", "NO"],
            "reasoning": {"enabled": False},
        },
    )
    return response.choices[0]

def get_responses(client: openai.OpenAI, model: str, system_prompt: str = SYSTEM_PROMPT, user_prompt: str = PROMPT, n: int = 1):
    """
    A function to get multiple responses from the LLM.

    It returns a list of response choices.
    """
    responses = []
    for _ in range(n):
        response_choice = prompt_llm(client, model, system_prompt, user_prompt)
        responses.append(response_choice)
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="Your OpenAI API key.")
    parser.add_argument("--base_url", type=str, help="The base URL for the OpenAI API.")
    parser.add_argument("--policy", type=str, required=True, help="The policy to use for the prompt.")
    parser.add_argument("--passage", type=str, required=True, help="The passage to label.")
    parser.add_argument("--model", type=str, help="The model to use for the prompt.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_API_BASE_URL")

    if not api_key or not base_url:
        raise ValueError("API key and base URL must be provided either as arguments or environment variables.")

    client = get_openai_client(api_key, base_url)

    model = args.model
    if not args.model:
        models = get_endpoint_models(client)
        if not models:
            raise ValueError("No models available from the OpenAI API endpoint.")

        print("Available models:")
        for model in models:
            print(model)
        model = input("Please select a model from the above list: ")

    # We prompt the LLM 10 times to get multiple responses for the same input, which allows us to analyze the consistency of the model's output.
    user_prompt = PROMPT.format(POLICY=args.policy, PASSAGE=args.passage)
    response_choice = get_responses(client, model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, n=10)

    # We extract the probabilities of the allowed tokens YES and NO from each response choice.
    probabilities = [get_probabilities(choice) for choice in response_choice]

    # We print the probabilities for each response choice.
    for i, prob in enumerate(probabilities):
        print(f"Response {i + 1}: {prob}")