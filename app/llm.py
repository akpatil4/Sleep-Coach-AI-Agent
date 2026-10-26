import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_client() -> OpenAI:
    token = os.getenv("METERLESS_RUNTIME_TOKEN")
    if not token:
        raise RuntimeError("METERLESS_RUNTIME_TOKEN is not set.")
    return OpenAI(
        api_key=token,
        base_url=os.environ["METERLESS_BASE_URL"],
        default_headers={"meterless-app": "sleep-experiment-coach",
                         "meterless-scope": "user"},
        timeout=60.0,
        max_retries=1,
    )

def chat(messages: list[dict], temperature: float = 0.2) -> dict:
    response = get_client().chat.completions.create(
        model=os.getenv("METERLESS_MODEL", "meterless/auto"),
        messages=messages,
        temperature=temperature,
    )
    return {"content": response.choices[0].message.content,
            "model": response.model}
