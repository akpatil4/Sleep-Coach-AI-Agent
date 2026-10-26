# app/llm.py
# EVERY model call in the project goes through this file, so "Nightshift is the
# primary AI provider" is true by construction.
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
        timeout=180.0,       # a first model load can take minutes
        max_retries=1,
    )


def _prompt_first(messages: list[dict]) -> list[dict]:
    """Nightshift requires the prompt (a user message) to be the FIRST message.
    Fold any leading system messages into that first user message."""
    system_parts, rest = [], list(messages)
    while rest and rest[0]["role"] == "system":
        system_parts.append(rest.pop(0)["content"])
    if not system_parts:
        return rest
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user",
                   "content": "\n\n".join(system_parts) + "\n\n" + rest[0]["content"]}
    else:
        rest.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})
    return rest


def _lowercase(messages: list[dict]) -> list[dict]:
    """Workaround: Nightshift's local model lane returned 502 for prompts with capitalised
    text (found by bisecting the prompt). Lowercasing is harmless to our checks, which
    compare case-insensitively."""
    return [{"role": m["role"], "content": m["content"].lower()} for m in messages]


def chat(messages: list[dict], temperature: float = 0.2) -> dict:
    response = get_client().chat.completions.create(
        model=os.getenv("METERLESS_MODEL", "meterless/auto"),
        messages=_lowercase(_prompt_first(messages)),
        temperature=temperature,
    )
    return {"content": response.choices[0].message.content,
            "model": response.model}   # the model that actually answered