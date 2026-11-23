"""OllamaProvider: the local model used in development.

Ollama runs the model on this machine and answers HTTP on localhost, so no personal data leaves it
and nothing here needs an account or an API key. That fits the original design ("Ollama as the local
LLM") and makes the project reproducible for anyone who clones it. The model name is a setting
(`OLLAMA_MODEL`, chosen in the model decision record), never part of the code.

The standard library's `urllib` is used for the HTTP call instead of adding a client library. The
whole exchange is one POST to `/api/chat`, so a dependency would be more code to understand than
the call itself, and the transport is a plain function that tests replace.

Request shape (checked against the Ollama structured-outputs documentation):
- `stream: false`, so the reply arrives as one complete message;
- `format`: the JSON Schema, so Ollama constrains the output to that shape;
- `options.temperature: 0` and a fixed `seed`, so the same prompt on the same model gives the same
  reply, which is what lets a rerun of an evaluation be compared with the first.
"""

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from communitybridge.llm.base import LLMProvider, LLMResponseError, LLMUnavailableError
from communitybridge.models.llm import ChatMessage, OllamaSettings

# (url, payload, timeout in seconds) -> the parsed JSON reply. Tests pass their own.
Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]

TEMPERATURE = 0
SEED = 42


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST `payload` as JSON and return the JSON object that comes back."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # The URL scheme is limited to http/https when the settings load (models/llm.py).
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:  # before URLError, which it extends
        raise LLMUnavailableError(f"Ollama answered HTTP {exc.code}: {_reason(exc)}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
        raise LLMUnavailableError(f"could not reach Ollama at {url}: {exc}") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("Ollama's reply was not valid JSON") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Ollama's reply was not a JSON object")
    return data


def _reason(error: urllib.error.HTTPError) -> str:
    """The `error` text Ollama puts in the body of a failed request, e.g. a model not pulled."""
    try:
        data = json.loads(error.read())
    except (json.JSONDecodeError, OSError):
        return error.reason or "no details"
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return data["error"]
    return error.reason or "no details"


class OllamaProvider(LLMProvider):
    """The local development model, reached over HTTP ( records the model choice).

    Language models only interpret the goal and explain results, so this class does
    no more than send the messages and return the reply. The HTTP call is passed in as
    `transport`, so tests can supply a fake one.
    """

    name = "ollama"

    def __init__(self, settings: OllamaSettings, transport: Transport = post_json) -> None:
        super().__init__()
        self.settings = settings
        self._transport = transport

    def _complete(self, messages: Sequence[ChatMessage], json_schema: dict[str, Any] | None) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "options": {"temperature": TEMPERATURE, "seed": SEED},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        data = self._transport(
            f"{self.settings.base_url}/api/chat", payload, self.settings.timeout_seconds
        )
        if isinstance(data.get("error"), str):
            raise LLMUnavailableError(f"Ollama reported an error: {data['error']}")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama's reply had no message text")
        return content
