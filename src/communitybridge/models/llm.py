"""Boundary models for the LLM layer: the chat message and the Ollama settings.

Both are `StrictModel`s like every other boundary in the project, so a
malformed message or a mistyped setting fails when it is built, not somewhere inside a request.
The settings are read-only: once the model name, URL and timeout are loaded they cannot be changed
by a later step, so every call in a run (and every figure in an evaluation) used the same
model.
"""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator

from communitybridge.models.common import NonEmptyStr, StrictModel

Role = Literal["system", "user", "assistant"]


class ChatMessage(StrictModel):
    """One message in a conversation with a model."""

    model_config = ConfigDict(frozen=True)

    role: Role
    # Not stripped (unlike NonEmptyStr): spacing and line breaks are part of a prompt.
    content: Annotated[str, Field(min_length=1)]


class OllamaSettings(StrictModel):
    """Where the local Ollama server is and which model to ask. Loaded from `.env` (config.py)."""

    model_config = ConfigDict(frozen=True)

    model: NonEmptyStr
    base_url: NonEmptyStr = "http://localhost:11434"
    # Local models on a CPU can be slow to load the first time, so the default is generous.
    timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 120

    @field_validator("base_url")
    @classmethod
    def _http_only_without_trailing_slash(cls, value: str) -> str:
        # urllib would also open file:// and ftp:// URLs; this layer only ever talks HTTP to Ollama.
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")
