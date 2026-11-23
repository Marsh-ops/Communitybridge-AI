"""The provider interface every language model sits behind, and its named errors.

The LLM sits behind a provider-agnostic interface: `OllamaProvider` for
development, `FakeProvider` for tests, and an optional capped hosted provider that is a fallback
only (the mitigation for an unreliable local model). Agents depend on
`LLMProvider`, never on a particular model, so swapping the model is a settings change and no test
ever needs a model running.

A provider does one thing: take messages and return the model's reply as text. Turning that text
into a checked value is `structured.generate`, so the rule "a reply must pass its schema" is written
once and applies to every provider. Language models only interpret the goal, coordinate and explain;
they never decide eligibility, and keeping the interface this small is part of how
that is enforced.

Every failure is an `LLMError` subclass, so an agent can catch them together, write a TraceEvent and
take its fallback path.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from communitybridge.models.llm import ChatMessage


class LLMError(Exception):
    """Base of every language-model failure."""


class LLMUnavailableError(LLMError):
    """The model can't be used right now: server down, too slow, or it answered with an error."""


class LLMResponseError(LLMError):
    """The server answered, but not in the shape expected (no message, not JSON)."""


class LLMOutputError(LLMError):
    """The model's reply still failed its schema after every retry."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class LLMProvider(ABC):
    """A language model that turns messages into a text reply.

    `calls` counts requests made, whether or not they succeeded. It is the "LLM calls per journey"
    figure for the efficiency indicators.
    """

    name: str

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, messages: Sequence[ChatMessage], *, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Ask the model once. `json_schema`, when given, constrains the reply to that shape."""
        if not messages:
            raise ValueError("a request needs at least one message")
        self.calls += 1
        return self._complete(messages, json_schema)

    @abstractmethod
    def _complete(self, messages: Sequence[ChatMessage], json_schema: dict[str, Any] | None) -> str:
        """Send one request. Subclasses raise `LLMError` subclasses and nothing else."""
