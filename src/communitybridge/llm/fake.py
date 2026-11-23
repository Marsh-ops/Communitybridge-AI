"""FakeProvider: scripted replies for tests.

The project design: tests never call Ollama; they use the FakeProvider. It replays a script, one
entry per call, so an agent test can say exactly what the "model" answers, including a bad answer
(to test the retry path) or a failure (to test the fallback path). It also records every request, so
a test can check what an agent sent to the model and what schema it asked for.

The script can hold:
- a string, returned as the reply;
- an exception, raised instead (use `LLMUnavailableError` to simulate a server that is down);
- a function of the messages, for a reply that depends on what was asked.
"""

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from communitybridge.llm.base import LLMProvider
from communitybridge.models.llm import ChatMessage

Reply = str | Exception | Callable[[Sequence[ChatMessage]], str]


class FakeProviderExhausted(RuntimeError):
    """The script ran out. Deliberately not an `LLMError`: an agent's fallback path must not swallow
    a mistake in a test script and make the test pass for the wrong reason."""


class FakeProvider(LLMProvider):
    """The model used in tests: it returns scripted replies in order and records every request.

    Tests never call Ollama, so the same agent code runs against a script
    and a test can check what the agent asked for. It raises `FakeProviderExhausted` when the
    script runs out.
    """

    name = "fake"

    def __init__(self, replies: Iterable[Reply]) -> None:
        super().__init__()
        self._script: deque[Reply] = deque(replies)
        self.requests: list[tuple[ChatMessage, ...]] = []
        self.schemas: list[dict[str, Any] | None] = []

    def _complete(self, messages: Sequence[ChatMessage], json_schema: dict[str, Any] | None) -> str:
        self.requests.append(tuple(messages))
        self.schemas.append(json_schema)
        if not self._script:
            raise FakeProviderExhausted(
                f"call {self.calls} has no scripted reply; the script had fewer entries"
            )
        reply = self._script.popleft()
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(messages)
        return reply
