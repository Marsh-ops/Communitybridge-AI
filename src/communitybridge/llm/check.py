"""Check that the configured model answers: `python -m communitybridge.llm.check`.

This is the live half of 's done-when ("gets a reply from Ollama"). It asks the model for one
tiny structured reply through the same `generate` path the agents will use, so a pass shows the
whole chain works: settings, server, model, JSON schema, validation. A failure says what to fix in
plain words, because the first run on a new machine is usually a missing `ollama pull`.

Exit code 0 means the model answered correctly, 1 means it did not. Nothing here is personal data
or a PATS question; it only proves the model can follow an output format.
"""

import sys
import time
from collections.abc import Mapping
from typing import Literal

from communitybridge.config import ConfigError, ollama_settings
from communitybridge.llm.base import LLMError, LLMProvider, LLMUnavailableError
from communitybridge.llm.ollama import OllamaProvider
from communitybridge.llm.structured import generate
from communitybridge.models.common import StrictModel
from communitybridge.models.llm import ChatMessage


class CheckReply(StrictModel):
    status: Literal["ready"]


def main(*, provider: LLMProvider | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Run the check. Tests pass a provider; otherwise Ollama is built from the settings."""
    model = None
    if provider is None:
        try:
            settings = ollama_settings(environ)
        except ConfigError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        provider = OllamaProvider(settings)
        model = settings.model

    started = time.perf_counter()
    try:
        result = generate(
            provider, CheckReply, [ChatMessage(role="user", content="Say that you are ready.")]
        )
    except LLMUnavailableError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        print(
            "Check that Ollama is running (ollama serve), that the model in OLLAMA_MODEL has "
            "been downloaded (ollama pull <model>), and Ollama's own log for the cause "
            "(on Windows: %LOCALAPPDATA%\\Ollama\\server.log).",
            file=sys.stderr,
        )
        return 1
    except LLMError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    seconds = time.perf_counter() - started
    calls = f"{result.attempts} call" + ("" if result.attempts == 1 else "s")
    label = f"{provider.name} ({model})" if model else provider.name
    print(f"OK: {label} gave a valid structured reply in {calls}, {seconds:.1f} s")
    print(f"reply: {result.value.model_dump_json()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
