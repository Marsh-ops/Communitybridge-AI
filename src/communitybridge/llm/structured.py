"""Validated generation: a model reply only counts if it passes its Pydantic schema.

The answer to an unreliable local model is "schema validation with
retries; capped hosted API behind the same interface". This is the first half. `generate` asks the
provider for JSON, validates the reply against a Pydantic model, and if it fails asks again, telling
the model which fields were wrong. It allows at most two retries (three calls in all). After that it raises `LLMOutputError` and the calling agent takes its fallback
path, so a model that keeps failing can never leave an unchecked value in the plan.

Two design points worth being able to explain:

- **The schema forbids unknown fields** (every boundary model is a `StrictModel`). A reply that
  invents a field such as `eligible` fails validation like any other bad reply, which is how the
  rule "no language model output can set or change an eligibility result" is
  enforced on the data itself and not just by asking nicely in a prompt.
- **Only bad output is retried.** If the server is down or too slow, `LLMUnavailableError` goes
  straight to the caller: asking again immediately would not help, and the Coordinator needs to know
  quickly so it can re-plan or escalate.

The correction message names the fields that failed and the rule they broke, never the rejected
values, because a value can be personal data and the message is also written into the trace.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from communitybridge.llm.base import LLMOutputError, LLMProvider
from communitybridge.models.llm import ChatMessage

# One call plus two retries. Small local models usually fix a missing field when told which one.
MAX_RETRIES = 2
# A reply with many problems is mostly wrong everywhere; five is enough to correct it.
MAX_PROBLEMS_SHOWN = 5

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class Generated(Generic[M]):
    """A validated value and how many calls it took (shown in the trace; a retry is a cost)."""

    value: M
    attempts: int


def generate(
    provider: LLMProvider,
    schema: type[M],
    messages: Sequence[ChatMessage],
    *,
    max_retries: int = MAX_RETRIES,
    check: Callable[[M], str | None] | None = None,
) -> Generated[M]:
    """Ask `provider` for a reply that validates as `schema`, retrying bad output.

    `check` is an optional second test for a rule a schema cannot express, such as "cites only
    sections that are in the decision". It gets the validated value and returns None if it
    is fine, or one sentence saying what is wrong. A reply that fails it is retried like a reply
    that fails the schema, within the same two retries, and the sentence is shown to the model. The
    sentence may also reach the trace and the final error, so it must not repeat personal data.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries can't be negative, got {max_retries}")
    json_schema = schema.model_json_schema()
    # The schema goes in the prompt as well as in the request: the Ollama documentation
    # recommends both, since the prompt grounds the model and the request constrains the output.
    base = [ChatMessage(role="system", content=_instruction(json_schema)), *messages]
    conversation = list(base)
    attempts = max_retries + 1
    problems = ""
    for attempt in range(1, attempts + 1):
        reply = provider.complete(conversation, json_schema=json_schema)
        try:
            value = schema.model_validate_json(reply)
        except ValidationError as exc:
            problems = _describe(exc)
        else:
            rule_problem = check(value) if check is not None else None
            if rule_problem is None:
                return Generated(value=value, attempts=attempt)
            problems = rule_problem
        # Replay only the latest failure, not every earlier one, so the prompt stays small.
        conversation = [
            *base,
            # An empty message isn't allowed, so say what happened instead.
            ChatMessage(role="assistant", content=reply or "(empty reply)"),
            ChatMessage(
                role="user",
                content=(
                    f"That reply was not accepted: {problems}. "
                    "Reply again with only the corrected JSON object."
                ),
            ),
        ]
    raise LLMOutputError(
        f"{schema.__name__}: no valid reply after {attempts} calls. Last problems: {problems}",
        attempts=attempts,
    )


def _instruction(json_schema: dict) -> str:
    return (
        "Reply with only a JSON object that matches this JSON Schema, and no other text:\n"
        + json.dumps(json_schema)
    )


def _describe(error: ValidationError) -> str:
    """Name each problem as `field: rule`, without echoing the value that broke it."""
    found = error.errors()
    shown = []
    for problem in found[:MAX_PROBLEMS_SHOWN]:
        field = ".".join(str(part) for part in problem["loc"])
        if problem["type"] == "json_invalid":
            shown.append("the reply was not valid JSON")
        elif problem["type"] == "model_type":
            shown.append("the reply must be a JSON object")
        elif field:
            shown.append(f"{field}: {problem['msg']}")
        else:
            shown.append(problem["msg"])
    text = "; ".join(shown)
    extra = len(found) - MAX_PROBLEMS_SHOWN
    return f"{text}; and {extra} more" if extra > 0 else text
