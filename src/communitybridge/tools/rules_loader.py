"""Load and validate the versioned PATS rules file.

Used by the rules engine (Eligibility) and the logistics calculator. Every failure
becomes a single named exception, `RulesDataError`, so callers handle one error type and the
message says exactly what is wrong (missing file, bad YAML, or which field failed validation).
A rules file that fails validation is never partly used.
"""

from pathlib import Path

import yaml
from pydantic import ValidationError

from communitybridge.config import RULES_PATH
from communitybridge.models.rules import RuleSet


class RulesDataError(Exception):
    """The PATS rules file is missing, unreadable or does not match the schema."""


def load_rules(path: Path = RULES_PATH) -> RuleSet:
    """Read the rules YAML at `path` and return a validated RuleSet."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RulesDataError(f"rules file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RulesDataError(f"rules file is not valid YAML: {path}: {exc}") from exc

    try:
        return RuleSet.model_validate(raw)
    except ValidationError as exc:
        raise RulesDataError(f"rules file failed validation: {path}:\n{exc}") from exc
