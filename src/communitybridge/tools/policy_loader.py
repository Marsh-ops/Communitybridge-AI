"""Load and validate the consent data-field policy (Consent Agent,; used by tools/consent.py).

Like the rules loader, every failure becomes one named exception, `PolicyDataError`. A policy that
fails validation is never partly used: no policy means no data release at all.
"""

from pathlib import Path

import yaml
from pydantic import ValidationError

from communitybridge.config import POLICY_PATH
from communitybridge.models.consent_policy import ConsentPolicy


class PolicyDataError(Exception):
    """The consent policy file is missing, unreadable or breaks a minimisation rule."""


def load_consent_policy(path: Path = POLICY_PATH) -> ConsentPolicy:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyDataError(f"consent policy not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise PolicyDataError(f"consent policy is not valid YAML: {path}: {exc}") from exc
    try:
        return ConsentPolicy.model_validate(raw)
    except ValidationError as exc:
        raise PolicyDataError(f"consent policy failed validation: {path}:\n{exc}") from exc
