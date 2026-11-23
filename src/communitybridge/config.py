"""Project paths and settings in one place, so no module hard-codes a file location.

The package is installed in editable mode from the repo, so the project root is two levels
above this package (src/communitybridge -> src -> root).
"""

import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from communitybridge.models.llm import OllamaSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Local settings (git-ignored). `.env.example` lists every setting. Only the LLM settings below are
# read from this file so far; MAX_SOURCE_AGE_DAYS is read from the real environment only.
ENV_PATH = PROJECT_ROOT / ".env"

# SQLite database built by scripts/seed_db.py from the synthetic dataset (*.db is git-ignored).
DB_PATH = DATA_DIR / "communitybridge.db"

# LangGraph's checkpoints: the graph state saved after every step, so a run can pause at an
# approval gate and resume later. It is a different file from DB_PATH on purpose: service_query
# opens the dataset read-only, and nothing the graph writes should be able to reach it.
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"

# Versioned PATS rules encoded from the V26.4 PDF. A guideline update changes this file only.
RULES_PATH = DATA_DIR / "rules" / "pats_v26_4.yaml"

# Minimum personal data per action, read only by the Consent Agent.
POLICY_PATH = DATA_DIR / "policy" / "data_fields.yaml"

# How old a source may be before the Verifier refuses to show a statement based on it as current.
# No figure is set by the guidelines, so 180 days (about six months) is the starting point; it can
# be changed with the MAX_SOURCE_AGE_DAYS setting (see .env.example) without touching the code.
DEFAULT_MAX_SOURCE_AGE_DAYS = 180


class ConfigError(Exception):
    """A setting has a value that can't be used."""


def max_source_age_days(environ: Mapping[str, str] | None = None) -> int:
    """The MAX_SOURCE_AGE_DAYS setting: a whole number of days, at least 1."""
    env = os.environ if environ is None else environ
    raw = env.get("MAX_SOURCE_AGE_DAYS", "").strip()
    if not raw:
        return DEFAULT_MAX_SOURCE_AGE_DAYS
    try:
        days = int(raw)
    except ValueError:
        raise ConfigError(
            f"MAX_SOURCE_AGE_DAYS must be a whole number of days, got {raw!r}"
        ) from None
    if days < 1:
        raise ConfigError(f"MAX_SOURCE_AGE_DAYS must be at least 1 day, got {days}")
    return days


def read_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    """Read NAME=value lines from a .env file. A missing file is no settings, not an error.

    Only whole-line comments are understood, and one pair of surrounding quotes is removed. That is
    all `.env.example` needs, and it avoids adding a dependency just to read a few lines.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name.strip()] = value
    return values


# Which setting a bad OllamaSettings field came from, so the error names what to edit in .env.
_OLLAMA_SETTING_NAMES = {
    "model": "OLLAMA_MODEL",
    "base_url": "OLLAMA_BASE_URL",
    "timeout_seconds": "OLLAMA_TIMEOUT_SECONDS",
}


def ollama_settings(
    environ: Mapping[str, str] | None = None, env_file: Path = ENV_PATH
) -> OllamaSettings:
    """The OLLAMA_* settings.

    With no `environ`, real environment variables are used and win over the .env file, so a single
    run can override the file. With an explicit `environ` (tests), only that mapping is used and the
    file is never read, so a developer's own .env cannot change a result.
    """
    if environ is None:
        values = {**read_env_file(env_file), **os.environ}
    else:
        values = dict(environ)
    raw: dict[str, object] = {"model": values.get("OLLAMA_MODEL", "").strip()}
    if values.get("OLLAMA_BASE_URL", "").strip():
        raw["base_url"] = values["OLLAMA_BASE_URL"].strip()
    timeout = values.get("OLLAMA_TIMEOUT_SECONDS", "").strip()
    if timeout:
        try:
            raw["timeout_seconds"] = float(timeout)
        except ValueError:
            raise ConfigError(
                f"OLLAMA_TIMEOUT_SECONDS must be a number of seconds, got {timeout!r}"
            ) from None
    if not raw["model"]:
        raise ConfigError(
            "OLLAMA_MODEL is not set. Copy .env.example to .env and set the model you pulled, "
            "for example OLLAMA_MODEL=qwen2.5:7b (see the model decision record)"
        )
    try:
        return OllamaSettings.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        name = _OLLAMA_SETTING_NAMES.get(str(first["loc"][0]), str(first["loc"][0]))
        raise ConfigError(f"{name} can't be used: {first['msg']}") from None


def reference_date_override(
    environ: Mapping[str, str] | None = None, env_file: Path = ENV_PATH
) -> date | None:
    """The COMMUNITYBRIDGE_TODAY setting: the date "next month" is worked out from, or None.

    Without it the web app uses the real date. The synthetic service data is built around
    November 2026 (December appointments are fixtures made to fail, for re-planning,), so a
    demonstration recorded in December would re-plan the core journey instead of showing it. This
    setting lets the demo, and only the demo, fix the date, in the same way the command-line demo
    uses its scenario's own date. The page always shows the date in use.

    Read like the Ollama settings: with no `environ`, real environment variables win over `.env`;
    with an explicit one (tests), only that mapping is used.
    """
    values = {**read_env_file(env_file), **os.environ} if environ is None else dict(environ)
    raw = values.get("COMMUNITYBRIDGE_TODAY", "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise ConfigError(
            f"COMMUNITYBRIDGE_TODAY must be a real date written YYYY-MM-DD, got {raw!r}"
        ) from None
