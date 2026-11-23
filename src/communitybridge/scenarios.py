"""Load and validate the evaluation and re-planning scenarios.

Used by the tests now, and by an evaluation script to measure the success
targets (100% eligibility accuracy, >= 90% field-level extraction accuracy). The re-planning
scenarios feed the failure-path tests.
"""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from communitybridge.config import DATA_DIR
from communitybridge.models.extraction_cases import ExtractionCaseSet
from communitybridge.models.replan import ReplanScenarioSet
from communitybridge.models.scenario import ScenarioSet

SCENARIOS_PATH = DATA_DIR / "scenarios" / "scenarios.yaml"
REPLAN_SCENARIOS_PATH = DATA_DIR / "scenarios" / "replan_scenarios.yaml"
EXTRACTION_HOLDOUT_PATH = DATA_DIR / "scenarios" / "extraction_holdout.yaml"

Model = TypeVar("Model", bound=BaseModel)


class ScenarioDataError(Exception):
    """A scenarios file is missing, unreadable or does not match the schema."""


def _load(path: Path, model: type[Model]) -> Model:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScenarioDataError(f"scenarios file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioDataError(f"scenarios file is not valid YAML: {path}: {exc}") from exc
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioDataError(f"scenarios file failed validation: {path}:\n{exc}") from exc


def load_scenarios(path: Path = SCENARIOS_PATH) -> ScenarioSet:
    return _load(path, ScenarioSet)


def load_replan_scenarios(path: Path = REPLAN_SCENARIOS_PATH) -> ReplanScenarioSet:
    return _load(path, ReplanScenarioSet)


def load_extraction_holdout(path: Path = EXTRACTION_HOLDOUT_PATH) -> ExtractionCaseSet:
    """The held-out goal-extraction cases."""
    return _load(path, ExtractionCaseSet)
