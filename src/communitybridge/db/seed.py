"""Load the synthetic dataset files, validate every record and build the SQLite database.

Used by scripts/seed_db.py and the tests. The database is always rebuilt from scratch from the
files in data/, so the files are the single source of truth (the ACCHO would maintain them in a
real deployment), and seeding twice always gives the same result.

Every record goes through a Pydantic model first, so a non-synthetic or malformed row stops
the seed with a `SeedDataError` instead of entering the database.
"""

import csv
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from communitybridge.config import DATA_DIR
from communitybridge.db.schema import create_schema
from communitybridge.models.dataset import Accommodation, Distance, ServiceRecord, Town

TOWNS_CSV = DATA_DIR / "geo" / "towns.csv"
DISTANCES_CSV = DATA_DIR / "geo" / "towns_distances.csv"
SERVICES_YAML = DATA_DIR / "services" / "services.yaml"
ACCOMMODATION_YAML = DATA_DIR / "services" / "accommodation.yaml"

Model = TypeVar("Model", bound=BaseModel)


class SeedDataError(Exception):
    """A dataset file is missing, malformed, or contains an invalid record."""


def _parse_bool(value: str, field: str, path: Path) -> bool:
    # CSV has no boolean type, so only the exact strings "true" and "false" are accepted.
    if value == "true":
        return True
    if value == "false":
        return False
    raise SeedDataError(f"{path}: {field} must be 'true' or 'false', got {value!r}")


def _validate(model: type[Model], rows: list[dict[str, Any]], path: Path) -> list[Model]:
    records = []
    for number, row in enumerate(rows, start=1):
        try:
            records.append(model.model_validate(row))
        except ValidationError as exc:
            raise SeedDataError(f"{path}: record {number} is invalid:\n{exc}") from exc
    return records


def _read_csv(path: Path, bool_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            rows: list[dict[str, Any]] = list(csv.DictReader(f))
    except FileNotFoundError as exc:
        raise SeedDataError(f"dataset file not found: {path}") from exc
    for row in rows:
        for field in bool_fields:
            row[field] = _parse_bool(row[field], field, path)
    return rows


def _read_yaml_list(path: Path, key: str) -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SeedDataError(f"dataset file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SeedDataError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise SeedDataError(f"{path}: expected a top-level list named '{key}'")
    return data[key]


def load_towns(path: Path = TOWNS_CSV) -> list[Town]:
    return _validate(Town, _read_csv(path, ("apy_lands", "synthetic")), path)


def load_distances(path: Path = DISTANCES_CSV) -> list[Distance]:
    return _validate(Distance, _read_csv(path, ("synthetic",)), path)


def load_services(path: Path = SERVICES_YAML) -> list[ServiceRecord]:
    return _validate(ServiceRecord, _read_yaml_list(path, "services"), path)


def load_accommodation(path: Path = ACCOMMODATION_YAML) -> list[Accommodation]:
    return _validate(Accommodation, _read_yaml_list(path, "accommodation"), path)


def load_non_specialist_categories(path: Path = SERVICES_YAML) -> dict[str, str]:
    """Specialty -> kind of health professional, for specialties that are not medical specialists.

    The engine uses it for PATS §6.4.4 (ineligible health professional services). It lives in the
    services file because it classifies the services in it.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SeedDataError(f"dataset file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SeedDataError(f"{path} is not valid YAML: {exc}") from exc
    categories = data.get("non_specialist_categories") if isinstance(data, dict) else None
    if not isinstance(categories, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in categories.items()
    ):
        raise SeedDataError(f"{path}: expected 'non_specialist_categories' as specialty: category")
    return categories


def _insert(
    conn: sqlite3.Connection, table: str, records: list[Any], row: Callable[[Any], tuple]
) -> None:
    rows = [row(record) for record in records]
    if rows:
        placeholders = ", ".join("?" * len(rows[0]))
        conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)


def seed_database(db_path: Path, data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Rebuild the database at `db_path` from the dataset files. Returns row counts per table."""
    towns = load_towns(data_dir / "geo" / "towns.csv")
    distances = load_distances(data_dir / "geo" / "towns_distances.csv")
    services = load_services(data_dir / "services" / "services.yaml")
    accommodation = load_accommodation(data_dir / "services" / "accommodation.yaml")

    db_path.unlink(missing_ok=True)  # always rebuild, never append
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        create_schema(conn)
        _insert(conn, "town", towns,
                lambda t: (t.town_id, t.name, t.region, int(t.apy_lands), 1))  # fmt: skip
        _insert(conn, "distance", distances,
                lambda d: (d.from_town_id, d.to_town_id, d.road_km, d.source, 1))  # fmt: skip
        _insert(conn, "service", services, lambda s: (
            s.service_id, s.name, s.specialty, s.town, s.mode, s.source,
            s.last_verified.isoformat(), 1,
        ))  # fmt: skip
        visits = [(s.service_id, d) for s in services for d in s.visit_dates]
        _insert(conn, "visit_schedule", visits, lambda v: (v[0], v[1].isoformat(), 1))
        _insert(conn, "accommodation", accommodation, lambda a: (
            a.accommodation_id, a.name, a.town, int(a.commercial), a.source,
            a.last_verified.isoformat(), 1,
        ))  # fmt: skip
        conn.commit()
        tables = ("town", "distance", "service", "visit_schedule", "accommodation")
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    except sqlite3.IntegrityError as exc:
        raise SeedDataError(f"database rejected a record: {exc}") from exc
    finally:
        conn.close()
