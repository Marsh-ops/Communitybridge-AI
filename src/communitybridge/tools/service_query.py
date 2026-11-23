"""Service Discovery tool: read-only queries on the seeded SQLite database.

This is the only tool the Service Discovery Agent may call. It
finds the services that count as *nearest* for a specialty (resident, visiting or telehealth,
PATS §6.4.6) and returns each with its distance, its `source` and its `last_verified` date, so
the Verifier can later block a stale or unsourced claim.

Why it is built this way (the design approach: language models interpret and explain; deterministic
tools decide):
- **Read-only.** The database is opened with SQLite's `mode=ro` flag, so SQLite itself refuses any
  write, whatever the code or an agent asks for. The agent cannot change the dataset.
- **Validated.** Every row goes through the same Pydantic models as the dataset files, so the
  agent only ever sees well-formed records. A bad row raises `ServiceQueryError`.
- **It never decides eligibility.** "Nearest" uses the same definition as the rules engine
  (`candidate_services`, `rank_by_distance`), but whether the client is *eligible* is decided only
  by the rules engine. A search result says which services count as nearest, not
  whether the client may use them.
- **Same data as the engine.** `towns()`, `services()` and `distance_table()` give the rules
  engine exactly what the agents read, so `load_engine_context(query)` can run from this database.
"""

import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pydantic import ValidationError

from communitybridge.config import DB_PATH
from communitybridge.models.common import Citation
from communitybridge.models.dataset import Distance, ServiceRecord, Town
from communitybridge.models.discovery import ServiceMatch, ServiceSearch
from communitybridge.models.rules import RuleSet
from communitybridge.tools.distance import DistanceTable
from communitybridge.tools.nearest_service import (
    NEAREST_RULE,
    candidate_services,
    rank_by_distance,
)
from communitybridge.tools.rule_access import find_rule, norm, text_list

_SEED_HINT = "Build it with: python scripts/seed_db.py"


class ServiceQueryError(Exception):
    """The database is missing, unreadable, closed or holds an invalid record, or the query is."""


class UnknownTownError(ServiceQueryError):
    """The client's home town is not in the dataset (a gap in the data, not a broken database).

    Its own type so the Service Discovery Agent can tell staff *why* it stopped without copying
    the exception text into the trace: that text holds the town the user typed.
    """


class ServiceQuery:
    """A read-only view of the service dataset in SQLite. Use as a context manager."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        # Belt and braces: `open()` already uses mode=ro, and this keeps a connection passed in
        # by someone else from writing too.
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        self._conn = conn

    @classmethod
    def open(cls, db_path: Path = DB_PATH) -> Self:
        path = Path(db_path)
        if not path.is_file():
            raise ServiceQueryError(f"database not found: {path}. {_SEED_HINT}")
        try:
            # mode=ro in a file: URI is what makes SQLite refuse every write.
            conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            # SQLite connects lazily, so read once now: a file that is not a database fails here
            # with a clear message, not on the first question an agent asks.
            conn.execute("SELECT name FROM sqlite_master LIMIT 1")
            return cls(conn)
        except sqlite3.Error as exc:
            raise ServiceQueryError(f"cannot open database {path}: {exc}") from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _rows(self, sql: str) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(sql).fetchall()
        except sqlite3.Error as exc:
            hint = f" {_SEED_HINT}" if "no such table" in str(exc) else ""
            raise ServiceQueryError(f"database error: {exc}.{hint}") from exc

    # --- the dataset, as validated models ---------------------------------------------------

    def towns(self) -> list[Town]:
        towns = []
        for row in self._rows(
            "SELECT town_id, name, region, apy_lands, synthetic FROM town ORDER BY town_id"
        ):
            towns.append(
                _validated(
                    Town,
                    "town",
                    row["town_id"],
                    town_id=row["town_id"],
                    name=row["name"],
                    region=row["region"],
                    apy_lands=bool(row["apy_lands"]),
                    synthetic=bool(row["synthetic"]),
                )
            )
        return towns

    def services(self) -> list[ServiceRecord]:
        # Visiting specialists' dates live in their own table (§6.4.6 counts visiting services).
        visits: dict[str, list[date]] = defaultdict(list)
        for row in self._rows(
            "SELECT service_id, visit_date FROM visit_schedule ORDER BY visit_date"
        ):
            visits[row["service_id"]].append(date.fromisoformat(row["visit_date"]))
        services = []
        for row in self._rows(
            "SELECT service_id, name, specialty, town_id, mode, source, last_verified, synthetic "
            "FROM service ORDER BY service_id"
        ):
            services.append(
                _validated(
                    ServiceRecord,
                    "service",
                    row["service_id"],
                    service_id=row["service_id"],
                    name=row["name"],
                    specialty=row["specialty"],
                    town=row["town_id"],
                    mode=row["mode"],
                    source=row["source"],
                    last_verified=row["last_verified"],
                    visit_dates=visits.get(row["service_id"], []),
                    synthetic=bool(row["synthetic"]),
                )
            )
        return services

    def distance_table(self) -> DistanceTable:
        distances = []
        for row in self._rows(
            "SELECT from_town_id, to_town_id, road_km, source, synthetic FROM distance"
        ):
            distances.append(
                _validated(
                    Distance,
                    "distance",
                    f"{row['from_town_id']} to {row['to_town_id']}",
                    from_town_id=row["from_town_id"],
                    to_town_id=row["to_town_id"],
                    road_km=row["road_km"],
                    source=row["source"],
                    synthetic=bool(row["synthetic"]),
                )
            )
        return DistanceTable(distances)

    # --- the question the Service Discovery Agent asks ------------------------------------------

    def nearest(self, origin_town_name: str, specialty: str, rules: RuleSet) -> ServiceSearch:
        """The services that count as nearest for `specialty`, from the client's home town.

        Returns an empty result if no service offers the specialty. Raises `ServiceQueryError` for
        an unknown town, and `DistanceNotFoundError` if the distance table lacks a needed pair.
        """
        wanted = specialty.strip()
        if not wanted:
            raise ServiceQueryError("the specialty is empty")
        origin = {norm(town.name): town for town in self.towns()}.get(norm(origin_town_name))
        if origin is None:
            raise UnknownTownError(f"origin town {origin_town_name!r} is not in the database")

        counted_modes = text_list(find_rule(rules, NEAREST_RULE), "counted_service_modes")
        candidates = candidate_services(self.services(), wanted, counted_modes)
        ranked = rank_by_distance(origin.town_id, candidates, self.distance_table())
        matches = [
            ServiceMatch(
                service=service,
                distance_km=distance,
                # The Verifier checks this citation's date for freshness.
                citation=Citation(
                    source=service.source,
                    section_or_record=f"service {service.service_id}",
                    last_verified=service.last_verified,
                ),
            )
            for service, distance in ranked
        ]
        nearest = [m for m in matches if matches and m.distance_km == matches[0].distance_km]
        others = matches[len(nearest) :]
        return ServiceSearch(
            origin_town_id=origin.town_id, specialty=wanted, nearest=nearest, others=others
        )


def _validated(model: type[Any], table: str, key: str, **fields: Any) -> Any:
    """Build a validated model from a database row, or name the row that is invalid."""
    try:
        return model(**fields)
    except ValidationError as exc:
        raise ServiceQueryError(f"{table} {key} in the database is invalid: {exc}") from exc
