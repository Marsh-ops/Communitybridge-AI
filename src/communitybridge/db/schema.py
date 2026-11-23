"""SQLite schema for the synthetic, community-administered service dataset.

Tables follow the data model. Agents only ever read this database (the Service Discovery
Agent through `tools/service_query.py`). The CHECK constraints repeat the key model rules, so
the database itself refuses a non-synthetic record or an unknown service mode, even if a seed
file is wrong.

PATS thresholds are deliberately not stored here. They live in data/rules/pats_v26_4.yaml so a
guideline update only changes the YAML.
"""

import sqlite3

TABLES = ("town", "distance", "service", "visit_schedule", "accommodation")

SCHEMA = """
CREATE TABLE IF NOT EXISTS town (
    town_id    TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    region     TEXT NOT NULL,
    apy_lands  INTEGER NOT NULL CHECK (apy_lands IN (0, 1)),  -- PATS §6.1.1
    synthetic  INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE IF NOT EXISTS distance (
    from_town_id TEXT NOT NULL REFERENCES town (town_id),
    to_town_id   TEXT NOT NULL REFERENCES town (town_id),
    road_km      REAL NOT NULL CHECK (road_km >= 0),
    source       TEXT NOT NULL,
    synthetic    INTEGER NOT NULL CHECK (synthetic = 1),
    PRIMARY KEY (from_town_id, to_town_id)
);

CREATE TABLE IF NOT EXISTS service (
    service_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    specialty     TEXT NOT NULL,
    town_id       TEXT NOT NULL REFERENCES town (town_id),
    mode          TEXT NOT NULL CHECK (mode IN ('resident', 'visiting', 'telehealth')),
    source        TEXT NOT NULL,
    last_verified TEXT NOT NULL,  -- ISO date; checked for freshness by the Verifier
    synthetic     INTEGER NOT NULL CHECK (synthetic = 1)
);

CREATE TABLE IF NOT EXISTS visit_schedule (
    service_id TEXT NOT NULL REFERENCES service (service_id),
    visit_date TEXT NOT NULL,
    synthetic  INTEGER NOT NULL CHECK (synthetic = 1),
    PRIMARY KEY (service_id, visit_date)
);

CREATE TABLE IF NOT EXISTS accommodation (
    accommodation_id TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    town_id          TEXT NOT NULL REFERENCES town (town_id),
    commercial       INTEGER NOT NULL CHECK (commercial IN (0, 1)),  -- PATS §6.6.1
    source           TEXT NOT NULL,
    last_verified    TEXT NOT NULL,
    synthetic        INTEGER NOT NULL CHECK (synthetic = 1)
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist. Safe to run more than once."""
    conn.executescript(SCHEMA)
