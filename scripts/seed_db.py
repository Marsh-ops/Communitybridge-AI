"""Build the SQLite database from the synthetic dataset files.

Usage (from the repo root, with the venv active):
    python scripts/seed_db.py               # writes data/communitybridge.db
    python scripts/seed_db.py --db other.db
"""

import argparse
import sys
from pathlib import Path

from communitybridge.config import DB_PATH
from communitybridge.db.seed import SeedDataError, seed_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the CommunityBridge SQLite database.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="database file to (re)build")
    args = parser.parse_args()
    try:
        counts = seed_database(args.db)
    except SeedDataError as exc:
        print(f"Seed failed: {exc}", file=sys.stderr)
        return 1
    print(f"Seeded {args.db} (all records synthetic):")
    for table, n in counts.items():
        print(f"  {table:<15} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
