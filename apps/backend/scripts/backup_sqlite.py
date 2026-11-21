#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.services.backup_service import backup_database


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Create a consistent SQLite backup with WAL support."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.database_path,
        help="SQLite database path (default: FILMFINDER_DB_PATH).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.database_path.parent / "backups",
        help="Backup directory.",
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=14,
        help="Number of backups to keep (0 disables pruning).",
    )
    args = parser.parse_args()
    destination = backup_database(
        args.source.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.retention,
    )
    print(destination)


if __name__ == "__main__":
    main()
