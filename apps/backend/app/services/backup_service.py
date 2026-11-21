from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def backup_database(
    source: Path,
    output_dir: Path,
    retention: int = 14,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"SQLite database not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    destination = output_dir / f"filmfinder-{timestamp}.db"
    with sqlite3.connect(str(source)) as source_connection:
        with sqlite3.connect(str(destination)) as destination_connection:
            source_connection.backup(destination_connection)
    _prune(output_dir, retention)
    return destination


def _prune(output_dir: Path, retention: int) -> None:
    if retention <= 0:
        return
    backups = sorted(output_dir.glob("filmfinder-*.db"))
    for old_backup in backups[:-retention]:
        old_backup.unlink(missing_ok=True)
