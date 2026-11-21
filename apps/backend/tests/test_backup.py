from __future__ import annotations

import sqlite3

from app.services.backup_service import backup_database


def test_sqlite_backup_copies_data_and_prunes(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO sample (value) VALUES ('kept')")
        connection.commit()

    output_dir = tmp_path / "backups"
    first = backup_database(source, output_dir, retention=1)
    second = backup_database(source, output_dir, retention=1)

    with sqlite3.connect(second) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == (
            "kept",
        )
    assert first.exists() is False
    assert second.exists()
