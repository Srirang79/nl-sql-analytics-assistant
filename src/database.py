"""Safe query execution: read-only connection, row cap and a hard timeout."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from . import config


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open the database read-only. Even if validation were bypassed, writes would fail at the SQLite level."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn


class QueryTimeout(Exception):
    pass


def run_query(conn: sqlite3.Connection, sql: str, max_rows: int = config.MAX_ROWS_RETURNED,
              timeout_s: float = config.QUERY_TIMEOUT_S) -> tuple[pd.DataFrame, bool, float]:
    """Execute ``sql``. Returns (dataframe, truncated, seconds). Aborts after ``timeout_s`` seconds."""
    start = time.monotonic()

    def _abort():
        return 1 if time.monotonic() - start > timeout_s else 0

    conn.set_progress_handler(_abort, 20_000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            raise QueryTimeout(f"The query ran longer than {timeout_s:.0f}s and was stopped. Make it more selective.") from e
        raise
    finally:
        conn.set_progress_handler(None, 0)
    truncated = len(rows) > max_rows
    df = pd.DataFrame(rows[:max_rows], columns=cols)
    return df, truncated, time.monotonic() - start


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
