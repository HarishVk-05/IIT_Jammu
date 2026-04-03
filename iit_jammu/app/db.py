from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text, event
from sqlalchemy.exc import OperationalError, DatabaseError
from langchain_community.utilities import SQLDatabase
from iit_jammu.config import SQLITE_PATH

logger = logging.getLogger(__name__)

def _make_engine():
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {SQLITE_PATH}. "
            "Run ingest/load_sqlite.py first."
        )
    
    engine = create_engine(
        f"sqlite:///{SQLITE_PATH}",
        connect_args = {"check_same_thread": False}
    )
    # enfore read-only at connection level
    @event.listens_for(engine, "connect")
    def _set_readonly(conn, _rec):
        conn.execute("PRAGMA query_only = ON")
    return engine

try:
    ENGINE = _make_engine()
except FileNotFoundError as exc:
    logger.warning("ENGINE not initialised: %s", exc)
    ENGINE = None

_BLOCKED_TOKENS = [
    "insert ", "update ", "delete ", "drop ",
    "alter ", "create ", "attach ", "pragma ",
    "--"
]

def _validate_sql(sql: str) -> None:
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are permitted.")
    for tok in _BLOCKED_TOKENS:
        if tok in lowered:
            raise ValueError(f"Unsafe SQL token detected: '{tok.strip()}")
        

def get_sql_db() -> SQLDatabase:
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {SQLITE_PATH}. "
            "Run ingest/load_sqlite.py first."
        )
    
    return SQLDatabase.from_uri(f"sqlite:///{SQLITE_PATH}")

def run_query(sql: str) -> pd.DataFrame:
    """
    Executes a validated SELECT query and return a DataFrame.
    """
    if ENGINE is None:
        raise RuntimeError(
            "Database is not available. Run ingest/load_sqlite.py first."
        )
    _validate_sql(sql)
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df
    except OperationalError as exc:
        logger.error("SQL operation error: %s | query: %s", exc, sql)
        raise ValueError(
            f"The generated query could not be executed: {exc}. "
            "Try rephrasing your question."
        ) from exc
    except DatabaseError as exc:
        logger.error("Database error: %s | query: %s", exc, sql)
        raise ValueError(
            f"A database error occurred: {exc}."
        ) from exc
    
def get_database_stats() -> dict | None:
    """
    Return basic coverage stats shown in the sidebar.
    Returns None if the DB is unavailable or tables are missing.
    """

    if ENGINE is None:
        return None
    
    sql = """
    SELECT
        COUNT(*)                       AS n_profiles,
        COUNT(DISTINCT float_id)       AS n_floats,
        MIN (timestamp)                AS date_min,
        MAX (timestamp)                AS date_max,
        ROUND(MIN(latitude), 2)        AS lat_min,
        ROUND(MAX(latitude), 2)        AS lat_max,
        ROUND(MIN(longitude), 2)       AS lon_min,
        ROUND(MAX(longitude), 2)       AS lon_max
    FROM profiles_metadata
    """

    try:
        with ENGINE.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        if row is None:
            return None
        return dict(row._mapping)
    except Exception as exc:
        logger.warning("Could not load dataset stats: %s", exc)
        return None