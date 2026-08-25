import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_list (
    code TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    is_st INTEGER DEFAULT 0,
    industry TEXT,
    updated TEXT
);
CREATE TABLE IF NOT EXISTS spot (
    code TEXT PRIMARY KEY,
    price REAL, pct_chg REAL, volume REAL, amount REAL,
    turnover REAL, vol_ratio REAL,
    pe_dyn REAL, pb REAL, total_mv REAL, circ_mv REAL, chg_60d REAL,
    updated TEXT
);
CREATE TABLE IF NOT EXISTS fin_q (
    code TEXT,
    period TEXT,
    eps REAL, roe REAL, gross_margin REAL,
    profit_yoy REAL, debt_ratio REAL,
    updated TEXT,
    PRIMARY KEY (code, period)
);
CREATE TABLE IF NOT EXISTS daily (
    code TEXT,
    date TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date);
CREATE TABLE IF NOT EXISTS index_daily (
    code TEXT,
    date TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    run_date TEXT, rank INTEGER, code TEXT,
    name TEXT, industry TEXT,
    score REAL, price REAL,
    reasons TEXT, metrics TEXT,
    ret_5d REAL, ret_10d REAL, ret_20d REAL, ret_60d REAL,
    PRIMARY KEY (run_date, code)
);
"""


@contextmanager
def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn):
    conn.executescript(SCHEMA)


def upsert_rows(conn, table: str, cols: list[str], rows: list[tuple]):
    if not rows:
        return 0
    placeholders = ",".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, rows)
    return len(rows)


def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn, key: str, value: str):
    conn.execute(
        "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)", (key, str(value))
    )


def load_daily(conn, code: str) -> "list[tuple]":
    return conn.execute(
        "SELECT date,open,high,low,close,volume,amount FROM daily WHERE code=? ORDER BY date",
        (code,),
    ).fetchall()


def last_daily_dates(conn) -> dict[str, str]:
    rows = conn.execute("SELECT code,MAX(date) FROM daily GROUP BY code").fetchall()
    return dict(rows)
