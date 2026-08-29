import json
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
    factor_set_id TEXT,
    PRIMARY KEY (run_date, code)
);
CREATE TABLE IF NOT EXISTS active_sets (
    run_at TEXT PRIMARY KEY,
    factors_json TEXT NOT NULL,
    oos REAL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS factor_library (
    code TEXT PRIMARY KEY,
    category TEXT,
    expr TEXT,
    params_json TEXT,
    desc TEXT,
    source TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS factor_eval (
    code TEXT,
    run_at TEXT,
    is_icir REAL,
    oos_icir REAL,
    oos_is_ratio REAL,
    stability REAL,
    turnover REAL,
    net_ir REAL,
    selected INTEGER,
    PRIMARY KEY (code, run_at)
);
CREATE TABLE IF NOT EXISTS evolution_log (
    run_at TEXT PRIMARY KEY,
    selected_json TEXT,
    old_oos REAL,
    new_oos REAL,
    promoted INTEGER,
    notes TEXT
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
    # 旧库迁移：factor_eval 后续增加了 turnover / net_ir 列，CREATE IF NOT EXISTS
    # 不会给已存在的表补列，这里幂等 ALTER（新库已含，捕获忽略）。
    for col in ("turnover REAL", "net_ir REAL"):
        try:
            conn.execute(f"ALTER TABLE factor_eval ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    # 旧库迁移：recommendations 增加 factor_set_id（因子集血缘）
    try:
        conn.execute("ALTER TABLE recommendations ADD COLUMN factor_set_id TEXT")
    except sqlite3.OperationalError:
        pass


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


# --------------------------------------------------------------------------
# active_sets：活跃因子集的单一真相源 + 单一写入者
# --------------------------------------------------------------------------

def write_active_set(conn, run_at: str, factors: list[str], oos=None,
                    source: str = "auto_evolve") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO active_sets (run_at, factors_json, oos, source) "
        "VALUES (?,?,?,?)",
        (run_at, json.dumps(list(factors), ensure_ascii=False), oos, source))


def get_active_set(conn, source: str | None = None) -> dict | None:
    sql = "SELECT run_at, factors_json, oos, source FROM active_sets"
    args: tuple = ()
    if source:
        sql += " WHERE source=? "
        args = (source,)
    sql += " ORDER BY run_at DESC LIMIT 1"
    row = conn.execute(sql, args).fetchone()
    if not row:
        # 过渡期回退：旧库可能只有 meta.active_factors（无 active_sets 表数据）
        if source in (None, "auto_evolve"):
            raw = get_meta(conn, "active_factors")
            if raw:
                try:
                    factors = json.loads(raw)
                except (TypeError, ValueError):
                    factors = []
                return {"run_at": None, "factors": factors,
                        "oos": None, "source": "auto_evolve"}
        return None
    return {"run_at": row[0], "factors": json.loads(row[1]),
            "oos": row[2], "source": row[3]}


def _sync_selected(conn, factors: list[str]) -> None:
    """factor_eval.selected 改为 active_sets 的派生字段：仅当前活跃因子
    的最新评估行标 1，其余归 0，从根上消除状态漂移。"""
    conn.execute("UPDATE factor_eval SET selected=0")
    if factors:
        ph = ",".join("?" * len(factors))
        conn.execute(
            f"UPDATE factor_eval SET selected=1 WHERE (code, run_at) IN ("
            f"SELECT code, MAX(run_at) FROM factor_eval WHERE code IN ({ph}) "
            f"GROUP BY code)", list(factors))


def set_active_factors(conn, factors: list[str], run_at: str | None = None,
                      oos=None, source: str = "auto_evolve") -> str:
    """唯一写入者：写 active_sets + 镜像 meta.active_factors + 同步 selected。

    自动进化与衰减降级都只走这里，保证 meta / factor_eval / 进化日志三处一致。
    """
    import datetime
    run_at = run_at or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_active_set(conn, run_at, factors, oos, source)
    set_meta(conn, "active_factors", json.dumps(list(factors), ensure_ascii=False))
    _sync_selected(conn, factors)
    return run_at


def load_daily(conn, code: str) -> "list[tuple]":
    return conn.execute(
        "SELECT date,open,high,low,close,volume,amount FROM daily WHERE code=? ORDER BY date",
        (code,),
    ).fetchall()


def last_daily_dates(conn) -> dict[str, str]:
    rows = conn.execute("SELECT code,MAX(date) FROM daily GROUP BY code").fetchall()
    return dict(rows)
