import json
import logging
import sqlite3
from contextlib import contextmanager

from . import config

log = logging.getLogger("fwsp.db")

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
CREATE TABLE IF NOT EXISTS daily_qfq (
    code TEXT,
    date TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_qfq_date ON daily_qfq(date);
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
CREATE TABLE IF NOT EXISTS index_cons (
    index_code TEXT,
    code TEXT,
    name TEXT,
    industry TEXT,
    snapshot_date TEXT,
    PRIMARY KEY (index_code, code, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_index_cons_code ON index_cons(code);
CREATE INDEX IF NOT EXISTS idx_index_cons_date ON index_cons(snapshot_date);
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
CREATE TABLE IF NOT EXISTS evidence_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    claim TEXT NOT NULL,
    source TEXT,
    as_of TEXT,
    quote TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence_log(run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_stage ON evidence_log(stage);
CREATE TABLE IF NOT EXISTS calc_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    output_name TEXT NOT NULL,
    output_value TEXT,
    func_name TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calc_run ON calc_log(run_id);
CREATE TABLE IF NOT EXISTS data_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    field TEXT NOT NULL,
    source_a TEXT,
    value_a TEXT,
    source_b TEXT,
    value_b TEXT,
    resolution TEXT,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS run_manifest (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    params_json TEXT,
    summary_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS backtest_results (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    params_json TEXT,
    total_return REAL,
    cagr REAL,
    max_drawdown REAL,
    sharpe REAL,
    n_trades INTEGER,
    win_rate REAL,
    avg_trade_ret REAL,
    bench_return REAL,
    equity_json TEXT,
    trades_json TEXT,
    summary_json TEXT
);
"""


@contextmanager
def get_conn(db_path=None):
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")  # 防并发 timer/查询重叠时 database is locked
    # 子代理 2 轮 R2-并发3: wal_autocheckpoint 调优, 避免长事务 checkpoint 阻塞
    conn.execute("PRAGMA wal_autocheckpoint=10000")
    # cache_size 默认 ~2MB 太小, 大查询频繁 -wal 几 GB 占盘
    conn.execute("PRAGMA cache_size=-64000")  # 64MB
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# 子代理 2 轮 R2-运维3: 用 PRAGMA user_version 显式管理 schema 版本
# 起始版本=2 (在原 schema + 上一轮索引迁移之上)
# 升级时: 在 _MIGRATIONS 列表中加 (version, label, fn) 元组, 升级 fn 必须幂等
CURRENT_SCHEMA_VERSION = 3
_MIGRATIONS = [
    (1, "initial_schema", lambda _c: None),  # 占位, 旧库已有
    (2, "add_indexes_and_fin_q_as_of", lambda _c: None),  # 已通过 ALTER 隐式迁移
    # 后续加: (3, "xxx", lambda c: c.execute("...")),
]


def _migrate(conn):
    """从 PRAGMA user_version 升级到 CURRENT_SCHEMA_VERSION, 逐个跑 _MIGRATIONS."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for ver, label, fn in _MIGRATIONS:
        if current >= ver:
            continue
        log.info("schema migrate: %s → v%d", label, ver)
        fn(conn)
        conn.execute(f"PRAGMA user_version = {ver}")
        conn.commit()
    if current < CURRENT_SCHEMA_VERSION:
        # 即便 _MIGRATIONS 列表与 CURRENT_SCHEMA_VERSION 不一致, 仍 bump 到底
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
        conn.commit()
    set_meta(conn, "schema_version", str(CURRENT_SCHEMA_VERSION))


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
    # 旧库迁移：fin_q 增加 as_of（财报披露日，point-in-time 质量面板用）
    try:
        conn.execute("ALTER TABLE fin_q ADD COLUMN as_of TEXT")
    except sqlite3.OperationalError:
        pass
    # 子代理 2 轮 R2-运维3: 显式 schema 版本号 + 迁移日志
    _migrate(conn)


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
    子代理 2 轮 R2-并发1: 三步非事务原子, 跨进程并发写会留不一致.
    修: 跨进程 fcntl 文件锁(其他进程持锁时此调用抛 RuntimeError 跳过).
    sqlite 自身在 with get_conn 退出时 commit, 这里三步在同一事务里.
    """
    from .lock import file_lock, LOCK_ACTIVE
    import datetime
    with file_lock(LOCK_ACTIVE, op=f"set_active_factors({source})") as got:
        if not got:
            raise RuntimeError(
                f"set_active_factors({source}) 锁被其他进程持有, 已跳过本次写入")
        run_at = run_at or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_active_set(conn, run_at, factors, oos, source)
        set_meta(conn, "active_factors", json.dumps(list(factors), ensure_ascii=False))
        _sync_selected(conn, factors)
    return run_at


# 子代理 1 critical-1: f-string 拼表名是反模式(实际有白名单常量保护),
# 这里统一一个白名单校验函数, 任何拼表名 SQL 都先过这一关.
_ALLOWED_TABLES = frozenset({
    "stock_list", "spot", "fin_q", "daily", "daily_qfq", "index_daily",
    "meta", "recommendations", "active_sets", "factor_library",
    "factor_eval", "evolution_log", "index_cons",
})


def validate_table_name(name: str) -> str:
    """白名单校验表名, 用于 f-string 拼 SQL 前. 子代理 1 critical #1 防御."""
    if name not in _ALLOWED_TABLES:
        raise ValueError(
            f"refused: table '{name}' not in whitelist "
            f"(allowed: {sorted(_ALLOWED_TABLES)})")
    return name


def load_daily(conn, code: str) -> "list[tuple]":
    return conn.execute(
        "SELECT date,open,high,low,close,volume,amount FROM daily WHERE code=? ORDER BY date",
        (code,),
    ).fetchall()


def last_daily_dates(conn) -> dict[str, str]:
    rows = conn.execute("SELECT code,MAX(date) FROM daily GROUP BY code").fetchall()
    return dict(rows)
