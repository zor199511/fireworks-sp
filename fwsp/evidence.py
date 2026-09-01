"""证据链与计算溯源模块

借鉴 Vibe-Research 的 evidence.json + calculations.json 机制，
为 fireworks-sp 的因子计算、回测、选股各阶段提供可追溯性。
"""
import json
import logging
import sqlite3
from datetime import datetime
from functools import wraps
from typing import Any, Callable

log = logging.getLogger("fwsp.evidence")

# ----------------------------------------------------------------
# Schema
# ----------------------------------------------------------------

EVIDENCE_SCHEMA = """
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


def ensure_evidence_schema(conn: sqlite3.Connection):
    """确保证据链表存在（幂等）"""
    conn.executescript(EVIDENCE_SCHEMA)


# ----------------------------------------------------------------
# Evidence Log
# ----------------------------------------------------------------

class EvidenceLogger:
    """证据链记录器，绑定到一次运行（run_id）"""

    def __init__(self, conn: sqlite3.Connection = None, run_id: str = None,
                 run_type: str = None, db_path: str = None):
        self._own_conn = conn is None
        if self._own_conn:
            from . import config
            self.conn = sqlite3.connect(db_path or config.DB_PATH)
            self.conn.execute("PRAGMA journal_mode=WAL")
        else:
            self.conn = conn
        self.run_id = run_id
        self.run_type = run_type
        self._started = datetime.now().isoformat()
        ensure_evidence_schema(self.conn)
        # 写入 run_manifest
        self.conn.execute(
            "INSERT OR REPLACE INTO run_manifest (run_id, run_type, started_at, status) VALUES (?,?,?,?)",
            (run_id, run_type, self._started, "running")
        )
        self.conn.commit()

    def log(self, stage: str, claim: str, source: str = None,
            as_of: str = None, quote: str = None):
        """记录一条证据"""
        self.conn.execute(
            "INSERT INTO evidence_log (run_id, stage, claim, source, as_of, quote, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.run_id, stage, claim, source, as_of, quote, datetime.now().isoformat())
        )

    def finish(self, status: str = "complete", summary: dict = None, error: str = None):
        """标记运行完成"""
        self.conn.execute(
            "UPDATE run_manifest SET finished_at=?, status=?, summary_json=?, error=? WHERE run_id=?",
            (datetime.now().isoformat(), status, json.dumps(summary, ensure_ascii=False) if summary else None,
             error, self.run_id)
        )
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.finish(status="failed", error=str(exc_val))
        else:
            self.finish(status="complete")
        if self._own_conn:
            self.conn.close()


# ----------------------------------------------------------------
# Calc Log (计算溯源)
# ----------------------------------------------------------------

class CalcTracer:
    """计算溯源记录器"""

    def __init__(self, conn: sqlite3.Connection, run_id: str):
        self.conn = conn
        self.run_id = run_id
        ensure_evidence_schema(conn)

    def log_calc(self, output_name: str, output_value: Any,
                 func_name: str, inputs: dict):
        """记录一次计算的输入/输出"""
        self.conn.execute(
            "INSERT INTO calc_log (run_id, output_name, output_value, func_name, inputs_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (self.run_id, output_name, str(output_value)[:500],
             func_name, json.dumps(inputs, ensure_ascii=False, default=str),
             datetime.now().isoformat())
        )


def trace_calc(output_name: str = None):
    """装饰器：自动记录函数的输入/输出到 calc_log

    用法:
        @trace_calc("factor_weights")
        def compute_weights(ic_series):
            ...
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            # 如果第一个参数是 CalcTracer，自动记录
            if args and isinstance(args[0], CalcTracer):
                tracer = args[0]
                name = output_name or fn.__name__
                inputs = {}
                for i, a in enumerate(args[1:], 1):
                    inputs[f"arg{i}"] = str(a)[:200] if not isinstance(a, (int, float, str, bool)) else a
                for k, v in kwargs.items():
                    inputs[k] = str(v)[:200] if not isinstance(v, (int, float, str, bool)) else v
                tracer.log_calc(name, result, fn.__name__, inputs)
            return result
        return wrapper
    return decorator


# ----------------------------------------------------------------
# Conflict Detection
# ----------------------------------------------------------------

def log_conflict(conn: sqlite3.Connection, field: str,
                 source_a: str, value_a: str,
                 source_b: str, value_b: str,
                 resolution: str = "manual"):
    """记录数据冲突"""
    ensure_evidence_schema(conn)
    conn.execute(
        "INSERT INTO data_conflicts (field, source_a, value_a, source_b, value_b, resolution, resolved_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (field, source_a, str(value_a), source_b, str(value_b),
         resolution, datetime.now().isoformat())
    )
    conn.commit()
