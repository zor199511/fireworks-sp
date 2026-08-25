import json
import logging

from .db import get_conn

log = logging.getLogger("fwsp.tracker")

HORIZONS = {"ret_5d": 5, "ret_10d": 10, "ret_20d": 20, "ret_60d": 60}


def _forward_return(bars: list[tuple], run_date: str, n: int,
                    use_next_open=True) -> float | None:
    """bars: [(date, open, close)] sorted asc, dates > run_date excluded base.
    Entry = next trading day's open after run_date (T+1 rule).
    Exit = close of Nth trading day after entry day."""
    future = [b for b in bars if b[0] > run_date]
    if len(future) < n + 1:
        return None
    entry = future[0][1] if use_next_open else future[0][2]
    if not entry or entry <= 0:
        return None
    exit_close = future[n][2]
    if not exit_close:
        return None
    return exit_close / entry - 1


def update_tracking() -> int:
    updated = 0
    with get_conn() as conn:
        recos = conn.execute(
            "SELECT rowid, run_date, code, price FROM recommendations "
            "WHERE ret_60d IS NULL OR ret_5d IS NULL "
            "OR ret_10d IS NULL OR ret_20d IS NULL").fetchall()
        cache: dict[str, list[tuple]] = {}
        for rowid, run_date, code, _price in recos:
            if code not in cache:
                rows = conn.execute(
                    "SELECT date,open,close FROM daily WHERE code=? "
                    "ORDER BY date", (code,)).fetchall()
                cache[code] = rows
            sets, vals = [], []
            for col, n in HORIZONS.items():
                cur = conn.execute(
                    f"SELECT {col} FROM recommendations WHERE rowid=?",
                    (rowid,)).fetchone()[0]
                if cur is None:
                    r = _forward_return(cache[code], str(run_date), n)
                    if r is not None:
                        sets.append(f"{col}=?")
                        vals.append(round(r * 100, 2))
            if sets:
                vals.append(rowid)
                conn.execute(
                    f"UPDATE recommendations SET {','.join(sets)} "
                    "WHERE rowid=?", vals)
                updated += 1
        conn.commit()
    log.info("tracking updated rows: %d", updated)
    return updated


def summary_stats() -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*), AVG(ret_5d), AVG(ret_10d), AVG(ret_20d), "
            "SUM(CASE WHEN ret_5d>0 THEN 1 ELSE 0 END),"
            "SUM(CASE WHEN ret_5d IS NOT NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN ret_10d>0 THEN 1 ELSE 0 END),"
            "SUM(CASE WHEN ret_10d IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM recommendations").fetchone()
    total, avg5, avg10, avg20, win5_n, tot5, win10_n, tot10 = rows
    return {
        "total_recommendations": total or 0,
        "avg_ret_5d": avg5, "avg_ret_10d": avg10, "avg_ret_20d": avg20,
        "win_rate_5d": (win5_n / tot5) if tot5 else None,
        "win_rate_10d": (win10_n / tot10) if tot10 else None,
        "tracked_5d": tot5 or 0,
    }
