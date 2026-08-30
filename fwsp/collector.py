import json
import logging
import random
import re
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests

# baostock still uses removed DataFrame.append (pandas >= 2.0); shim it.
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, *args, **kwargs):
        import warnings
        warnings.warn("DataFrame.append shim used", stacklevel=2)
        if isinstance(other, dict):
            other = pd.DataFrame([other])
        elif isinstance(other, list):
            other = pd.DataFrame(other)
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append

from .config import SPOT_COLS, YJBB_COLS
from .db import last_daily_dates, upsert_rows

log = logging.getLogger("fwsp.collector")
TZ = ZoneInfo("Asia/Shanghai")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def today_cn() -> date:
    return datetime.now(TZ).date()


def robust(fn, *args, tries=6, max_wait=45.0, **kwargs):
    last_exc = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            wait = min(2**i + random.random(), max_wait)
            log.warning("%s attempt %d/%d failed: %s, retry in %.1fs",
                        getattr(fn, "__name__", fn), i + 1, tries,
                        type(e).__name__, wait)
            time.sleep(wait)
    raise RuntimeError(f"{fn} failed after {tries} attempts") from last_exc


def _f(v):
    try:
        v = float(v)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- spot (EM)

def _fetch_spot_akshare():
    df = robust(ak.stock_zh_a_spot_em)
    df = df.rename(columns=SPOT_COLS)
    out = []
    for r in df.itertuples(index=False):
        code = str(r.code).zfill(6)
        name = str(getattr(r, "name", ""))
        out.append({
            "code": code, "name": name,
            "price": _f(getattr(r, "price", None)),
            "pct_chg": _f(getattr(r, "pct_chg", None)),
            "open": _f(getattr(r, "open", None)),
            "high": _f(getattr(r, "high", None)),
            "low": _f(getattr(r, "low", None)),
            "volume": (_f(getattr(r, "volume", None)) or 0) * 100.0,  # 手 -> 股
            "amount": _f(getattr(r, "amount", None)),       # 元
            "turnover": _f(getattr(r, "turnover", None)),
            "vol_ratio": _f(getattr(r, "vol_ratio", None)),
            "pe_dyn": _f(getattr(r, "pe_dyn", None)),
            "pb": _f(getattr(r, "pb", None)),
            "total_mv": _f(getattr(r, "total_mv", None)),   # 元
            "circ_mv": _f(getattr(r, "circ_mv", None)),     # 元
        })
    return out


# ----------------------------------------------------------- spot (Tencent)

_TENCENT_URL = "https://qt.gtimg.cn/q="


def _fetch_spot_tencent(codes: list[str]) -> list[dict]:
    out = []
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    for i in range(0, len(codes), 60):
        chunk = codes[i:i + 60]
        q = ",".join(
            ("sh" if c[0] == "6" else "sz") + c for c in chunk)
        resp = robust(sess.get, _TENCENT_URL + q, timeout=10)
        resp.encoding = "gbk"
        for m in re.finditer(r'v_(?:sh|sz)(\d{6})="([^"]*)"', resp.text):
            code, body = m.group(1), m.group(2).split("~")
            if len(body) < 47 or not body[3]:
                continue
            total_mv_yi = _f(body[45])
            circ_mv_yi = _f(body[44])
            out.append({
                "code": code, "name": body[1],
                "price": _f(body[3]), "pct_chg": _f(body[32]),
                "open": _f(body[5]), "high": _f(body[33]), "low": _f(body[34]),
                "volume": (_f(body[36]) or 0) * 100.0,          # 手 -> 股
                "amount": (_f(body[37]) or 0) * 1e4,            # 万 -> 元
                "turnover": _f(body[38]),
                "vol_ratio": None,
                "pe_dyn": _f(body[39]),                          # PE TTM
                "pb": _f(body[46]),
                "total_mv": total_mv_yi * 1e8 if total_mv_yi else None,
                "circ_mv": circ_mv_yi * 1e8 if circ_mv_yi else None,
            })
        time.sleep(random.uniform(0.3, 0.8))
    return out


def _fetch_stock_list_baostock(conn):
    """Last-resort universe builder when EM spot is throttled on first run."""
    import baostock as bs
    lg = bs.login()
    try:
        rs = bs.query_stock_basic()
        data = rs.get_data()
    finally:
        bs.logout()
    now = str(today_cn())
    rows = []
    for _, r in data.iterrows():
        bs_code = r["code"]
        if not bs_code.startswith(("sh.6", "sz.0", "sz.3")):
            continue
        if r.get("type") not in ("1", "") or r.get("status") != "1":
            continue
        code = bs_code.split(".")[1]
        name = r["code_name"]
        is_st = 1 if ("ST" in name.upper() or "退" in name) else 0
        rows.append((code, name, "sh" if code[0] == "6" else "sz",
                     is_st, None, now))
    upsert_rows(conn, "stock_list",
                ["code", "name", "exchange", "is_st", "industry", "updated"],
                rows)
    log.info("baostock stock_basic list stored: %d", len(rows))
    return [r[0] for r in rows]


def fetch_spot(conn, known_codes: list[str] | None = None):
    rows = []
    source = "em"
    try:
        rows = _fetch_spot_akshare()
    except Exception as e:  # noqa: BLE001
        log.warning("EM spot unavailable (%s); falling back to Tencent", e)
    seed = list(known_codes or [])
    if not rows and not seed:
        seed = _fetch_stock_list_baostock(conn)
    if not rows:
        source = "tencent"
        probe = ["600519"] if "600519" not in seed else []
        rows = _fetch_spot_tencent(probe + seed)
        if not rows:
            raise RuntimeError("all spot sources failed")
    now = str(today_cn())
    stocks, spots, todays = [], [], []
    for r in rows:
        code = r["code"]
        prefix = code[0]
        exchange = "sh" if prefix == "6" else ("bj" if prefix in "849" else "sz")
        name = r["name"]
        is_st = 1 if ("ST" in name.upper() or "退" in name) else 0
        stocks.append((code, name, exchange, is_st, None, now))
        spots.append((code, r["price"], r["pct_chg"], r["volume"], r["amount"],
                      r["turnover"], r["vol_ratio"], r["pe_dyn"], r["pb"],
                      r["total_mv"], r["circ_mv"], None, now))
        if r["price"] and r["open"] and r["high"] and r["low"]:
            todays.append((code, now, r["open"], r["high"], r["low"],
                           r["price"], r["volume"], r["amount"]))
    n1 = upsert_rows(conn, "stock_list",
                     ["code", "name", "exchange", "is_st", "industry", "updated"],
                     stocks)
    n2 = upsert_rows(conn, "spot",
                     ["code", "price", "pct_chg", "volume", "amount", "turnover",
                      "vol_ratio", "pe_dyn", "pb", "total_mv", "circ_mv",
                      "chg_60d", "updated"], spots)
    n3 = 0
    if todays and source == "tencent":
        pass  # bar append handled separately to avoid double-insert with EM path
    return {"stocks": n1, "quotes": n2, "source": source, "todays_bars": todays}


# ------------------------------------------------------- financial quarter

def recent_quarter_ends(ref: date, count=2) -> list[str]:
    ends = []
    y = ref.year
    while len(ends) < count:
        for md in ("-12-31", "-09-30", "-06-30", "-03-31"):
            qd = date.fromisoformat(f"{y}{md}")
            if qd < ref:
                ends.append(str(qd))
        y -= 1
    return ends[:count]


def fetch_fin_quarterly(conn, periods: list[str]) -> dict:
    got = {}
    for period in periods:
        try:
            df = robust(ak.stock_yjbb_em, date=period.replace("-", ""))
        except Exception as e:  # noqa: BLE001
            log.warning("yjbb %s unavailable: %s", period, e)
            continue
        df = df.rename(columns=YJBB_COLS)
        industry = {}
        if "所处行业" in df.columns:
            for c, ind in zip(df["code"], df["所处行业"]):
                industry[str(c).zfill(6)] = ind

        debt = {}
        try:
            zdf = robust(ak.stock_zcfz_em, date=period.replace("-", ""))
            for c, v in zip(zdf["股票代码"], zdf["资产负债率"]):
                fv = _f(v)
                if fv is not None:
                    debt[str(c).zfill(6)] = fv
        except Exception as e:  # noqa: BLE001
            log.warning("zcfz %s unavailable (debt_ratio NULL): %s", period, e)

        rows = []
        for r in df.itertuples(index=False):
            code = str(r.code).zfill(6)
            rows.append((code, period, _f(getattr(r, "eps", None)),
                         _f(getattr(r, "roe", None)),
                         _f(getattr(r, "gross_margin", None)),
                         _f(getattr(r, "profit_yoy", None)),
                         debt.get(code), str(today_cn())))
        upsert_rows(conn, "fin_q",
                    ["code", "period", "eps", "roe", "gross_margin",
                     "profit_yoy", "debt_ratio", "updated"], rows)
        if industry:
            up_rows = [(ind, str(today_cn()), c)
                       for c, ind in industry.items()]
            conn.executemany(
                "UPDATE stock_list SET industry=?, updated=? WHERE code=?",
                up_rows)
        got[period] = len(rows)
        time.sleep(random.uniform(1.0, 2.0))
    return got


# ------------------------------------------------------------- daily bars

_DAILY_COLS = ["code", "date", "open", "high", "low", "close",
               "volume", "amount"]


def fetch_daily_hist_em(code: str, start: str, end: str,
                        adjust: str = "") -> list[tuple]:
    """拉取单只日线。adjust='' 不复权(写 daily), adjust='qfq' 前复权(写 daily_qfq)。

    前复权价格连续、除权日不跳变，因子/回测直接消费可消除分红除权造成的
    虚假跳变（fireworks-sp 已知限制修复项）。
    """
    symbol = str(code).zfill(6)
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                            start_date=start.replace("-", ""),
                            end_date=end.replace("-", ""), adjust=adjust)
    if df is None or df.empty:
        return []

    def col(name):
        return name if name in df.columns else None

    d, o = col("日期"), col("开盘")
    h, l = col("最高"), col("最低")
    c, v = col("收盘"), col("成交量")
    a = col("成交额")
    return [(symbol, str(row[d])[:10], row[o], row[h], row[l], row[c],
             row[v], row[a]) for _, row in df.iterrows()]


# daily_qfq 的列与 daily 一致（前复权 OHLCV），独立表以保留不复权原始价。
_DAILY_QFQ_COLS = ["code", "date", "open", "high", "low", "close",
                   "volume", "amount"]


def fetch_daily_hist_em_qfq(code: str, start: str, end: str) -> list[tuple]:
    """前复权日线（写 daily_qfq）。薄封装 fetch_daily_hist_em(adjust='qfq')。"""
    return fetch_daily_hist_em(code, start, end, adjust="qfq")


def fetch_daily_hist_baostock(code: str, start: str, end: str) -> list[tuple]:
    import baostock as bs
    bs_code = ("sh." if code[0] == "6" else "sz.") + str(code).zfill(6)
    lg = bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            bs_code, "date,open,high,low,close,volume,amount,tradestatus",
            start_date=start, end_date=end, frequency="d", adjustflag="3")
        data = rs.get_data()
    finally:
        bs.logout()
    out = []
    for _, r in data.iterrows():
        if r.get("tradestatus") not in ("1", ""):
            continue
        out.append((code, r["date"], _f(r["open"]), _f(r["high"]),
                    _f(r["low"]), _f(r["close"]), _f(r["volume"]),
                    _f(r["amount"])))
    return out


def update_all_dailies_bootstrap(conn, codes: list[str], start: str,
                                 workers=6):
    """Threaded EM download; failures retried sequentially via baostock."""
    todo = [(c, start) for c in codes]
    done = fail = 0
    failed_codes: list[str] = []
    batch: list[tuple] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def flush():
        nonlocal batch
        upsert_rows(conn, "daily", _DAILY_COLS, batch)
        conn.commit()
        batch = []

    end = str(today_cn())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_daily_hist_em, c, st, end): (c, st)
                for c, st in todo}
        for fut in as_completed(futs):
            code, st = futs[fut]
            try:
                rows = fut.result()
                batch.extend(rows)
                done += 1
            except Exception:  # noqa: BLE001
                fail += 1
                failed_codes.append(code)
            if len(batch) >= 3000:
                flush()
            if done % 500 == 0:
                flush()
                log.info("bootstrap progress: %d/%d fail=%d", done, len(todo), fail)
    flush()

    if failed_codes:
        log.info("baostock fallback for %d failed codes", len(failed_codes))
        ok2 = 0
        for i, code in enumerate(failed_codes):
            try:
                rows = fetch_daily_hist_baostock(code, start, end)
                upsert_rows(conn, "daily", _DAILY_COLS, rows)
                conn.commit()
                ok2 += 1
            except Exception as e:  # noqa: BLE001
                log.debug("baostock %s failed: %s", code, e)
            if (i + 1) % 200 == 0:
                log.info("fallback progress: %d/%d", i + 1, len(failed_codes))
            time.sleep(random.uniform(0.05, 0.2))
        log.info("baostock fallback done: ok=%d/%d", ok2, len(failed_codes))
    return done, fail


def codes_missing_history(conn, codes: list[str], min_days=200):
    placeholders = ",".join("?" * len(codes))
    rows = conn.execute(
        f"SELECT code FROM daily WHERE code IN ({placeholders}) "
        "GROUP BY code HAVING COUNT(*) >= ?",
        (*codes, min_days)).fetchall()
    have = {r[0] for r in rows}
    return [c for c in codes if c not in have]


def bootstrap_baostock(conn, codes: list[str], start: str,
                       progress_every=200):
    """Single-session bulk download via baostock (no per-code login).
    Auto-relogins once on session death/hang and continues."""
    import socket

    import baostock as bs
    old_timeout = socket.getdefaulttimeout()
    if old_timeout is None or old_timeout > 20:
        socket.setdefaulttimeout(20)  # detect hung sockets
    end = str(today_cn())
    lg = bs.login()
    done = fail = 0
    batch: list[tuple] = []

    def query_one(bs_code):
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,tradestatus",
            start_date=start, end_date=end,
            frequency="d", adjustflag="3")
        return rs.get_data()

    try:
        for i, code in enumerate(codes):
            bs_code = ("sh." if code[0] == "6" else "sz.") + code
            data = None
            try:
                data = query_one(bs_code)
            except Exception:  # noqa: BLE001
                try:
                    bs.logout()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(random.uniform(2.0, 4.0))
                try:
                    lg = bs.login()
                    data = query_one(bs_code)
                except Exception as e:  # noqa: BLE001
                    fail += 1
                    log.debug("bs %s failed after relogin: %s", code, e)
                    continue
            for _, r in (data.iterrows() if data is not None else []):
                if r.get("tradestatus") not in ("1", ""):
                    continue
                batch.append((code, r["date"], _f(r["open"]),
                              _f(r["high"]), _f(r["low"]),
                              _f(r["close"]), _f(r["volume"]),
                              _f(r["amount"])))
            done += 1
            if len(batch) >= 3000:
                upsert_rows(conn, "daily", _DAILY_COLS, batch)
                conn.commit()
                batch = []
            if (i + 1) % progress_every == 0:
                if batch:
                    upsert_rows(conn, "daily", _DAILY_COLS, batch)
                    conn.commit()
                    batch = []
                log.info("baostock bootstrap: %d/%d (fail=%d)",
                         i + 1, len(codes), fail)
        if batch:
            upsert_rows(conn, "daily", _DAILY_COLS, batch)
            conn.commit()
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
        socket.setdefaulttimeout(old_timeout)
    return done, fail


def append_today_bars_from_snapshot(conn, todays_bars: list[tuple]):
    have = last_daily_dates(conn)
    fresh = [b for b in todays_bars if b[1] > have.get(b[0], "")]
    n = upsert_rows(conn, "daily", _DAILY_COLS, fresh)
    # 同步前复权当日 bar：仅对当日新增的股票抓 qfq 写入 daily_qfq，
    # 消除除权日跳变（增量窗口小、调用少，成本低）。
    if fresh:
        qfq_rows = []
        for b in fresh:
            code = b[0]
            try:
                q_rows = fetch_daily_hist_em_qfq(code, b[1], b[1])
                qfq_rows.extend([r for r in q_rows if r[1] == b[1]])
            except Exception:  # noqa: BLE001
                pass
        if qfq_rows:
            upsert_rows(conn, "daily_qfq", _DAILY_QFQ_COLS, qfq_rows)
    return n


def backfill_gaps(conn, codes: list[str], lookback_days=10):
    """Re-fetch recent window per code to heal gaps (e.g. cron missed days)."""
    import sqlite3
    end = str(today_cn())
    healed = 0
    for i, code in enumerate(codes):
        row = conn.execute(
            "SELECT MAX(date) FROM daily WHERE code=?", (code,)).fetchone()
        last = row[0] if row else None
        if last and (date.fromisoformat(end) -
                     date.fromisoformat(last)).days <= 1:
            continue
        start_d = date.fromisoformat(end)
        start = str(date.fromordinal(start_d.toordinal() - lookback_days))
        try:
            rows = fetch_daily_hist_baostock(code, start, end)
            fresh = [r for r in rows if not last or r[1] > last]
            if fresh:
                upsert_rows(conn, "daily", _DAILY_COLS, fresh)
                healed += len(fresh)
            # 增量缺口顺带补前复权
            qfq = fetch_daily_hist_em_qfq(code, start, end)
            qfq_fresh = [r for r in qfq if not last or r[1] > last]
            if qfq_fresh:
                upsert_rows(conn, "daily_qfq", _DAILY_QFQ_COLS, qfq_fresh)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(random.uniform(0.05, 0.15))
    conn.commit()
    return healed


def fetch_daily_hist_baostock_qfq(code: str, start: str, end: str) -> list[tuple]:
    """前复权日线（baostock adjustflag='1'，写 daily_qfq）。

    baostock 前复权价连续、除权日不跳变，且与 akshare EM 不通的环境（如受限
    服务器）仍可工作。返回 (code,date,open,high,low,close,volume,amount)。
    """
    import baostock as bs
    bs_code = ("sh." if code[0] == "6" else "sz.") + str(code).zfill(6)
    lg = bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            bs_code, "date,open,high,low,close,volume,amount,tradestatus",
            start_date=start, end_date=end, frequency="d", adjustflag="1")
        data = rs.get_data()
    finally:
        bs.logout()
    out = []
    for _, r in (data.iterrows() if data is not None else []):
        if r.get("tradestatus") not in ("1", ""):
            continue
        out.append((code, r["date"], _f(r["open"]), _f(r["high"]),
                    _f(r["low"]), _f(r["close"]), _f(r["volume"]),
                    _f(r["amount"])))
    return out


def refetch_qfq_all(conn, codes: list[str], start: str, workers=6,
                    source: str = "akshare"):
    """一次性全量重抓前复权日线写 daily_qfq（修复已知限制用）。

    不复权 daily 表保持不变。source='akshare' 用东方财富(并发,本机网络通);
    source='baostock' 用 baostock 前复权(adjustflag='1',单会话串行+自动重登,
    适配 akshare EM 不通的受限服务器)。失败静默跳过，不影响主流程。
    """
    if source == "baostock":
        return _refetch_qfq_baostock(conn, codes, start)
    done = fail = 0
    batch: list[tuple] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def flush():
        nonlocal batch
        if batch:
            upsert_rows(conn, "daily_qfq", _DAILY_QFQ_COLS, batch)
            conn.commit()
            batch = []

    end = str(today_cn())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_daily_hist_em_qfq, c, start, end): c
                for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                rows = fut.result()
                batch.extend(rows)
                done += 1
            except Exception:  # noqa: BLE001
                fail += 1
            if len(batch) >= 3000:
                flush()
            if done % 500 == 0:
                flush()
                log.info("qfq refetch progress: %d/%d fail=%d",
                         done, len(codes), fail)
    flush()
    return done, fail


def _refetch_qfq_baostock(conn, codes: list[str], start: str):
    """baostock 前复权全量重抓（单会话串行，断线自动重登）。"""
    import socket

    import baostock as bs
    old_timeout = socket.getdefaulttimeout()
    if old_timeout is None or old_timeout > 20:
        socket.setdefaulttimeout(20)
    end = str(today_cn())
    done = fail = 0
    batch: list[tuple] = []

    def flush():
        nonlocal batch
        if batch:
            upsert_rows(conn, "daily_qfq", _DAILY_QFQ_COLS, batch)
            conn.commit()
            batch = []

    lg = bs.login()
    for i, code in enumerate(codes):
        bs_code = ("sh." if code[0] == "6" else "sz.") + code
        data = None
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume,amount,tradestatus",
                start_date=start, end_date=end, frequency="d", adjustflag="1")
            data = rs.get_data()
        except Exception:  # noqa: BLE001
            try:
                bs.logout()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(random.uniform(2.0, 4.0))
            try:
                lg = bs.login()
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,open,high,low,close,volume,amount,tradestatus",
                    start_date=start, end_date=end, frequency="d", adjustflag="1")
                data = rs.get_data()
            except Exception:  # noqa: BLE001
                fail += 1
                log.debug("baostock qfq %s failed after relogin", code)
                continue
        for _, r in (data.iterrows() if data is not None else []):
            if r.get("tradestatus") not in ("1", ""):
                continue
            batch.append((code, r["date"], _f(r["open"]), _f(r["high"]),
                          _f(r["low"]), _f(r["close"]), _f(r["volume"]),
                          _f(r["amount"])))
        done += 1
        if len(batch) >= 3000:
            flush()
        if (i + 1) % 200 == 0:
            if batch:
                flush()
            log.info("baostock qfq refetch: %d/%d (fail=%d)",
                     i + 1, len(codes), fail)
    if batch:
        flush()
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass
    socket.setdefaulttimeout(old_timeout)
    return done, fail


# ------------------------------------------------------------------ index

def fetch_index_baostock(conn, bs_code: str, start: str):
    import baostock as bs

    row = conn.execute("SELECT MAX(date) FROM index_daily WHERE code=?",
                       (bs_code,)).fetchone()
    if row and row[0]:
        d = date.fromisoformat(row[0])
        start = str(date.fromordinal(d.toordinal() + 1))
    if start > str(today_cn()):
        return 0
    lg = bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            bs_code, "date,open,high,low,close,volume,amount",
            start_date=start, end_date=str(today_cn()),
            frequency="d", adjustflag="3")
        data = rs.get_data()
    finally:
        bs.logout()
    rows = [(bs_code, r["date"], _f(r["open"]), _f(r["high"]), _f(r["low"]),
             _f(r["close"]), _f(r["volume"]), _f(r["amount"]))
            for _, r in data.iterrows()]
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
    return upsert_rows(conn, "index_daily", cols, rows)


def universe_codes(conn) -> list[str]:
    rows = conn.execute(
        "SELECT s.code FROM stock_list s JOIN spot p ON p.code=s.code "
        "WHERE s.exchange IN ('sh','sz') AND s.is_st=0 "
        "AND p.price IS NOT NULL AND p.price > 0").fetchall()
    return [r[0] for r in rows]
