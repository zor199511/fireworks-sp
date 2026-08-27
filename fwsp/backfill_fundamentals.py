import logging
import time
from datetime import date, datetime

import akshare as ak
import pandas as pd

from fwsp.db import get_conn, upsert_rows

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill")

COLS = ["code", "period", "eps", "roe", "gross_margin", "profit_yoy",
        "debt_ratio", "as_of", "updated"]

QUARTERS = ("0331", "0630", "0930", "1231")


def ensure_as_of(conn):
    conn.execute("ALTER TABLE fin_q ADD COLUMN as_of TEXT")
    conn.commit()
    log.info("已确保 fin_q.as_of 列存在")


def periods(start_year=2023):
    today = date.today()
    out = []
    for y in range(start_year, today.year + 1):
        for q in QUARTERS:
            d = datetime.strptime(f"{y}{q}", "%Y%m%d").date()
            if d > today:
                return out
            out.append(f"{y}{q}")
    return out


def backfill(conn, start_year=2023):
    ensure_as_of(conn)
    total = 0
    for p in periods(start_year):
        t0 = time.time()
        try:
            df = ak.stock_yjbb_em(date=p)
        except Exception as e:  # noqa: BLE001
            log.warning("周期 %s 获取失败: %s", p, e)
            continue
        if df is None or len(df) == 0:
            continue
        ren = {"股票代码": "code", "净资产收益率": "roe",
               "净利润-同比增长": "profit_yoy", "销售毛利率": "gross_margin",
               "最新公告日期": "as_of"}
        df = df.rename(columns=ren)
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["period"] = p
        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
        df["updated"] = datetime.now().isoformat(timespec="seconds")
        rows = []
        for _, r in df.iterrows():
            rows.append((r["code"], p, None,
                         _num(r.get("roe")), _num(r.get("gross_margin")),
                         _num(r.get("profit_yoy")), None,
                         None if pd.isna(r["as_of"]) else str(r["as_of"].date()),
                         r["updated"]))
        n = upsert_rows(conn, "fin_q", COLS, rows)
        conn.commit()
        total += n
        log.info("周期 %s: %d 行, 耗时 %.1fs (累计 %d)", p, n,
                 time.time() - t0, total)
    log.info("回补完成，共 %d 行", total)


def _num(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    with get_conn() as conn:
        backfill(conn, start_year=2023)
