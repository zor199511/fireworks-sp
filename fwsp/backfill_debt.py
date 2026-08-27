import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

from fwsp.db import get_conn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill_debt")


def _prefix(code):
    return "sh" + code if code.startswith("6") else "sz" + code


def fetch_one(code):
    sym = _prefix(code)
    for attempt in range(4):
        try:
            df = ak.stock_financial_report_sina(stock=sym, symbol="资产负债表")
            break
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    else:
        return code, []
    if df is None or len(df) == 0:
        return code, []
    if "报告日" not in df.columns or "资产总计" not in df.columns \
            or "负债合计" not in df.columns:
        return code, []
    out = []
    for _, r in df.iterrows():
        rd = r.get("报告日")
        asset = r.get("资产总计")
        liab = r.get("负债合计")
        if rd is None or pd.isna(rd) or asset is None or pd.isna(asset):
            continue
        try:
            period = str(int(rd))
        except (ValueError, TypeError):
            continue
        if liab is None or pd.isna(liab) or asset == 0:
            continue
        try:
            debt = float(liab) / float(asset) * 100.0
        except (ValueError, TypeError):
            continue
        if not (0 <= debt <= 300):
            continue
        out.append((str(code).zfill(6), period, round(debt, 4)))
    return code, out


def main():
    with get_conn() as conn:
        codes = [r[0] for r in conn.execute(
            "SELECT code FROM stock_list ORDER BY code").fetchall()]
        filled = dict(conn.execute(
            "SELECT code, COUNT(*) FROM fin_q WHERE debt_ratio IS NOT NULL "
            "GROUP BY code").fetchall())
        total_by_code = dict(conn.execute(
            "SELECT code, COUNT(*) FROM fin_q GROUP BY code").fetchall())
    need = [c for c in codes
            if filled.get(c, 0) < total_by_code.get(c, 0)]
    log.info("待回补 %d 只（共 %d）", len(need), len(codes))

    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(fetch_one, c): c for c in need}
        for fut in as_completed(futs):
            code, rows = fut.result()
            results[code] = rows
            done += 1
            if done % 500 == 0:
                log.info("进度 %d/%d", done, len(need))

    total = 0
    t0 = time.time()
    with get_conn() as conn:
        for code, rows in results.items():
            for c, period, debt in rows:
                conn.execute(
                    "UPDATE fin_q SET debt_ratio=? WHERE code=? AND period=? "
                    "AND debt_ratio IS NULL",
                    (debt, c, period))
                total += 1
        conn.commit()
    log.info("写入 %d 条 debt_ratio，耗时 %.1fs", total, time.time() - t0)


if __name__ == "__main__":
    main()
