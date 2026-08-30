"""拉取主要宽基指数的当前成分股快照写入 index_cons 表。

数据源:akshare
- 000300 沪深 300: index_stock_cons_csindex
- 399905 中证 500: index_detail_hist_cni (cni 国证指数)
- 399006 创业板指: index_detail_hist_cni
- 000905 中证 500 也可尝试 index_stock_cons_csindex

历史调样日期 akshare 不直接提供,仅记当前快照;作为"今日推荐 universe
过滤器"使用(为推荐页加成分股标记,提升可读性)。不替代回测 PIT(daily
首/末日),回测偏差仍存在但已记入 README 已知限制。
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import akshare as ak
import pandas as pd

from fwsp.db import get_conn, upsert_rows

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill_index")

# 中证指数(code->akshare fn);symbol 字段;source 用于统一 snapshot_date
SOURCES = [
    ("000300", "csi", "ak.index_stock_cons_csindex"),
    ("399905", "cni", "ak.index_detail_hist_cni"),
    ("399006", "cni", "ak.index_detail_hist_cni"),
    ("399001", "cni", "ak.index_detail_hist_cni"),  # 深证成指
]

COLS = ["index_code", "code", "name", "industry", "snapshot_date"]


def fetch_csi(symbol: str) -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol=symbol)
    # 列名: 成分券代码 成分券名称 (沪深 300/中证 500)
    df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
    df["industry"] = None
    return df[["code", "name", "industry"]]


def fetch_cni(symbol: str) -> pd.DataFrame:
    df = ak.index_detail_hist_cni(symbol=symbol)
    df = df.rename(columns={"样本代码": "code", "样本简称": "name", "所属行业": "industry"})
    return df[["code", "name", "industry"]]


def backfill(only: list[str] | None = None):
    snap_date = date.today().isoformat()
    fetched = []
    with get_conn() as conn:
        from fwsp.db import init_schema
        init_schema(conn)  # 确保新表（index_cons）已建
        for idx_code, src, _ in SOURCES:
            if only and idx_code not in only:
                continue
            try:
                if src == "csi":
                    df = fetch_csi(idx_code)
                else:
                    df = fetch_cni(idx_code)
            except Exception as e:  # noqa: BLE001
                log.warning("拉取 %s 失败: %s", idx_code, e)
                continue
            df["code"] = df["code"].astype(str).str.zfill(6)
            rows = [(idx_code, r["code"], r["name"], r.get("industry"), snap_date)
                    for _, r in df.iterrows()]
            # 删旧 snapshot
            conn.execute(
                "DELETE FROM index_cons WHERE index_code=? AND snapshot_date=?",
                (idx_code, snap_date))
            n = upsert_rows(conn, "index_cons", COLS, rows)
            conn.commit()
            log.info("指数 %s: %d 行 (snapshot=%s)", idx_code, n, snap_date)
            fetched.append((idx_code, n))
    return fetched


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="仅抓取指定指数,如 000300 399006")
    args = ap.parse_args()
    res = backfill(args.only)
    log.info("汇总: %s", res)
