#!/usr/bin/env python3
"""回补 daily_qfq 表至 2015-01-01（baostock 前复权）"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwsp.db import get_conn, init_schema
from fwsp.collector import refetch_qfq_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_qfq")

START = "2015-01-01"


def main():
    with get_conn() as conn:
        init_schema(conn)
        codes = [r[0] for r in conn.execute("SELECT code FROM stock_list").fetchall()]

        if not codes:
            log.error("stock_list 为空")
            return

        log.info("使用 baostock 回补 %d 只股票 daily_qfq (起始 %s)", len(codes), START)

        # 使用 baostock 源，单会话串行，自动重登
        done, fail = refetch_qfq_all(conn, codes, START, source="baostock")

        log.info("完成: 成功=%d, 失败=%d", done, fail)


if __name__ == "__main__":
    main()
