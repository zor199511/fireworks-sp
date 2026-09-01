#!/usr/bin/env python3
"""回补 daily 表至 2015-01-01（baostock 历史日线）"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwsp.db import get_conn, init_schema
from fwsp.collector import bootstrap_baostock

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_daily")

START = "2015-01-01"


def main():
    with get_conn() as conn:
        init_schema(conn)

        # 获取全部股票代码
        codes = [r[0] for r in conn.execute("SELECT code FROM stock_list").fetchall()]
        if not codes:
            log.error("stock_list 为空，请先运行 collector 更新股票列表")
            return

        log.info("开始回补 %d 只股票 daily 数据，起始日期 %s", len(codes), START)
        t0 = time.time()

        done, fail = bootstrap_baostock(conn, codes, START, progress_every=500)

        elapsed = time.time() - t0
        log.info("回补完成: 成功=%d, 失败=%d, 耗时=%.1f分钟", done, fail, elapsed / 60)

        # 验证
        r = conn.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily").fetchone()
        log.info("daily 最终状态: %s ~ %s, %d 个交易日", r[0], r[1], r[2])

    # 回补 daily_qfq（前复权）- 需要单独的连接
    log.info("开始回补 daily_qfq（前复权）...")
    from fwsp.collector import fetch_daily_hist_em_qfq
    from fwsp.db import upsert_rows

    with get_conn() as conn:
        qfq_done = 0
        qfq_fail = 0
        batch = []

        for i, code in enumerate(codes):
            try:
                rows = fetch_daily_hist_em_qfq(code, START, "2023-12-31")
                for r in rows:
                    batch.append(r)
                qfq_done += 1
            except Exception as e:
                qfq_fail += 1
                if (i + 1) % 100 == 0:
                    log.warning("qfq %s 失败: %s", code, e)

            if len(batch) >= 5000:
                upsert_rows(conn, "daily_qfq",
                            ("code", "date", "open", "high", "low", "close", "volume", "amount"),
                            batch)
                conn.commit()
                batch = []

            if (i + 1) % 500 == 0:
                if batch:
                    upsert_rows(conn, "daily_qfq",
                                ("code", "date", "open", "high", "low", "close", "volume", "amount"),
                                batch)
                    conn.commit()
                    batch = []
                log.info("qfq 进度: %d/%d (fail=%d)", i + 1, len(codes), qfq_fail)

        if batch:
            upsert_rows(conn, "daily_qfq",
                        ("code", "date", "open", "high", "low", "close", "volume", "amount"),
                        batch)
            conn.commit()

        log.info("qfq 回补完成: 成功=%d, 失败=%d", qfq_done, qfq_fail)

        r2 = conn.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_qfq").fetchone()
        log.info("daily_qfq 最终状态: %s ~ %s, %d 个交易日", r2[0], r2[1], r2[2])


if __name__ == "__main__":
    main()
