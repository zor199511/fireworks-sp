import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwsp import db, pipeline  # noqa: E402
from fwsp.config import LOG_DIR  # noqa: E402
from datetime import datetime  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"update_{datetime.now():%Y%m%d}.log", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("fwsp.update")


def main():
    ap = argparse.ArgumentParser(description="fireworks-sp data updater")
    ap.add_argument("--full", action="store_true",
                    help="re-download full history from HISTORY_START")
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--source", choices=["auto", "baostock"], default="auto",
                    help="bootstrap history source (baostock bypasses EM)")
    ap.add_argument("--refetch-qfq", action="store_true",
                    help="仅全量重抓前复权日线写 daily_qfq（修复不复权已知限制，约15-20分钟）")
    args = ap.parse_args()

    with db.get_conn() as conn:
        db.init_schema(conn)
        if args.refetch_qfq:
            done, fail = pipeline.refetch_qfq(conn)
            log.info("qfq 重抓完成: ok=%d fail=%d", done, fail)
            return
        pipeline.update_all(conn, full=args.full, skip_daily=args.skip_daily,
                            source=args.source)

        counts = conn.execute(
            "SELECT (SELECT COUNT(*) FROM stock_list),"
            "(SELECT COUNT(*) FROM daily),"
            "(SELECT COUNT(DISTINCT code) FROM daily),"
            "(SELECT MAX(date) FROM daily)").fetchone()
        log.info("DB summary: stocks=%s daily_rows=%s daily_codes=%s "
                 "last_day=%s", *counts)


if __name__ == "__main__":
    main()
