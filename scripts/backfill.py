"""Resume/backfill full history for universe codes missing >=200 bars."""
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwsp import collector, db  # noqa: E402
from fwsp.config import HISTORY_START, LOG_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"backfill_{datetime.now():%Y%m%d}.log", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("fwsp.backfill")


def main():
    with db.get_conn() as conn:
        codes = collector.universe_codes(conn)
        missing = collector.codes_missing_history(conn, codes)
        log.info("universe=%d missing_history=%d", len(codes), len(missing))
        if not missing:
            print("nothing to backfill")
            return
        done, fail = collector.bootstrap_baostock(conn, missing,
                                                  HISTORY_START)
        log.info("backfill done: ok=%d fail=%d", done, fail)
        n = conn.execute(
            "SELECT COUNT(DISTINCT code) FROM daily").fetchone()[0]
        print(f"codes with history now: {n}")


if __name__ == "__main__":
    main()
