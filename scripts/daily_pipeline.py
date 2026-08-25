import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwsp import pipeline  # noqa: E402
from fwsp.config import LOG_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"daily_{datetime.now():%Y%m%d}.log", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("fwsp.daily")


def main():
    recos, stats, sent = pipeline.daily_run(push=True)
    log.info("daily run done: picks=%d pushed=%s", len(recos), sent)
    for r in recos:
        print(f"#{r['rank']} {r['code']} {r['name']} score={r['score']:.0f}")


if __name__ == "__main__":
    main()
