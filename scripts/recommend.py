import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwsp import screener  # noqa: E402
from fwsp.config import LOG_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "recommend.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("fwsp.recommend")


def main():
    ap = argparse.ArgumentParser(description="run daily stock screen")
    ap.add_argument("--top", type=int, default=None)
    args = ap.parse_args()

    from fwsp.config import FILTERS
    top = screener.run_screen(top_n=args.top or FILTERS["top_n"])

    print()
    print(f"=== fireworks-sp daily picks ({len(top)}) ===")
    for r in top:
        mv_yi = (r.get("total_mv") or 0) / 1e8
        print(f"#{r.get('score',0):5.1f} {r['code']} {r['name'] or '':<6} "
              f"[{(r.get('industry') or '?')[:6]:<6}] "
              f"PE={r.get('pe_dyn') or 0:5.1f} ROE={r.get('roe') or 0:4.1f}% "
              f"市值={mv_yi:.0f}亿 | {'; '.join(r['reasons'])}")
    if not top:
        print("no candidates passed filters today")


if __name__ == "__main__":
    main()
