from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "fireworks.db"

HISTORY_START = "2023-01-01"

BENCHMARK_CODE = "sh.000300"  # CSI 300

EXCHANGE_PREFIX_SH = ("6",)
EXCHANGE_EXCLUDED_PREFIX = ("8", "4", "92")  # BJ / NEEQ excluded in v1

QUARTER_ENDS = ("-03-31", "-06-30", "-09-30", "-12-31")

FILTERS = {
    "min_total_mv": 50e8,
    "max_pe": 40.0,
    "min_roe": 8.0,
    "max_debt": 65.0,
    "min_amount_20d": 50e6,
    "max_per_industry": 3,
    "top_n": 10,
}

SPOT_COLS = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "今开": "open",
    "最高": "high",
    "最低": "low",
    "涨跌幅": "pct_chg",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "量比": "vol_ratio",
    "市盈率-动态": "pe_dyn",
    "市净率": "pb",
    "总市值": "total_mv",
    "流通市值": "circ_mv",
    "60日涨跌幅": "chg_60d",
}

YJBB_COLS = {
    "股票代码": "code",
    "每股收益": "eps",
    "净利润-同比增长": "profit_yoy",
    "净资产收益率": "roe",
    "销售毛利率": "gross_margin",
}
