from fwsp import db


def _stock_row(code, name="测试股", exchange="sz", is_st=0):
    return (code, name, exchange, is_st, None, "2026-08-25")


def _daily_row(code, date, close=10.0):
    return (code, date, close, close * 1.01, close * 0.99, close,
            1_000_000.0, 1e7)


class TestSchema:
    def test_init_idempotent(self, mem_conn):
        db.init_schema(mem_conn)
        db.init_schema(mem_conn)
        tables = {r[0] for r in mem_execute(mem_conn,
                   "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"stock_list", "spot", "fin_q", "daily", "index_daily",
                "recommendations", "meta"} <= tables


def mem_execute(conn, sql):
    return conn.execute(sql).fetchall()


class TestUpsert:
    def test_upsert_inserts_and_replaces(self, mem_conn):
        cols = ["code", "name", "exchange", "is_st", "industry", "updated"]
        n = db.upsert_rows(mem_conn, "stock_list", cols,
                           [_stock_row("000001", "平安A")])
        assert n == 1
        db.upsert_rows(mem_conn, "stock_list", cols,
                       [_stock_row("000001", "平安银行")])
        row = mem_execute(mem_conn,
                          "SELECT name FROM stock_list WHERE code='000001'")
        assert row[0][0] == "平安银行"

    def test_empty_rows_noop(self, mem_conn):
        assert db.upsert_rows(mem_conn, "stock_list",
                              ["code"], []) == 0


class TestMeta:
    def test_set_get_roundtrip(self, mem_conn):
        db.set_meta(mem_conn, "last_update", "2026-08-25T17:30:00")
        assert db.get_meta(mem_conn, "last_update") == "2026-08-25T17:30:00"

    def test_get_missing_returns_none(self, mem_conn):
        assert db.get_meta(mem_conn, "nope") is None


class TestDailyQueries:
    def test_last_daily_dates(self, mem_conn):
        cols = ["code", "date", "open", "high", "low", "close",
                "volume", "amount"]
        db.upsert_rows(mem_conn, "daily", cols, [
            _daily_row("000001", "2026-08-20"),
            _daily_row("000001", "2026-08-21"),
            _daily_row("000002", "2026-08-19"),
        ])
        have = db.last_daily_dates(mem_conn)
        assert have["000001"] == "2026-08-21"
        assert have["000002"] == "2026-08-19"

    def test_universe_excludes_st_and_bj(self, mem_conn):
        cols = ["code", "name", "exchange", "is_st", "industry", "updated"]
        spot_cols = ["code", "price", "pct_chg", "volume", "amount",
                     "turnover", "vol_ratio", "pe_dyn", "pb", "total_mv",
                     "circ_mv", "chg_60d", "updated"]
        spot = lambda c: (c, 10.0, 1.0, 1e6, 1e7, 2.0, 1.0, 20.0, 2.0,
                          80e8, 60e8, 5.0, "2026-08-25")
        db.upsert_rows(mem_conn, "stock_list", cols, [
            _stock_row("600000", exchange="sh"),
            _stock_row("000001", "ST垃圾", is_st=1),
            _stock_row("830001", exchange="bj"),
        ])
        db.upsert_rows(mem_conn, "spot", spot_cols,
                       [spot("600000"), spot("000001"), spot("830001")])
        codes = set()
        from fwsp.collector import universe_codes
        codes = set(universe_codes(mem_conn))
        assert codes == {"600000"}
