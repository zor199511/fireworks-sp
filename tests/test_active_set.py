"""P0-1 三源合一：active_sets 单一真相源 / 单一写入者 / selected 不再漂移。"""
import sqlite3

from fwsp.db import (get_conn, init_schema, set_active_factors,
                     get_active_set)
from scripts.auto_evolve import auto_demote


def _seed(conn):
    init_schema(conn)
    # A 有新旧两行评估；B/C 各一行；A 最新行稳定性转负(衰减)，B 正常，C 正常
    rows = [
        ("A", "2026-01-01 00:00:00", 0.5, 0.4, 0.8, -1.0, 0.1, 0.5, 0),
        ("A", "2026-02-01 00:00:00", 0.5, 0.4, 0.8, -1.0, 0.1, -0.3, 0),
        ("B", "2026-02-01 00:00:00", 0.6, 0.5, 0.9, 1.0, 0.2, 0.6, 0),
        ("C", "2026-02-01 00:00:00", 0.4, 0.3, 0.7, 2.0, 0.1, 0.7, 0),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO factor_eval "
        "(code,run_at,is_icir,oos_icir,oos_is_ratio,stability,turnover,net_ir,selected) "
        "VALUES (?,?,?,?,?,?,?,?,?)", rows)


def test_set_active_factors_synced_selected():
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    run_at = set_active_factors(conn, ["A", "B"], run_at="2026-02-02 00:00:00",
                                 oos=0.7, source="auto_evolve")

    aset = get_active_set(conn, "auto_evolve")
    assert aset["factors"] == ["A", "B"]
    assert aset["run_at"] == run_at

    sel = {(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT code,run_at,selected FROM factor_eval").fetchall()}
    # 活跃因子最新行 selected=1
    assert sel[("A", "2026-02-01 00:00:00")] == 1
    assert sel[("B", "2026-02-01 00:00:00")] == 1
    # 非活跃(C) 与 历史行(A 旧行) selected=0 —— 不再漂移
    assert sel[("C", "2026-02-01 00:00:00")] == 0
    assert sel[("A", "2026-01-01 00:00:00")] == 0
    conn.close()


def test_auto_demote_removes_decayed_only():
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    set_active_factors(conn, ["A", "B", "C"], run_at="2026-02-02 00:00:00",
                         oos=0.7, source="auto_evolve")

    dem = auto_demote(conn)
    demoted = {c for c, _ in dem["demoted"]}
    # A 稳定性转负(-1.0) 应被降级；B/C 正常保留
    assert demoted == {"A"}
    assert set(dem["kept"]) == {"B", "C"}

    aset = get_active_set(conn, "auto_evolve")
    assert aset["factors"] == ["B", "C"]
    # 降级后 selected 同步：A 最新行归 0，B/C 保留
    sel = {(r[0], r[1]): r[2] for r in conn.execute("SELECT code,run_at,selected FROM factor_eval").fetchall()}
    assert sel[("A", "2026-02-01 00:00:00")] == 0
    assert sel[("B", "2026-02-01 00:00:00")] == 1
    conn.close()
