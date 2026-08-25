"""QA 回归测试：apply_industry_cap 边界缺陷固化。

由 QA 编写，用于证明当前实现存在以下缺陷（修复前预期 FAIL）：
  D1. NaN industry（pd.read_sql 真实链路产物）被误当行业参与 cap，违反 AC#2
  D2. top_n<=0 时仍返回元素（break 判断在 append 之后）
  D3. cap<=0 时静默排除所有有行业股票（语义陷阱，期望显式处理或文档约定）

修复后本文件应全绿。业务代码修复归主程。
"""
import sqlite3

import pandas as pd

from fwsp.screener import apply_industry_cap


def _recs_from_null_industry(n_null: int) -> list[dict]:
    """模拟 hard_filter_rows -> pd.read_sql 路径下 NULL industry 的真实类型。

    注意必须用 混合字符串+NULL 列（与生产一致）：此时 pandas 把所有 NULL
    变成同一个 float('nan') 对象；全 NULL 列反而保持 None，无法复现缺陷。
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (code TEXT, industry TEXT)")
    rows = [(f"C{i}", f"行业{i % 2}") for i in range(8)]
    rows += [(f"N{i}", None) for i in range(n_null)]
    conn.executemany("INSERT INTO t VALUES (?,?)", rows)
    return pd.read_sql("SELECT code, industry FROM t", conn).to_dict("records")


class TestQaIndustryCapDefects:
    def test_d1_nan_industry_not_capped(self):
        """AC#2: industry=NULL 不受约束。生产路径（混合列 read_sql）产出
        共享的 float('nan')，当前实现把全部 nan 记入同一计数桶导致误 cap。"""
        recs = _recs_from_null_industry(n_null=6)
        scored = [{"code": r["code"], "score": 100 - i,
                   "industry": r["industry"]} for i, r in enumerate(recs)]
        # 8 只正常行业股(每行业4只) + 6 只 NULL 行业股；cap=3 淘汰 2 只正常股
        # 后剩 12 只有效候选，故 top_n 需 >=12 才能容纳全部 6 只 NULL 行业股
        top = apply_industry_cap(scored, top_n=12, cap=3)
        null_picked = [r["code"] for r in top if r["code"].startswith("N")]
        assert len(null_picked) == 6, \
            f"NULL-industry 被误 cap，仅选出 {len(null_picked)}/6: {null_picked}"

    def test_d2_top_n_zero_returns_empty(self):
        scored = [{"code": "A1", "score": 90, "industry": "银行"}]
        assert apply_industry_cap(scored, top_n=0, cap=3) == []

    def test_d2b_top_n_negative_returns_empty(self):
        scored = [{"code": "A1", "score": 90, "industry": "银行"}]
        assert apply_industry_cap(scored, top_n=-2, cap=3) == []

    def test_d3_cap_zero_means_unlimited(self):
        """cap=0 语义已由 TL 裁决为'不限制'：跳过行业计数，仅按 top_n 截断。"""
        scored = [{"code": f"A{i}", "score": 90 - i, "industry": "银行"}
                  for i in range(5)]
        top = apply_industry_cap(scored, top_n=3, cap=0)
        assert [r["code"] for r in top] == ["A0", "A1", "A2"]
