import numpy as np
import pandas as pd
import pytest

from fwsp.factor_factory import (FactorSpec, compute_factor, expand_recipes,
                                  load_base_recipes, _mp)


def _panel(T=40, C=15, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    codes = [f"{i:06d}" for i in range(C)]
    idx = pd.Index(dates, name="date")
    cols = pd.Index(codes, name="code")

    def mat(base, vol):
        return pd.DataFrame(rng.normal(base, vol, (T, C)), index=idx, columns=cols)

    return {
        "open": mat(100, 1),
        "high": mat(102, 1),
        "low": mat(98, 1),
        "close": mat(100, 1),
        "volume": mat(1e6, 1e5),
        "amount": mat(1e8, 1e6),
    }


class TestNoLookahead:
    """任何算子不得引入未来值：扰动未来单元格不应改变历史因子值。"""

    SPECS = [
        FactorSpec("sma", "t", "sma(close, 10)", {}, ""),
        FactorSpec("zscore", "t", "zscore(close, 10)", {}, ""),
        FactorSpec("std", "t", "std(returns(close,1), 10)", {}, ""),
        FactorSpec("ts_rank", "t", "ts_rank(close, 10)", {}, ""),
        FactorSpec("cs_rank", "t", "cs_rank(zscore(close,10))", {}, ""),
        FactorSpec("ref", "t", "ref(close, 3)", {}, ""),
        FactorSpec("corr", "t", "corr(returns(close,1), volume, 10)", {}, ""),
        FactorSpec("min", "t", "min(low, 10)", {}, ""),
        FactorSpec("max", "t", "max(high, 10)", {}, ""),
        FactorSpec("sign", "t", "sign(sub(close, sma(close,10)))", {}, ""),
    ]

    def test_future_perturbation_does_not_leak(self):
        panels = _panel(T=40, C=15, seed=1)
        k = 39  # 最后一行（未来）
        col = "000000"
        for spec in self.SPECS:
            f1 = compute_factor(panels, spec)
            # 扰动未来 close 的一个单元格
            perturbed = {k2: v.copy() for k2, v in panels.items()}
            perturbed["close"].iloc[k, 0] = perturbed["close"].iloc[k, 0] + 1e6
            f2 = compute_factor(perturbed, spec)
            # 第一行(远早于扰动)必须完全不变
            assert np.allclose(f1.iloc[0].to_numpy(), f2.iloc[0].to_numpy(),
                               equal_nan=True), f"{spec.id} 发生前视泄漏"


class TestExpandRecipes:
    def test_yaml_loads(self):
        base = load_base_recipes()
        assert isinstance(base, list) and len(base) >= 20
        for r in base:
            assert "id" in r and "category" in r and "expr" in r

    def test_count_ge_100(self):
        base = load_base_recipes()
        grid = {"windows": [5, 10, 20, 60, 120], "windows2": [20, 60, 120]}
        specs = expand_recipes(base, grid)
        assert len(specs) >= 100, f"仅生成 {len(specs)} 个候选"
        ids = [s.id for s in specs]
        assert len(ids) == len(set(ids)), "存在重复 id"

    def test_expr_substituted(self):
        base = [{"id": "x", "category": "t", "expr": "sma(close, {w})",
                 "scan": {"w": "windows"}, "desc": "d"}]
        specs = expand_recipes(base, {"windows": [5, 10]})
        assert specs[0].expr == "sma(close, 5)"
        assert specs[1].expr == "sma(close, 10)"


class TestComputeFactor:
    def test_shape_and_nan_safe(self):
        panels = _panel(T=30, C=10, seed=2)
        spec = FactorSpec("pp", "pp", "zscore(close, 5)", {}, "")
        out = compute_factor(panels, spec)
        assert out.shape == panels["close"].shape
        assert not out.isna().all().all()


class TestTsRollingOperators:
    """ts_min/ts_max/ts_mean 必须与 pandas rolling 基准逐点一致（补全算子后锁定）。"""

    def test_ts_min_max_mean_match_pandas(self):
        panels = _panel(T=60, C=10, seed=3)
        w = 5
        mp = _mp(w)
        cases = {
            "ts_min": lambda s: s.rolling(w, min_periods=mp).min(),
            "ts_max": lambda s: s.rolling(w, min_periods=mp).max(),
            "ts_mean": lambda s: s.rolling(w, min_periods=mp).mean(),
        }
        for op, fn in cases.items():
            out = compute_factor(panels, FactorSpec(f"t_{op}", "t",
                                                    f"{op}(close,5)", {}, ""))
            exp = fn(panels["close"])
            assert out.round(8).equals(exp.round(8)), f"{op} 与 pandas 基准不一致"
