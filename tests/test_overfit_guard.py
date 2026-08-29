import math

import numpy as np
import pandas as pd
import pytest

from fwsp.overfit_guard import (factor_health, oos_is_ratio, ratio_alert,
                                stability_check)


class TestOosIsRatio:
    def test_basic(self):
        assert math.isclose(oos_is_ratio(2.0, 1.0), 0.5)
        assert math.isclose(oos_is_ratio(2.0, 4.0), 2.0)

    def test_zero_is(self):
        assert math.isnan(oos_is_ratio(0.0, 1.0))

    def test_none(self):
        assert math.isnan(oos_is_ratio(None, 1.0))


class TestRatioAlert:
    def test_above_hi(self):
        alert, msg = ratio_alert(1.0, 6.0, hi=5.0, lo=0.3)
        assert alert and "过高" in msg

    def test_below_lo(self):
        alert, msg = ratio_alert(1.0, 0.2, hi=5.0, lo=0.3)
        assert alert and "失效" in msg

    def test_normal(self):
        alert, msg = ratio_alert(1.0, 2.0, hi=5.0, lo=0.3)
        assert not alert and "正常" in msg

    def test_boundary_exact(self):
        # 边界值不应触发
        assert not ratio_alert(1.0, 5.0, hi=5.0, lo=0.3)[0]
        assert not ratio_alert(1.0, 0.3, hi=5.0, lo=0.3)[0]


class TestStabilityCheck:
    def test_unstable_when_negative(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.02, 0.05, 300))  # 正 IC，应稳定
        r = stability_check(s, window=100, thr=0.0)
        assert not r["unstable"]

        s2 = pd.Series(rng.normal(-0.02, 0.05, 300))  # 负 IC，失稳
        r2 = stability_check(s2, window=100, thr=0.0)
        assert r2["unstable"]


class TestFactorHealth:
    def test_collects_alerts(self):
        rows = [
            {"code": "A", "is_icir": 1.0, "oos_icir": 6.0, "stability": 1.0,
             "selected": 1},
            {"code": "B", "is_icir": 1.0, "oos_icir": 0.2, "stability": 1.0,
             "selected": 1},
            {"code": "C", "is_icir": 1.0, "oos_icir": 2.0, "stability": -1.0,
             "selected": 1},
            {"code": "D", "is_icir": 1.0, "oos_icir": 2.0, "stability": 1.0,
             "selected": 1},
        ]
        w = factor_health(rows)
        assert len(w) == 3  # A(过高) B(过低) C(稳定性负)
        assert any("A" in x for x in w)
        assert any("B" in x for x in w)
        assert any("C" in x for x in w)
        assert not any("D" in x for x in w)
