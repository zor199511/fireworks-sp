import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False,
                                   min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False,
                                      min_periods=n).mean()
    rs = gain / loss.where(loss > 0)
    out = 100 - 100 / (1 + rs)
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    return out.astype("float64")


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def compute_features(df: pd.DataFrame) -> dict | None:
    """df: columns [date,open,high,low,close,volume,amount] sorted by date."""
    if df is None or len(df) < 70:
        return None
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)

    ma5, ma10, ma20, ma60 = sma(c, 5), sma(c, 10), sma(c, 20), sma(c, 60)
    ma120 = sma(c, 120)
    dif, dea, _ = macd(c)
    rsi14 = rsi(c)
    vol_ma5, vol_ma20 = sma(v, 5), sma(v, 20)
    high20 = h.rolling(20, min_periods=20).max()
    high250 = h.rolling(250, min_periods=60).max()
    amt_ma20 = sma(df["amount"].astype(float), 20)
    ret_20d = c.pct_change(20) * 100
    ret_60d = c.pct_change(60) * 100

    i = -1
    last = df.iloc[i]

    def f(x):
        try:
            x = float(x)
            import math
            return None if math.isnan(x) else x
        except (TypeError, ValueError):
            return None

    feats = {
        "date": str(last["date"]),
        "close": float(last["close"]),
        "ma5": f(ma5.iloc[i]), "ma10": f(ma10.iloc[i]),
        "ma20": f(ma20.iloc[i]), "ma60": f(ma60.iloc[i]),
        "ma120": f(ma120.iloc[i]),
        "dif": f(dif.iloc[i]), "dea": f(dea.iloc[i]),
        "rsi": f(rsi14.iloc[i]),
        "vol": float(last["volume"]), "vol_ma5": f(vol_ma5.iloc[i]),
        "vol_ma20": f(vol_ma20.iloc[i]),
        "high20": f(high20.iloc[i]), "high250": f(high250.iloc[i]),
        "amount_ma20": f(amt_ma20.iloc[i]),
        "ret_20d": f(ret_20d.iloc[i]), "ret_60d": f(ret_60d.iloc[i]),
        "open_today": float(last["open"]),
        "chg_pct": None,
        "vol_prev": float(df["volume"].iloc[-2]) if len(df) > 1 else None,
    }
    prev_close = float(df["close"].iloc[-2])
    if prev_close:
        feats["chg_pct"] = (feats["close"] - prev_close) / prev_close * 100
    return feats
