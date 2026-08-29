"""交易成本常量（单一真相源）。

回测与多因子回测共用，避免在 backtest.py / multifactor.py 各定义一份。
买入含佣金+滑点；卖出额外含印花税（仅卖出收）。
"""
from __future__ import annotations

COMMISSION = 0.00025   # 券商佣金（双边）
STAMP_TAX = 0.0005    # 印花税（仅卖出）
SLIPPAGE = 0.001      # 滑点（双边估算）

COST_BUY = COMMISSION + SLIPPAGE
COST_SELL = COMMISSION + STAMP_TAX + SLIPPAGE
