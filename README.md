# 🎆 fireworks-sp

> *fireworks-sp = fireworks **stock picker***——每日从 A 股里挑候选标的的系统。

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![data](https://img.shields.io/badge/data-akshare%20%7C%20baostock-orange)
![universe](https://img.shields.io/badge/A%E8%82%A1-5200%2B%20stocks-brightgreen)
![multifactor](https://img.shields.io/badge/multifactor-walk--forward%20%2B%20greedy-orange)

> 每日 A 股候选池生成器：**可解释评分 + 多因子回测 + 自动因子挖掘 + 推荐追踪 + 微信推送**的闭环。
> 从 5200+ 只股票里，用「基本面硬过滤 + 技术面反转打分」产出每日候选，并用一套**walk-forward 多因子框架**做量化验证与因子选择，每个交易日 17:30 推到你的微信。

## ✨ 为什么

A 股 5200+ 只股票，人工翻一遍不现实；免费行情软件要么广告满天，要么不解释「为什么选它」。
fireworks-sp 把筛选逻辑**写死成可读的规则**：每条推荐都附带打分理由（20 日回调 X%、RSI 超卖、MACD 底金叉…），多因子结论也给出**选中因子与样本外夏普**，不黑箱、可追溯。

## 🎯 特性

**一、反转策略（每日候选池）**
- **基本面硬过滤**：市值 ≥ 50 亿、PE∈(0,40]、ROE ≥ 8%、负债率 ≤ 65%（阈值见 `fwsp/config.py` 的 `FILTERS`）
- **技术面反转打分**：偏好「深度回调的优质股企稳」——回测验证反转策略优于追涨
  - 20 日回调 −35%~−12% → +30；RSI < 38 → +20；止跌企稳 → +20；长期趋势未破 → +15；底部 MACD 金叉 → +10（满分 100）
- **行业分散约束**：每行业最多 3 只，避免选股全挤在一个板块（`max_per_industry`）
- **面板向量化回测**：T+1 开盘买入、持有 10 日、−8% 止损、含交易费用，给出胜率/收益参考
- **推荐追踪**：持续统计历史推荐的实际表现（5 日胜率、平均收益）

**二、多因子框架（量化验证与因子选择）**
- **27 个价量因子**（`fwsp/factors.py`）：反转/动量（ret_5d…ret_120d、rev_5d/20d）、流动性（amihud、amt_ratio、vol_ratio）、趋势（ma5_ma20、close_ma20/60/120、macd 系列）、相对强度（hi20/60/120_dist、rsi_14）等
- **RankIC 因子检验**：对每个因子计算全期 RankIC 与 ICIR，筛掉无效因子
- **walk-forward 回测**（`fwsp/multifactor.py`）：滚动训练窗训练因子权重 → 信号日打分选 TopN → T+1 开盘买入 → 持有 N 日 / 固定止损 / 跟踪止损离场；支持**周度或月度调仓**
- **前向贪心因子挖掘**（`mine_factors`）：每步加入使 OOS 夏普最高的因子，直至无提升，防过拟合
- **时间变化质量面板**（`quality_panel`）：按财报披露日（`as_of`）前向填充 ROE/毛利率/债务率/市值/ST，质量门槛随历史变化而非用当前值
- **冻结 holdout 验证**（`scripts/factor_mine.py`）：挖掘在 IS 窗口（2024-06→2026-01）进行，验证用 2026-01 至今的**真样本外**，并与原反转策略对比
- **实时推荐**（`live_recommend`）：用最新训练窗在当日信号生成 Top10 + 各因子贡献明细
- **一键重挖**：`scripts/factor_mine.py` 自动挖掘 → holdout 对比 → 写 meta（`multifactor_selected` / `multifactor_summary`）

**三、部署与推送**
- **Streamlit 看板**：今日推荐 / 个股查询 / 策略回测 / **多因子策略** / 推荐追踪
  - 多因子策略页：生成实时推荐、重新挖掘因子、参数化回测（调仓频率 周/月、持有天数、止损%、跟踪止损%、TopN）
- **微信每日推送**：Server酱，同日去重省额度

## 🏗️ 架构

```
数据层   akshare(东财) / 腾讯行情(兜底) / baostock(K线+指数) / 历史财务回补 → SQLite
策略层   基本面硬过滤 + 技术面反转打分 + 行业分散
多因子层 27 价量因子 → RankIC → walk-forward 回测 → 前向贪心选因子 → 时间变化质量面板
回测层   面板向量化周度/月度回测: T+1开盘买、持有N日、-8%止损/跟踪止损、含费用
展示层   Streamlit 看板 (今日推荐/个股查询/策略回测/多因子策略/推荐追踪)
推送层   systemd timer 每交易日 17:30 → Server酱微信
```

## 📊 选股逻辑

**反转策略（每日候选池）**

| 维度 | 规则 |
|---|---|
| 市值 | total_mv ≥ 50 亿 |
| 估值 | 0 < PE ≤ 40 |
| 盈利质量 | ROE ≥ 8% |
| 财务风险 | 负债率 ≤ 65% |
| 流动性 | 20 日均成交额 ≥ 5000 万 |
| 技术面 | 反转打分（回调深度 + RSI + 企稳 + 趋势 + MACD），满分 100 |
| 行业 | 每行业 ≤ 3 只（top_n = 10） |

**多因子框架**

- 因子：`z = (因子 - 截面均值) / 截面标准差`，shift(1) 防前视；RankIC 用秩相关
- 调仓：walk-forward，滚动 252 日训练窗估计各因子权重（按 IC 符号×幅度），信号日对候选打分取 TopN
- 退出：持有 `horizon` 日，或触发固定止损（`-8%`）、或跟踪止损（峰值回撤 `trail%`）
- 质量门槛：时间变化——早期用当时可得的财报，避免现代值泄漏到历史

> **冻结 holdout 实况（2026-01 至今，真样本外）**：PIT universe + 真实财报披露日 + 宽基指数快照三重收口下，贪心挖出 4 因子 `amihud / hi60_dist / log_amt20 / gap_up`，总收益 **+22.7%**、夏普 **1.42**、回撤 −18.7%、胜率 48.2%；同期原反转策略 −6.9% / 夏普 −0.38。IS 窗口 +258% / 夏普 2.65。**以上均为回测，非实盘。**

## 🚀 安装与运行

```bash
git clone https://github.com/zor199511/fireworks-sp.git
cd fireworks-sp
uv sync                                         # 安装依赖
uv run python scripts/update_data.py --full     # 首次全量建库（约 15-20 分钟）
uv run python scripts/recommend.py              # 生成今日推荐
uv run streamlit run scripts/dashboard.py       # 看板 http://<host>:8501

# 多因子：历史财务回补（让质量门槛随时间变化）
uv run python fwsp/backfill_fundamentals.py     # 回补 2023Q1–2026Q2 ROE/利润/毛利率
uv run python fwsp/backfill_debt.py             # 回补历史债务率（sina 源，可断点续跑）

# 一键挖掘因子 + 冻结 holdout 验证 + 写 meta（约 15-20 分钟）
uv run python scripts/factor_mine.py
```

## 🔔 微信推送

推送经项目内 `scripts/notify.py` → Server酱。把 key 写到 `scripts/notify.json`（已被 `.gitignore` 忽略，不会入库）：

```json
{"serverchan_sendkey": "你的SCT..."}
```

`daily_pipeline.py` 每日筛选后自动推送，同日去重（省免费额度）。

## ⚙️ 常驻部署（systemd）

生产环境用用户级 systemd 常驻（本项目已部署于独立服务器，经 Tailscale 访问 `:8501`）：

- `fireworks-dashboard.service`：Streamlit 常驻 `:8501`，崩溃自拉起
- `fireworks-daily.timer`：Mon–Fri 17:30 触发 `fireworks-daily.service`（跑 `daily_pipeline.py`）

```bash
# 单元文件放 ~/.config/systemd/user/，然后：
loginctl enable-linger $USER
systemctl --user enable --now fireworks-dashboard.service fireworks-daily.timer
```

> 多因子挖掘与历史财务回补为按需任务，可在看板「多因子策略」页点按或在服务器手动运行，不计入每日 timer。

## 📈 数据规模

- Universe：5200+ 只 A 股（剔除 ST、北交/三板）
- 历史：2023-01-03 起日线（约 3 年，百万级 K 线）
- 历史财务：2023Q1–2026Q2 的 ROE / 净利润同比 / 毛利率 / 资产负债率（部分季度债务率为最新期快照）
- 数据源容灾：

| 数据 | 主源 | 兜底 |
|---|---|---|
| 行情快照 | 东财 push2 | 腾讯 qt.gtimg.cn |
| 历史K线 | 东财 kline | baostock 单会话批量 |
| 财务季度 | 东财 datacenter | sina 资产负债表（债务率） |
| 指数 | baostock | — |

> 东财对本机有限流，系统按「少调用、可降级」设计；增量更新直接切快照追加当日 OHLCV。

## ⚠️ 已知限制

- **复权**：数据层已支持前复权（`daily_qfq` 表），`load_panels` 默认吃 qfq，除权日跳变已消除；
  但**存量历史需重抓**才能落地（`python -c "from fwsp import collector, db; ..."` 或新增 CLI 触发
  `collector.refetch_qfq_all` 全量重抓 qfq，约 15–20 分钟，受 akshare 限流）。重抓前旧库自动回退不复权价。
- **幸存者偏差（已轻量收口 + PIT universe）**：质量门槛改为 point-in-time——
  基本面按财报 `as_of` 前向填充（披露前 False，披露后才 True），IC 计算不再
  「用今天财报评判历史」；同时叠加 **PIT universe 边界**——`quality_panel`
  末段按 `daily` 表每只 code 的首/末日叠加上市前/退市后剔除（`_pit_boundaries`
  向量化），新股在 first_date 前不入 universe、停牌/退市股在 last_date 后不
  入 universe。as_of 缺失时回退 `period + 30d`（财报披露日保守近似，旧库
  `fin_q.as_of` 多为 NULL 时仍能工作）。但**未引入成分股变动标记**（如纳入
  /调出沪深 300 的具体日期），结果仍可能偏乐观。
- **多因子 OOS 窗口仅约 8 个月**，且仍属回测；选中因子 OOS 与「全因子」OOS 接近，说明在正确质量门槛下因子筛选增量有限
- 债务率历史不完整：早期季度多为缺失（按「不违规」处理），最新期已密集；上游限流恢复后可重跑 `fwsp/backfill_debt.py` 补全
- 银行等高杠杆行业债务率天然 >85%，会被质量门槛排除（已知局限）

## 📜 License

MIT。

## ⚠️ 免责声明

输出为量化筛选候选池，**非投资建议**。股市有风险，入市需谨慎。

---

如果这套筛选帮你省了翻行情的时间，**点个 ⭐**。
