# 🎆 fireworks-sp

> *fireworks-sp = fireworks **stock picker***——每日从 A 股里挑候选标的的系统。

![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![data](https://img.shields.io/badge/data-akshare%20%7C%20baostock-orange)
![universe](https://img.shields.io/badge/A%E8%82%A1-5008%20stocks-brightgreen)

> 每日 A 股候选池生成器：**可解释评分 + 面板回测 + 推荐追踪 + 微信推送**的闭环。
> 从 5000+ 只股票里，用基本面硬过滤 + 技术面反转打分，挑出「深度回调的优质股」，每个交易日 17:30 推到你的微信。

## ✨ 为什么

A 股 5000+ 只股票，人工翻一遍不现实；免费行情软件要么广告满天，要么不解释「为什么选它」。
fireworks-sp 把筛选逻辑**写死成可读的规则**：每条推荐都附带打分理由（20 日回调 X%、RSI 超卖、MACD 底金叉…），不黑箱、可追溯。

## 🎯 特性

- **基本面硬过滤**：市值 ≥ 50 亿、PE∈(0,40]、ROE ≥ 8%、负债率 ≤ 65%（阈值见 `fwsp/config.py` 的 `FILTERS`）
- **技术面反转打分**：偏好「深度回调的优质股企稳」——回测验证反转策略优于追涨
  - 20 日回调 −35%~−12% → +30；RSI < 38 → +20；止跌企稳 → +20；长期趋势未破 → +15；底部 MACD 金叉 → +10（满分 100）
- **行业分散约束**：每行业最多 3 只，避免选股全挤在一个板块（`max_per_industry`）
- **面板向量化回测**：T+1 开盘买入、持有 10 日、−8% 止损、含交易费用，给出胜率/收益参考
- **推荐追踪**：持续统计历史推荐的实际表现（5 日胜率、平均收益）
- **微信每日推送**：Server酱，同日去重省额度
- **Streamlit 看板**：今日推荐 / 个股查询 / 策略回测 / 推荐追踪

## 🏗️ 架构

```
数据层   akshare(东财) / 腾讯行情(兜底) / baostock(K线+指数) → SQLite
策略层   基本面硬过滤 + 技术面反转打分 + 行业分散
回测层   面板向量化周度回测: T+1开盘买、持有10日、-8%止损、含费用
展示层   Streamlit 看板 (今日推荐/个股查询/策略回测/推荐追踪)
推送层   systemd timer 每交易日 17:30 → Server酱微信
```

## 📊 选股逻辑

| 维度 | 规则 |
|---|---|
| 市值 | total_mv ≥ 50 亿 |
| 估值 | 0 < PE ≤ 40 |
| 盈利质量 | ROE ≥ 8% |
| 财务风险 | 负债率 ≤ 65% |
| 流动性 | 20 日均成交额 ≥ 5000 万 |
| 技术面 | 反转打分（回调深度 + RSI + 企稳 + 趋势 + MACD），满分 100 |
| 行业 | 每行业 ≤ 3 只（top_n = 10） |

## 🚀 安装与运行

```bash
git clone https://github.com/zor199511/fireworks-sp.git
cd fireworks-sp
uv sync                                         # 安装依赖
uv run python scripts/update_data.py --full     # 首次全量建库（约 15-20 分钟）
uv run python scripts/recommend.py              # 生成今日推荐
uv run streamlit run scripts/dashboard.py       # 看板 http://<host>:8501
```

## 🔔 微信推送

推送经项目内 `scripts/notify.py` → Server酱。把 key 写到 `scripts/notify.json`：

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

## 📈 数据规模

- Universe：5008 只 A 股（剔除 ST、北交/三板）
- 历史：2023-01-01 起日线（约 3 年，百万级 K 线）
- 数据源容灾：

| 数据 | 主源 | 兜底 |
|---|---|---|
| 行情快照 | 东财 push2 | 腾讯 qt.gtimg.cn |
| 历史K线 | 东财 kline | baostock 单会话批量 |
| 财务季度 | 东财 datacenter | 缺失字段跳过 |
| 指数 | baostock | — |

> 东财对本机有限流，系统按「少调用、可降级」设计；增量更新直接切快照追加当日 OHLCV。

## ⚠️ 已知限制（v1）

- K 线为**不复权**价格：分红除权日指标小幅跳变；回测未复权
- 回测仅重放技术面打分（基本面用当前值），且用当前成分股池 → **幸存者偏差**，结果偏乐观
- PE 用静态阈值而非分位

## 📜 License

MIT。

## ⚠️ 免责声明

输出为量化筛选候选池，**非投资建议**。股市有风险，入市需谨慎。

---

如果这套筛选帮你省了翻行情的时间，**点个 ⭐**。
