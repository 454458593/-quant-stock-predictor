# Quant Stock Predictor

一个面向 A 股和美股的轻量级量化研究项目：下载日线行情、生成技术特征、训练方向预测模型，并用严格的时间顺序进行样本外回测。

> 本项目仅用于研究与学习，不构成投资建议。概率预测不等于确定收益，实盘前请加入交易成本、滑点和更严格的走步验证。

## 功能

- 支持美股代码，例如 AAPL、NVDA
- 支持 6 位 A 股代码自动补全交易所后缀，例如 600519 转为 600519.SS
- 生成收益率、均线偏离、波动率、RSI、量价等特征
- 使用逻辑回归输出下一交易日上涨概率
- 按时间切分训练集和测试集，避免随机切分造成未来数据泄漏
- 输出预测明细、净值曲线和回测指标

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
quant-stock-predict AAPL --start 2018-01-01
quant-stock-predict 600519 --start 2018-01-01 --threshold 0.55
```

结果保存在 outputs 目录，包括 predictions.csv 和 equity_curve.csv。

## 下一步路线

1. 引入 AkShare/Tushare，补充 A 股复权行情、财务和资金面数据。
2. 增加多股票横截面选股与基准指数比较。
3. 使用 walk-forward 走步训练、手续费和滑点模型。
4. 增加财报、估值、13F 和事件驱动特征。
5. 纸面交易验证稳定后，再考虑对接券商。
